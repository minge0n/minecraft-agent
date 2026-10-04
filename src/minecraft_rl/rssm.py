"""Stage 2G recurrent state-space model (RSSM) of the T-maze.

See docs/stage2g.md. The state is split into a deterministic recurrent part h_t
and a stochastic categorical latent z_t:

    h_t = GRUCell(h_{t-1}, [z_{t-1}, onehot(a_{t-1})])     h_0 = 0
    prior      p(z_t | h_t)                                 before seeing o_t
    posterior  q(z_t | h_t, onehot(o_t))                    after seeing o_t
    heads on s_t = [h_t, z_t]: o_t, reward r_t and continuation c_t of the
    transition that led to s_t

Observations reach h only through z, so anything the model must remember about
an observation has to pass through the posterior latent and pay its KL price.
Imagination advances h and samples z from the prior, without decoding
observations.
"""

import math
from typing import Any

import torch
from torch import nn

from minecraft_rl.tmaze import Action, Observation
from minecraft_rl.world_model import (
    ACTIONS,
    OBSERVATIONS,
    Config,
    Episodes,
    Predictions,
    stack_predictions,
)

CUE_INTERVENTION_SEED = 7_000_000


def sample_one_hot(
    logits: torch.Tensor,
    generator: torch.Generator | None,
    straight_through: bool,
    mode: bool = False,
) -> torch.Tensor:
    """A one-hot sample (..., V, K) from independent categoricals with the given
    logits, by the Gumbel-max trick. With `straight_through`, the backward pass
    uses the gradient of the probabilities instead of the non-differentiable
    sample (value: sample, gradient: d softmax). With `mode`, the most likely
    class replaces the sample (an evaluation diagnostic, never used in
    training)."""
    if mode:
        index = logits.argmax(-1)
        return nn.functional.one_hot(index, logits.shape[-1]).to(logits.dtype)
    uniform = torch.rand(logits.shape, generator=generator).to(logits.device)
    gumbel = -torch.log(-torch.log(uniform.clamp(1e-20, 1.0 - 1e-7)))
    index = (logits + gumbel).argmax(-1)
    sample = nn.functional.one_hot(index, logits.shape[-1]).to(logits.dtype)
    if not straight_through:
        return sample
    probabilities = torch.softmax(logits, -1)
    return sample + probabilities - probabilities.detach()


def categorical_kl_per_variable(
    posterior_logits: torch.Tensor, prior_logits: torch.Tensor
) -> torch.Tensor:
    """KL(q || p) of each of the V latent variables, shape (..., V)."""
    log_q = torch.log_softmax(posterior_logits, -1)
    log_p = torch.log_softmax(prior_logits, -1)
    return (log_q.exp() * (log_q - log_p)).sum(-1)


def categorical_kl(
    posterior_logits: torch.Tensor, prior_logits: torch.Tensor
) -> torch.Tensor:
    """KL(q || p) summed over the V latent variables, shape (...)."""
    return categorical_kl_per_variable(posterior_logits, prior_logits).sum(-1)


def kl_loss(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    prior_scale: float,
    posterior_scale: float,
    free_nats: float,
) -> torch.Tensor:
    """Per-state KL loss (...): prior_scale KL(sg(q) || p) trains the prior
    toward the posterior, posterior_scale max(free_nats, KL(q || sg(p))) trains
    the posterior toward the prior. Both parts have the value of KL(q || p); with
    both scales 1 and free_nats 0 their gradient is that of KL(q || p). Below
    free_nats the posterior may carry information without penalty, while the
    prior still always learns to predict it."""
    prior_part = categorical_kl(posterior_logits.detach(), prior_logits)
    posterior_part = categorical_kl(posterior_logits, prior_logits.detach())
    return prior_scale * prior_part + posterior_scale * posterior_part.clamp(
        min=free_nats
    )


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy summed over the V latent variables, shape (...)."""
    log_q = torch.log_softmax(logits, -1)
    return -(log_q.exp() * log_q).sum((-1, -2))


def first_cue_states(observations: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mask (episodes, K + 1) of the first valid state of each episode whose
    observation is a cue. In the T-maze this is k = 0. In the signal T-maze it
    is the state where the drawn signal appears, if the episode reaches it."""
    cue = valid & (
        (observations == Observation.CUE_LEFT) | (observations == Observation.CUE_RIGHT)
    )
    first = cue & (cue.long().cumsum(1) == 1)
    return first


def mean_or_nan(values: torch.Tensor) -> float:
    return values.mean().item() if values.numel() else float("nan")


class RSSM(nn.Module):
    """Sizes: H = config.hidden, V x K = config.latent_variables x
    config.latent_classes, s_t = [h_t, z_t] of size H + V K."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hidden = config.hidden
        self.variables = config.latent_variables
        self.classes = config.latent_classes
        self.latent_size = self.variables * self.classes
        self.kl_prior_scale = config.kl_prior_scale
        self.kl_posterior_scale = config.kl_posterior_scale
        self.free_nats = config.free_nats
        self.prediction_samples = config.prediction_samples
        self.deterministic = False
        self.mode_latents = False
        self.policy_state_size = self.hidden + self.latent_size
        features = self.hidden + self.latent_size
        self.cell = nn.GRUCell(self.latent_size + ACTIONS, self.hidden)
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.latent_size),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden + OBSERVATIONS, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.latent_size),
        )
        self.observation_head = nn.Linear(features, OBSERVATIONS)
        self.reward_head = nn.Linear(features, 1)
        self.continue_head = nn.Linear(features, 1)

    def latent(
        self, logits: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        """A flat one-hot latent (..., V K) outside training: a sample, or the
        most likely class of each variable while `mode_latents` is set."""
        return sample_one_hot(logits, generator, False, self.mode_latents).flatten(-2)

    def prior_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h).unflatten(-1, (self.variables, self.classes))

    def posterior_logits(
        self, h: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        seen = nn.functional.one_hot(observations, OBSERVATIONS).float()
        joint = torch.cat([h, seen], -1)
        return self.posterior_net(joint).unflatten(-1, (self.variables, self.classes))

    def transition(
        self, h: torch.Tensor, z: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        taken = nn.functional.one_hot(actions, ACTIONS).float()
        return self.cell(torch.cat([z, taken], -1), h)

    def predict(self, features: torch.Tensor) -> Predictions:
        return Predictions(
            self.observation_head(features),
            self.reward_head(features).squeeze(-1),
            self.continue_head(features).squeeze(-1),
        )

    def filter(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        generator: torch.Generator | None,
        straight_through: bool = False,
        prior_at: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Posterior states for o_0 .. o_K (batch, K + 1) and a_0 .. a_{K-1}
        (batch, K): h (batch, K + 1, H), sampled z (batch, K + 1, V K), and the
        prior and posterior logits (batch, K + 1, V, K).

        `prior_at` (batch, K + 1) is an evaluation intervention: at the marked
        states, z is sampled from the prior instead of the posterior, so the
        observation there cannot enter the latent state."""
        batch, length = observations.shape
        h = torch.zeros(batch, self.hidden, device=observations.device)
        hs, zs, priors, posteriors = [], [], [], []
        for k in range(length):
            if k > 0:
                h = self.transition(h, zs[-1], actions[:, k - 1])
            prior = self.prior_logits(h)
            posterior = self.posterior_logits(h, observations[:, k])
            source = posterior
            if prior_at is not None:
                source = torch.where(prior_at[:, k, None, None], prior, posterior)
            if straight_through:
                z = sample_one_hot(source, generator, True).flatten(-2)
            else:
                z = self.latent(source, generator)
            hs.append(h)
            zs.append(z)
            priors.append(prior)
            posteriors.append(posterior)
        return {
            "h": torch.stack(hs, 1),
            "z": torch.stack(zs, 1),
            "prior": torch.stack(priors, 1),
            "posterior": torch.stack(posteriors, 1),
        }

    def _states(self, episodes: Episodes) -> tuple[torch.Tensor, torch.Tensor]:
        """Observations o_0 .. o_T (episodes, T + 1) of every state the episode
        visits and the validity of each state."""
        observations = torch.cat(
            [episodes.observations[:, :1], episodes.next_observations], 1
        )
        valid = torch.cat([episodes.mask[:, :1], episodes.mask], 1)
        return observations, valid

    def training_losses(
        self, episodes: Episodes, generator: torch.Generator | None
    ) -> dict[str, torch.Tensor]:
        """Negative evidence lower bound per valid state: reconstruction of o_k,
        reward and continuation of the transition into s_k (k >= 1), plus the KL
        between posterior q(z_k | h_k, o_k) and prior p(z_k | h_k).

        The KL term is `kl_loss`: separate prior and posterior parts, with free
        nats on the posterior part only. `kl` in the result is the plain
        KL(q || p) for reporting. `kl_below_free_nats` is the share of states
        whose posterior part gets no gradient because its KL is below
        `free_nats`.
        """
        observations, valid_states = self._states(episodes)
        filtered = self.filter(observations, episodes.actions, generator, True)
        predictions = self.predict(torch.cat([filtered["h"], filtered["z"]], -1))
        state_weight = valid_states.float()
        transition_weight = episodes.mask.float()
        states = state_weight.sum()
        transitions = transition_weight.sum()

        reconstruction = nn.functional.cross_entropy(
            predictions.observation_logits.transpose(1, 2),
            observations,
            reduction="none",
        )
        reward = (predictions.reward[:, 1:] - episodes.rewards) ** 2
        continuation = nn.functional.binary_cross_entropy_with_logits(
            predictions.continue_logits[:, 1:], episodes.continues, reduction="none"
        )
        posterior, prior = filtered["posterior"], filtered["prior"]
        plain_kl = categorical_kl(posterior, prior)
        regularizer = kl_loss(
            posterior,
            prior,
            self.kl_prior_scale,
            self.kl_posterior_scale,
            self.free_nats,
        )
        terms = {
            "reconstruction": (reconstruction * state_weight).sum() / states,
            "reward": (reward * transition_weight).sum() / transitions,
            "continuation": (continuation * transition_weight).sum() / transitions,
            "kl": (plain_kl * state_weight).sum() / states,
            "kl_below_free_nats": ((plain_kl < self.free_nats).float() * state_weight)
            .sum()
            .detach()
            / states,
        }
        total = (
            terms["reconstruction"]
            + terms["reward"]
            + terms["continuation"]
            + (regularizer * state_weight).sum() / states
        )
        return terms | {"total": total}

    def diagnostics(
        self, episodes: Episodes, generator: torch.Generator | None
    ) -> dict[str, Any]:
        """Latent statistics on valid states (nats). The cue step is the first
        state of each episode that shows a cue: k = 0 in the T-maze, the state
        where the drawn signal appears in the signal T-maze. Its observation is
        the only one the prior cannot predict. One bit of cue costs at least
        ln 2 = 0.693 nats of KL there.

        - kl_*: plain KL(q || p) before the free-nats threshold. The dynamics
          part KL(sg(q) || p) and the representation part KL(q || sg(p)) have
          the same value. They differ only in which network gets the gradient.
        - kl_loss_effective: the KL term that the loss uses, after the
          threshold. kl_posterior_part_effective is its posterior part,
          max(free_nats, KL).
        - below_free_nats_fraction: share of states whose KL is below
          `free_nats`. Those states give the posterior no KL gradient.
        - kl_per_variable_*: mean KL of each of the V variables.
          active_variables counts variables with a mean KL above 0.01 nats.
        - classes_used: number of (variable, class) pairs that are the most
          likely posterior class in at least one valid state, out of V K.
          class_perplexity_mean is exp(entropy) of the class frequency of each
          variable over all valid states, averaged over variables (1 to K).
        - prior_posterior_agreement*: share of (state, variable) pairs where
          the most likely prior class equals the most likely posterior class.
        - cue_coding_variables: variables whose most frequent posterior class
          at the cue step differs between the two cues, -1 without both cues.
        - cue_from_prior: the junction-turn reward sign accuracy when the
          latent at the first cue step, or at every state that shows a cue, is
          sampled from the prior instead of the posterior. Near 0.5 for every
          cue state means that the cue reaches the reward only through z. It
          uses its own generator, so it does not change the random numbers of
          the caller.
        """
        observations, valid = self._states(episodes)
        with torch.no_grad():
            filtered = self.filter(observations, episodes.actions, generator)
            terms = self.training_losses(episodes, generator)
        prior_logits, posterior_logits = filtered["prior"], filtered["posterior"]
        kl_variables = categorical_kl_per_variable(posterior_logits, prior_logits)
        kl = kl_variables.sum(-1)
        regularizer = kl_loss(
            posterior_logits,
            prior_logits,
            self.kl_prior_scale,
            self.kl_posterior_scale,
            self.free_nats,
        )
        prior_entropy = categorical_entropy(prior_logits)
        posterior_entropy = categorical_entropy(posterior_logits)
        cue = first_cue_states(observations, valid)
        later = valid & ~cue
        reconstruction = nn.functional.cross_entropy(
            self.predict(
                torch.cat([filtered["h"], filtered["z"]], -1)
            ).observation_logits.transpose(1, 2),
            observations,
            reduction="none",
        )
        cue_coding = -1
        has_cue = cue.any(1)
        cue_codes = posterior_logits[cue].argmax(-1)
        cue_seen = observations[cue]
        cue_left = cue_seen == Observation.CUE_LEFT
        cue_right = cue_seen == Observation.CUE_RIGHT
        if cue_left.any() and cue_right.any():
            left_code = cue_codes[cue_left].mode(0).values
            right_code = cue_codes[cue_right].mode(0).values
            cue_coding = int((left_code != right_code).sum())
        posterior_class = posterior_logits[valid].argmax(-1)
        winners = nn.functional.one_hot(posterior_class, self.classes)
        frequency = winners.float().mean(0)
        perplexity = torch.exp(-(frequency * frequency.clamp(min=1e-12).log()).sum(-1))
        agreement = prior_logits.argmax(-1) == posterior_logits.argmax(-1)
        return {
            "reconstruction_loss": terms["reconstruction"].item(),
            "reward_loss": terms["reward"].item(),
            "continuation_loss": terms["continuation"].item(),
            "reconstruction_loss_cue_step": mean_or_nan(reconstruction[cue]),
            "kl_mean": kl[valid].mean().item(),
            "kl_cue_step": mean_or_nan(kl[cue]),
            "kl_later_steps": mean_or_nan(kl[later]),
            "kl_later_steps_max": kl[later].max().item() if later.any() else 0.0,
            "kl_loss_effective": regularizer[valid].mean().item(),
            "kl_posterior_part_effective": kl[valid]
            .clamp(min=self.free_nats)
            .mean()
            .item(),
            "free_nats": self.free_nats,
            "below_free_nats_fraction": (kl[valid] < self.free_nats)
            .float()
            .mean()
            .item(),
            "below_free_nats_fraction_cue_step": mean_or_nan(
                (kl[cue] < self.free_nats).float()
            ),
            "below_free_nats_fraction_later_steps": mean_or_nan(
                (kl[later] < self.free_nats).float()
            ),
            "kl_per_variable_cue_step": kl_variables[cue].mean(0).tolist()
            if cue.any()
            else [],
            "kl_per_variable_all_steps": kl_variables[valid].mean(0).tolist(),
            "active_variables": int((kl_variables[valid].mean(0) > 0.01).sum()),
            "prior_entropy_cue_step": mean_or_nan(prior_entropy[cue]),
            "prior_entropy_later_steps": mean_or_nan(prior_entropy[later]),
            "posterior_entropy_mean": posterior_entropy[valid].mean().item(),
            "posterior_entropy_cue_step": mean_or_nan(posterior_entropy[cue]),
            "maximum_entropy": self.variables * math.log(self.classes),
            "classes_used": int(winners.amax(0).sum()),
            "classes_total": self.latent_size,
            "class_perplexity_mean": perplexity.mean().item(),
            "prior_posterior_agreement": agreement[valid].float().mean().item(),
            "prior_posterior_agreement_cue_step": mean_or_nan(agreement[cue].float()),
            "episodes_with_cue": int(has_cue.sum()),
            "cue_coding_variables": cue_coding,
            "cue_from_prior": self.cue_from_prior(episodes, observations, cue),
        }

    def cue_from_prior(
        self, episodes: Episodes, observations: torch.Tensor, cue: torch.Tensor
    ) -> dict[str, float]:
        """Junction-turn reward sign accuracy of the posterior states with the
        normal posterior latent, with a prior latent at the first cue step
        (`cue`), and with a prior latent at every state that shows a cue. An
        agent that turns or waits on the cue cell sees the cue again, so only
        the last variant removes every observation of the cue from z."""
        turn = (
            episodes.mask
            & (episodes.observations == Observation.JUNCTION)
            & (episodes.actions != Action.FORWARD)
        )
        if not turn.any():
            return {"turns": 0}
        every_cue = cue | (
            (observations == Observation.CUE_LEFT)
            | (observations == Observation.CUE_RIGHT)
        )
        result: dict[str, float] = {"turns": int(turn.sum())}
        variants = (
            ("posterior", None),
            ("cue_from_prior", cue),
            ("every_cue_from_prior", every_cue),
        )
        for name, prior_at in variants:
            with torch.no_grad():
                filtered = self.filter(
                    observations,
                    episodes.actions,
                    torch.Generator().manual_seed(CUE_INTERVENTION_SEED),
                    prior_at=prior_at,
                )
                reward = self.predict(
                    torch.cat([filtered["h"], filtered["z"]], -1)
                ).reward[:, 1:]
            correct = (reward[turn] > 0) == (episodes.rewards[turn] > 0)
            result[f"reward_sign_accuracy_{name}"] = correct.float().mean().item()
        return result

    def one_step(
        self, episodes: Episodes, generator: torch.Generator | None
    ) -> Predictions:
        """Prediction of every transition t (episodes, time) from the posterior
        state after o_t and a prior latent for the next state, averaged over
        `prediction_samples` latent samples: the observation as the mixture of the
        sampled categoricals, reward and continuation probability as means."""
        observations, _ = self._states(episodes)
        filtered = self.filter(observations, episodes.actions, generator)
        h = filtered["h"][:, 1:]
        prior = filtered["prior"][:, 1:]
        observation_probability = 0.0
        reward = 0.0
        continue_probability = 0.0
        for _ in range(1 if self.mode_latents else self.prediction_samples):
            z = self.latent(prior, generator)
            predictions = self.predict(torch.cat([h, z], -1))
            observation_probability = observation_probability + torch.softmax(
                predictions.observation_logits, -1
            )
            reward = reward + predictions.reward
            continue_probability = continue_probability + torch.sigmoid(
                predictions.continue_logits
            )
        samples = 1 if self.mode_latents else self.prediction_samples
        continue_probability = (continue_probability / samples).clamp(1e-6, 1 - 1e-6)
        return Predictions(
            torch.log(observation_probability / samples),
            reward / samples,
            torch.logit(continue_probability),
        )

    def open_loop(
        self,
        episodes: Episodes,
        start: int,
        horizon: int,
        generator: torch.Generator | None,
    ) -> Predictions:
        """Predictions (episodes, horizon) for transitions start .. start + horizon - 1
        from one latent trajectory: the posterior state after the real o_0 .. o_start,
        then h advanced with the real actions and z sampled from the prior."""
        horizon = min(horizon, episodes.actions.shape[1] - start)
        filtered = self.filter(
            episodes.observations[:, : start + 1],
            episodes.actions[:, :start],
            generator,
        )
        h, z = filtered["h"][:, -1], filtered["z"][:, -1]
        steps = []
        for offset in range(horizon):
            h = self.transition(h, z, episodes.actions[:, start + offset])
            z = self.latent(self.prior_logits(h), generator)
            steps.append(self.predict(torch.cat([h, z], -1)))
        return stack_predictions(steps)

    def policy_states(
        self, episodes: Episodes, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Posterior states s_t = [h_t, z_t] (episodes, time, H + V K) on which a_t
        is chosen."""
        filtered = self.filter(
            episodes.observations, episodes.actions[:, :-1], generator
        )
        return torch.cat([filtered["h"], filtered["z"]], -1)

    def initial_policy_states(
        self, observations: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        h = torch.zeros(observations.shape[0], self.hidden, device=observations.device)
        z = self.latent(self.posterior_logits(h, observations), generator)
        return torch.cat([h, z], -1)

    def observe_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        observations: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        h, z = states.split([self.hidden, self.latent_size], -1)
        h = self.transition(h, z, actions)
        z = self.latent(self.posterior_logits(h, observations), generator)
        return torch.cat([h, z], -1)

    def imagine_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The next latent state with z from the prior, the reward of the
        transition and its continuation probability. No observation is decoded."""
        h, z = states.split([self.hidden, self.latent_size], -1)
        h = self.transition(h, z, actions)
        z = self.latent(self.prior_logits(h), generator)
        following = torch.cat([h, z], -1)
        predictions = self.predict(following)
        return following, predictions.reward, torch.sigmoid(predictions.continue_logits)
