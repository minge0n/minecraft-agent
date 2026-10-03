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

import torch
from torch import nn

from minecraft_rl.tmaze import Observation
from minecraft_rl.world_model import (
    ACTIONS,
    OBSERVATIONS,
    Config,
    Episodes,
    Predictions,
    stack_predictions,
)


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


def categorical_kl(
    posterior_logits: torch.Tensor, prior_logits: torch.Tensor
) -> torch.Tensor:
    """KL(q || p) summed over the V latent variables, shape (...)."""
    log_q = torch.log_softmax(posterior_logits, -1)
    log_p = torch.log_softmax(prior_logits, -1)
    return (log_q.exp() * (log_q - log_p)).sum((-1, -2))


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
    ) -> dict[str, torch.Tensor]:
        """Posterior states for o_0 .. o_K (batch, K + 1) and a_0 .. a_{K-1}
        (batch, K): h (batch, K + 1, H), sampled z (batch, K + 1, V K), and the
        prior and posterior logits (batch, K + 1, V, K)."""
        batch, length = observations.shape
        h = torch.zeros(batch, self.hidden, device=observations.device)
        hs, zs, priors, posteriors = [], [], [], []
        for k in range(length):
            if k > 0:
                h = self.transition(h, zs[-1], actions[:, k - 1])
            prior = self.prior_logits(h)
            posterior = self.posterior_logits(h, observations[:, k])
            if straight_through:
                z = sample_one_hot(posterior, generator, True).flatten(-2)
            else:
                z = self.latent(posterior, generator)
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
    ) -> dict[str, float]:
        """Latent statistics on valid states (nats). The cue step is k = 0, the
        only state whose observation the prior cannot predict; one bit of cue
        costs at least ln 2 = 0.693 nats of KL there.

        - kl_*: plain KL(q || p). The dynamics part KL(sg(q) || p) and the
          representation part KL(q || sg(p)) have the same value; they differ
          only in which network gets the gradient.
        - below_free_nats_fraction: share of states whose KL is below
          `free_nats`. Those states give the posterior no KL gradient.
        - classes_used: number of (variable, class) pairs that are the most
          likely posterior class in at least one valid state, out of V K.
          Low values mean unused stochastic capacity.
        - cue_coding_variables: variables whose most frequent posterior class
          at the first step differs between the two cues; -1 when the first
          observation never shows both cues (the Stage 2H signal T-maze).
        """
        observations, valid = self._states(episodes)
        with torch.no_grad():
            filtered = self.filter(observations, episodes.actions, generator)
            terms = self.training_losses(episodes, generator)
        kl = categorical_kl(filtered["posterior"], filtered["prior"])
        prior_entropy = categorical_entropy(filtered["prior"])
        posterior_entropy = categorical_entropy(filtered["posterior"])
        cue = torch.zeros_like(valid)
        cue[:, 0] = valid[:, 0]
        later = valid & ~cue
        reconstruction = nn.functional.cross_entropy(
            self.predict(
                torch.cat([filtered["h"], filtered["z"]], -1)
            ).observation_logits.transpose(1, 2),
            observations,
            reduction="none",
        )
        codes = filtered["posterior"][:, 0].argmax(-1)
        cue_right = episodes.observations[:, 0] == Observation.CUE_RIGHT
        cue_left = episodes.observations[:, 0] == Observation.CUE_LEFT
        cue_coding = -1
        if cue_right.any() and cue_left.any():
            left_code = codes[cue_left].mode(0).values
            right_code = codes[cue_right].mode(0).values
            cue_coding = int((left_code != right_code).sum())
        winners = nn.functional.one_hot(
            filtered["posterior"][valid].argmax(-1), self.classes
        )
        return {
            "reconstruction_loss": terms["reconstruction"].item(),
            "reward_loss": terms["reward"].item(),
            "continuation_loss": terms["continuation"].item(),
            "reconstruction_loss_cue_step": reconstruction[cue].mean().item(),
            "kl_mean": kl[valid].mean().item(),
            "kl_cue_step": kl[cue].mean().item(),
            "kl_later_steps": kl[later].mean().item(),
            "kl_later_steps_max": kl[later].max().item(),
            "free_nats": self.free_nats,
            "below_free_nats_fraction": (kl[valid] < self.free_nats)
            .float()
            .mean()
            .item(),
            "below_free_nats_fraction_cue_step": (kl[cue] < self.free_nats)
            .float()
            .mean()
            .item(),
            "prior_entropy_cue_step": prior_entropy[cue].mean().item(),
            "prior_entropy_later_steps": prior_entropy[later].mean().item(),
            "posterior_entropy_mean": posterior_entropy[valid].mean().item(),
            "posterior_entropy_cue_step": posterior_entropy[cue].mean().item(),
            "maximum_entropy": self.variables * math.log(self.classes),
            "classes_used": int(winners.amax(0).sum()),
            "classes_total": self.latent_size,
            "cue_coding_variables": cue_coding,
        }

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
