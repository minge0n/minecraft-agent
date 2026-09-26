import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from minecraft_rl.tick_control import (
    PlayerSnapshot,
    ProbeSnapshot,
    TickControlClient,
    TickControlError,
)

WARMUP_STEPS = 40
MOB_STEPS = 60
FORWARD_STEPS = 10
MIN_MOB_APPROACH_BLOCKS = 1.0
MOVE_EPSILON = 1e-6
SERVER_TICK_TRACE_LINE = "[Server thread/INFO] (mcbot) server tick end"


def wait_for_world(port: int, deadline: float) -> TickControlClient:
    while time.monotonic() < deadline:
        try:
            control = TickControlClient.connect(port)
        except OSError:
            time.sleep(1)
            continue
        try:
            status = control.status()
            if status.frozen and status.client_gated and not status.paused:
                return control
        except (OSError, TickControlError, ValueError):
            pass
        control.close()
        time.sleep(1)
    raise TimeoutError("Minecraft did not reach a frozen, gated, unpaused world")


def check_step(control: TickControlClient, expected_before: int, action="NOOP") -> int:
    step = control.step(action)
    if (step.tick_before, step.tick_after) != (expected_before, expected_before + 1):
        raise AssertionError(
            f"step {step.step_id}: {step.tick_before} -> {step.tick_after}, "
            f"expected {expected_before} -> {expected_before + 1}"
        )
    return step.tick_after


def horizontal_move(before: PlayerSnapshot, after: PlayerSnapshot) -> float:
    return abs(after.server_x - before.server_x) + abs(after.server_z - before.server_z)


def check_player_gate(control: TickControlClient, wait_seconds: float) -> dict:
    idle_before = control.debug_player()
    status_before = control.status()
    time.sleep(wait_seconds)
    idle_after = control.debug_player()
    status_after = control.status()
    if (
        idle_after != idle_before
        or status_after.client_ticks != status_before.client_ticks
    ):
        raise AssertionError(
            f"player or client advanced while idle: {idle_before} -> {idle_after}"
        )

    tick = status_after.game_time
    per_step = []
    for _ in range(3):
        before = control.debug_player()
        step = control.step()
        after = control.debug_player()
        deltas = {
            "game_time": after.game_time - before.game_time,
            "server_player_ticks": after.server_tick_count - before.server_tick_count,
            "server_play_time": after.server_play_time - before.server_play_time,
            "client_ticks": after.client_tick - before.client_tick,
            "client_player_ticks": after.client_tick_count - before.client_tick_count,
        }
        if (step.tick_before, step.tick_after) != (tick, tick + 1) or set(
            deltas.values()
        ) != {1}:
            raise AssertionError(f"one STEP must advance every clock by 1: {deltas}")
        tick = step.tick_after
        per_step.append(deltas)
    return {
        "player_idle_before": asdict(idle_before),
        "player_idle_after": asdict(idle_after),
        "per_step_clock_deltas": per_step,
    }


def check_action_timing(control: TickControlClient) -> dict:
    tick = control.status().game_time
    for _ in range(5):
        tick = check_step(control, tick)
    noop_before = control.debug_player()
    tick = check_step(control, tick)
    noop_after = control.debug_player()
    if horizontal_move(noop_before, noop_after) > MOVE_EPSILON:
        raise AssertionError(f"player moved on NOOP: {noop_before} -> {noop_after}")

    trace = [asdict(noop_after)]
    previous = noop_after
    for index in range(FORWARD_STEPS):
        tick = check_step(control, tick, "FORWARD")
        snapshot = control.debug_player()
        if index == 0 and horizontal_move(previous, snapshot) <= MOVE_EPSILON:
            raise AssertionError(
                f"FORWARD on step {tick} did not move the server player on that tick"
            )
        trace.append(asdict(snapshot))
        previous = snapshot

    tick = check_step(control, tick)
    tick = check_step(control, tick)
    rest_before = control.debug_player()
    tick = check_step(control, tick)
    rest_after = control.debug_player()
    return {
        "noop_move_blocks": horizontal_move(noop_before, noop_after),
        "first_forward_move_blocks": horizontal_move(
            PlayerSnapshot(**trace[0]), PlayerSnapshot(**trace[1])
        ),
        "forward_total_blocks": horizontal_move(
            PlayerSnapshot(**trace[0]), PlayerSnapshot(**trace[-1])
        ),
        "move_after_release_blocks": horizontal_move(rest_before, rest_after),
        "forward_trace": trace,
    }


def run_probes(
    control: TickControlClient, wait_seconds: float, steps: int
) -> dict[str, object]:
    tick = control.status().game_time
    for _ in range(WARMUP_STEPS):
        tick = check_step(control, tick)

    idle_before = control.status()
    time.sleep(wait_seconds)
    idle_after = control.status()
    if idle_after.game_time != idle_before.game_time:
        raise AssertionError(
            f"world time advanced while idle: {idle_before} -> {idle_after}"
        )

    single = control.step()
    if (single.tick_before, single.tick_after) != (tick, tick + 1):
        raise AssertionError(f"single step advanced {single}")
    tick = single.tick_after

    started = time.perf_counter()
    first = tick
    first_client = control.status().client_ticks
    for _ in range(steps):
        tick = check_step(control, tick)
    step_seconds = time.perf_counter() - started
    after_steps = control.status()
    if after_steps.game_time != tick or tick - first != steps:
        raise AssertionError(
            f"expected {steps} ticks from {first}, status {after_steps}"
        )
    client_advance = after_steps.client_ticks - first_client
    if client_advance != steps:
        raise AssertionError(f"expected {steps} client ticks, got {client_advance}")

    player = check_player_gate(control, wait_seconds)
    action = check_action_timing(control)

    tick = control.status().game_time
    control.debug_spawn()
    tick = check_step(control, tick)
    entity_idle_before = control.debug_probe()
    time.sleep(wait_seconds)
    entity_idle_after = control.debug_probe()
    if entity_idle_after != entity_idle_before:
        raise AssertionError(
            f"entities changed while idle: {entity_idle_before} -> {entity_idle_after}"
        )

    trajectory: list[ProbeSnapshot] = [entity_idle_after]
    for _ in range(MOB_STEPS):
        tick = check_step(control, tick)
        snapshot = control.debug_probe()
        if snapshot.game_time != tick:
            raise AssertionError(f"probe at {snapshot.game_time}, expected {tick}")
        trajectory.append(snapshot)

    fall = trajectory[0].armor_stand_y - trajectory[-1].armor_stand_y
    approach = trajectory[0].husk_player_distance - trajectory[-1].husk_player_distance
    if fall <= 0:
        raise AssertionError("armor stand did not fall while stepping")
    if approach < MIN_MOB_APPROACH_BLOCKS:
        raise AssertionError(f"husk approached only {approach:.3f} blocks")

    return {
        "idle_seconds": wait_seconds,
        "idle_status_before": asdict(idle_before),
        "idle_status_after": asdict(idle_after),
        "single_step": asdict(single),
        "multi_step": {
            "requested": steps,
            "advanced": after_steps.game_time - first,
            "client_ticks_advanced": after_steps.client_ticks - first_client,
            "seconds": step_seconds,
            "steps_per_second": steps / step_seconds,
        },
        "status_after_steps": asdict(after_steps),
        "player_gate": player,
        "action_timing": action,
        "entity_idle_before": asdict(entity_idle_before),
        "entity_idle_after": asdict(entity_idle_after),
        "armor_stand_fall_blocks": fall,
        "husk_approach_blocks": approach,
        "entity_trajectory": [asdict(snapshot) for snapshot in trajectory],
    }


def keep_open(control: TickControlClient, port: int) -> None:
    print("Probes passed; stepping ~20 ticks/s. Press ` in the window to toggle")
    print("human control of mouse and keyboard. Ctrl+C here to quit.")
    try:
        while True:
            if control.status().paused:
                raise AssertionError(
                    "singleplayer pause state appeared while observing"
                )
            control.step()
    except KeyboardInterrupt:
        control.close()
        with TickControlClient.connect(port) as fresh:
            fresh.quit()


def summarize(result: dict[str, object]) -> dict[str, object]:
    summary = {
        key: value for key, value in result.items() if key != "entity_trajectory"
    }
    action = summary.get("action_timing")
    if isinstance(action, dict):
        summary["action_timing"] = {
            key: value for key, value in action.items() if key != "forward_trace"
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Minecraft lockstep tick-gate probe"
    )
    parser.add_argument("--port", type=int, default=47123)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--wait", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="keep the client running and stepping for observation until Ctrl+C",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/minecraft-tick-gate/result.json")
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".minecraft.log")
    environment = os.environ | {
        "MCBOT_TICK_TRACE": "1",
        "MCBOT_TICK_PORT": str(args.port),
        "MCBOT_WORLD_SEED": str(args.seed),
    }
    result: dict[str, object] = {"seed": args.seed, "minecraft_log": str(log_path)}
    with log_path.open("w", encoding="utf-8") as log:
        game = subprocess.Popen(
            [str(root / "scripts" / "gradle"), "runClient"],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with wait_for_world(
                args.port, time.monotonic() + args.startup_timeout
            ) as control:
                result |= run_probes(control, args.wait, args.steps)
                if args.keep_open:
                    keep_open(control, args.port)
                else:
                    control.quit()
            result["passed"] = True
        except Exception as error:
            result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}
        finally:
            try:
                game.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(game.pid, signal.SIGTERM)
                game.wait(timeout=30)

    trace_seen = SERVER_TICK_TRACE_LINE in log_path.read_text(encoding="utf-8")
    result["server_thread_tick_trace_seen"] = trace_seen
    if not trace_seen:
        result |= {"passed": False, "error": "no integrated-server tick trace in log"}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summarize(result), indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
