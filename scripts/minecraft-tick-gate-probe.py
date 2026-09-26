import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


class TickControl:
    def __init__(self, port: int) -> None:
        self.connection = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.reader = self.connection.makefile("r", encoding="utf-8")

    def request(self, command: str) -> list[str]:
        self.connection.sendall(f"v1 {command}\n".encode())
        fields = self.reader.readline().split()
        if fields[:2] != ["v1", command]:
            raise RuntimeError(f"{command} failed: {' '.join(fields)}")
        return fields[2:]

    def status(self) -> dict[str, object]:
        game_time, server_tick, frozen, paused = self.request("STATUS")
        return {
            "game_time": int(game_time),
            "server_tick": int(server_tick),
            "frozen": frozen == "true",
            "paused": paused == "true",
        }

    def step(self) -> tuple[int, int, int]:
        step_id, before, after = map(int, self.request("STEP"))
        return step_id, before, after

    def probe(self) -> dict[str, float]:
        game_time, stand_y, husk_x, husk_z, distance = self.request("DEBUG_PROBE")
        return {
            "game_time": int(game_time),
            "armor_stand_y": float(stand_y),
            "husk_x": float(husk_x),
            "husk_z": float(husk_z),
            "husk_player_distance": float(distance),
        }


def wait_for_world(port: int, deadline: float) -> TickControl:
    while time.monotonic() < deadline:
        try:
            control = TickControl(port)
            if control.status()["frozen"]:
                return control
        except (OSError, RuntimeError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("Minecraft did not reach a frozen integrated-server world")


def run_probes(
    control: TickControl, wait_seconds: float, steps: int
) -> dict[str, object]:
    for _ in range(40):
        control.step()

    idle_before = control.status()
    time.sleep(wait_seconds)
    idle_after = control.status()
    if idle_after["game_time"] != idle_before["game_time"]:
        raise AssertionError(
            f"World time advanced while idle: {idle_before} -> {idle_after}"
        )

    step_id, single_before, single_after = control.step()
    if (single_before, single_after) != (
        idle_after["game_time"],
        idle_after["game_time"] + 1,
    ):
        raise AssertionError(f"Single step advanced {single_before} -> {single_after}")

    started = time.perf_counter()
    last = single_after
    for _ in range(steps):
        step_id, before, after = control.step()
        if (before, after) != (last, last + 1):
            raise AssertionError(
                f"Step {step_id} advanced {before} -> {after}, expected from {last}"
            )
        last = after
    step_seconds = time.perf_counter() - started
    after_steps = control.status()
    if after_steps["game_time"] != last:
        raise AssertionError(f"World time moved after the last step: {after_steps}")

    control.request("DEBUG_SPAWN")
    control.step()
    entity_idle_before = control.probe()
    time.sleep(wait_seconds)
    entity_idle_after = control.probe()
    if entity_idle_after != entity_idle_before:
        raise AssertionError(
            f"Entities changed while idle: {entity_idle_before} -> {entity_idle_after}"
        )

    trajectory = [entity_idle_after]
    for _ in range(20):
        control.step()
        trajectory.append(control.probe())
    fell = trajectory[-1]["armor_stand_y"] < trajectory[0]["armor_stand_y"]
    moved = (
        trajectory[-1]["husk_player_distance"] != trajectory[0]["husk_player_distance"]
    )
    if not fell:
        raise AssertionError("Armor stand did not fall while stepping")
    if not moved:
        raise AssertionError("Husk did not move while stepping")

    return {
        "idle_seconds": wait_seconds,
        "idle_status_before": idle_before,
        "idle_status_after": idle_after,
        "single_step": {
            "step_id": step_id,
            "before": single_before,
            "after": single_after,
        },
        "multi_step": {
            "requested": steps,
            "advanced": last - single_after,
            "seconds": step_seconds,
            "steps_per_second": steps / step_seconds,
        },
        "status_after_steps": after_steps,
        "entity_idle_before": entity_idle_before,
        "entity_idle_after": entity_idle_after,
        "entity_trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Minecraft world tick-gate probe"
    )
    parser.add_argument("--port", type=int, default=47123)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--wait", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/minecraft-tick-gate/result.json")
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_name("minecraft.log")
    environment = os.environ | {
        "MCBOT_TICK_TRACE": "0",
        "MCBOT_TICK_PORT": str(args.port),
        "MCBOT_WORLD_SEED": str(args.seed),
    }
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
            control = wait_for_world(args.port, time.monotonic() + args.startup_timeout)
            result = run_probes(control, args.wait, args.steps)
            result |= {
                "passed": True,
                "seed": args.seed,
                "minecraft_log": str(log_path),
            }
            control.request("QUIT")
        except Exception as error:
            result = {
                "passed": False,
                "error": str(error),
                "minecraft_log": str(log_path),
            }
        finally:
            try:
                game.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(game.pid, signal.SIGTERM)
                game.wait(timeout=30)

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
