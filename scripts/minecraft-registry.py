"""Read the categorical registries from the installed Minecraft 26.3 client
and write them with their fingerprint (docs/decisions/registry-vocabulary.md).

Launches the project-owned client once, asks `DEBUG_REGISTRY` and `SCHEMA`,
checks that the sizes agree, and writes `--output`. No step runs.

    .venv/bin/python scripts/minecraft-registry.py
"""

import argparse
import json
from pathlib import Path

from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.minecraft_registry import PINNED_FINGERPRINT, PINNED_VERSION, Registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/registry/minecraft-26.3.json")
    )
    parser.add_argument("--port", type=int, default=47154)
    args = parser.parse_args()
    log = args.output.with_suffix(".minecraft.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with launched_client(args.port, 12345, log, 240.0, background=True) as client:
        names = client.privileged().registry()
        schema = client.request("SCHEMA")
    registry = Registry(PINNED_VERSION, {k: tuple(v) for k, v in names.items()})
    registry.check_schema(schema)
    registry.save(args.output)
    if registry.fingerprint != PINNED_FINGERPRINT:
        raise SystemExit(
            f"the installed registry is not the pinned one: {registry.fingerprint}"
        )
    print(
        json.dumps(
            {"fingerprint": registry.fingerprint, "sizes": registry.sizes()}, indent=1
        )
    )


if __name__ == "__main__":
    main()
