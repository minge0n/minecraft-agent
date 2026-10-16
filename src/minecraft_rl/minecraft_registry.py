"""The fixed categorical registries of the pinned Minecraft version
(docs/decisions/registry-vocabulary.md).

The environment reports every categorical observation value as a raw
registry index of Minecraft 26.3: a block, fluid or entity type on a ray,
an item in a slot, a status effect. `Registry` holds the name of every
index, read once from the running game (`DEBUG_REGISTRY`), and a
fingerprint of the whole list. A model records the fingerprint, so data or
a game whose registry differs is refused instead of read with the wrong
identities.

Names are for diagnostics and metadata only. A model never sees a name or
any feature derived from one: an index is only a row of a learned table.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_FORMAT = "minecraft-registry-v1"
# The registry of the pinned Minecraft 26.3 client, read by
# `scripts/minecraft-registry.py`. The file itself is generated under the
# ignored `runs/`; this fingerprint pins what it must contain.
PINNED_VERSION = "26.3"
PINNED_FINGERPRINT = "706b7869b5855014691c67e700d981dfcdfdbd54c26dd37db72964e1d0b2ffa6"
PINNED_SIZES = {"block": 1286, "fluid": 5, "entity": 161, "item": 1658, "effect": 41}
FAMILIES = ("block", "fluid", "entity", "item", "effect")
# Schema fields that count the raw indices of each family.
SCHEMA_SIZES = {
    "block": "block_types",
    "fluid": "fluid_types",
    "entity": "entity_types",
    "item": "item_types",
    "effect": "effect_types",
}


@dataclass(frozen=True)
class Registry:
    minecraft_version: str
    names: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        missing = set(FAMILIES) - set(self.names)
        if missing:
            raise ValueError(f"registry misses families {sorted(missing)}")

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the version and every name in index order."""
        text = json.dumps(
            {
                "minecraft_version": self.minecraft_version,
                "names": {f: list(self.names[f]) for f in FAMILIES},
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(text).hexdigest()

    def sizes(self) -> dict[str, int]:
        """Model categories per family. Observed effect values are the effect
        index plus 1, with 0 for an empty effect slot, so effects have one
        category more than registered effects. Item index 0 is air, the
        empty slot, and it is a registered item."""
        out = {family: len(self.names[family]) for family in FAMILIES}
        out["effect"] += 1
        return out

    def check_schema(self, schema: dict[str, Any]) -> None:
        """Refuse an observation schema whose registry sizes differ."""
        for family, field in SCHEMA_SIZES.items():
            if schema[field] != len(self.names[family]):
                raise ValueError(
                    f"registry mismatch: schema has {schema[field]} {family} types, "
                    f"the registry {len(self.names[family])}"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": REGISTRY_FORMAT,
            "minecraft_version": self.minecraft_version,
            "fingerprint": self.fingerprint,
            "names": {f: list(self.names[f]) for f in FAMILIES},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Registry":
        if data.get("format") != REGISTRY_FORMAT:
            raise ValueError(f"not a {REGISTRY_FORMAT} file")
        registry = cls(
            data["minecraft_version"],
            {f: tuple(data["names"][f]) for f in FAMILIES},
        )
        if registry.fingerprint != data["fingerprint"]:
            raise ValueError("registry fingerprint does not match its names")
        return registry

    @classmethod
    def load(cls, path: Path) -> "Registry":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def pinned(cls, path: Path) -> "Registry":
        """The registry file of the pinned version, refused if it differs."""
        registry = cls.load(path)
        if registry.fingerprint != PINNED_FINGERPRINT:
            raise ValueError(
                f"{path} is not the pinned Minecraft {PINNED_VERSION} registry"
            )
        return registry

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n", encoding="utf-8")
