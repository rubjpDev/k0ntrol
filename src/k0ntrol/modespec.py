from dataclasses import dataclass
from importlib.resources import files

import yaml

from k0ntrol.config import AGENTS


class ModeError(Exception):
    """Raised when a mode cannot be loaded."""


@dataclass(frozen=True)
class ModeSpec:
    name: str
    instructions: str
    model: str
    temperature: float | None
    one_shot: bool


def load_mode(name: str, cfg: dict) -> ModeSpec:
    if name not in AGENTS:
        raise ModeError(f"unknown mode: {name}")
    data = _read_mode(name)
    agent = cfg["agents"][name]
    return ModeSpec(
        name=data["name"],
        instructions=data["instructions"],
        model=agent["model"],
        temperature=agent.get("temperature"),
        one_shot=data["one_shot"],
    )


def load_all_modes(cfg: dict) -> dict[str, ModeSpec]:
    return {name: load_mode(name, cfg) for name in AGENTS}


def _read_mode(name: str) -> dict:
    path = files("k0ntrol") / "modes" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
