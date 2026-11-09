"""API keys k0 saves for the user: ``~/.k0ntrol/credentials``, mode 0600.

The file holds ``KEY=VALUE`` lines. :func:`load_credentials` copies the keys
that name a known provider credential into the environment, never over a
variable the user already set. Secrets never go in ``config.yaml``.
"""

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

from k0ntrol.providers import REGISTRY

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILE_MODE = 0o600


def credentials_path(home: Path) -> Path:
    return home / ".k0ntrol" / "credentials"


def known_credential_names() -> frozenset[str]:
    return frozenset(name for spec in REGISTRY.values() for name in spec.credentials)


def save_credential(home: Path, name: str, value: str) -> None:
    """Write ``name=value``, replacing an existing ``name`` line."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid credential name: {name!r}")
    if "\n" in value or "\r" in value:
        raise ValueError("a credential value cannot span lines")
    path = credentials_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [line for line in _read_lines(path) if _line_key(line) != name]
    _write_private(path, "\n".join([*kept, f"{name}={value}"]) + "\n")


def load_credentials(
    home: Path, environ: MutableMapping[str, str] = os.environ
) -> list[str]:
    """Export saved keys not already set; return the names set, not values."""
    known = known_credential_names()
    loaded = []
    for line in _read_lines(credentials_path(home)):
        name = _line_key(line)
        if name in known and not environ.get(name):
            environ[name] = line.partition("=")[2].strip()
            loaded.append(name)
    return loaded


def _read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line for line in text.splitlines() if _line_key(line) is not None]


def _line_key(line: str) -> str | None:
    """The key of a ``KEY=VALUE`` line; None for comments and blanks."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    return stripped.partition("=")[0].strip()


def _write_private(path: Path, text: str) -> None:
    """Create the file as 0600 before any secret reaches it."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), _FILE_MODE)
        handle.write(text)
