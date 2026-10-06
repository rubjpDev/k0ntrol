import re
from pathlib import Path


class MentionError(Exception):
    """Raised when an @file mention cannot be resolved."""


def mention_paths(text: str, root: Path) -> list[Path]:
    root_path = root.resolve()
    paths = []
    for match in re.finditer(r"@[^\s]+", text):
        token = match.group()[1:]
        paths.append(_resolve_mention(token, root_path))
    return paths


def _resolve_mention(token: str, root: Path) -> Path:
    candidate = Path(token)
    if candidate.is_absolute():
        raise MentionError(f"absolute mention path: {token}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise MentionError(f"mention path outside root: {token}") from error
    if not resolved.is_file():
        raise MentionError(f"mention is not an existing file: {token}")
    return resolved
