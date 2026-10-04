from pathlib import Path


def should_delegate(path: Path, threshold_lines: int) -> bool:
    """Return whether a file exceeds the line threshold for delegation."""
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    return line_count > threshold_lines
