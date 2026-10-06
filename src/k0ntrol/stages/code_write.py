from collections.abc import Callable
from pathlib import Path

from k0ntrol.modespec import ModeSpec


class CodeWriteError(Exception):
    """Raised when a model response cannot safely write one file."""


Invoke = Callable[[ModeSpec, str], str]


def code_write(
    spec: str,
    *,
    root: Path,
    mode: ModeSpec,
    invoke: Invoke,
    target: Path | None,
    reference_text: str,
) -> Path:
    root_path = root.resolve()
    prompt = _build_prompt(spec, target, reference_text)
    response = invoke(mode, prompt)
    relative, body = _parse_response(response)
    path = _resolve_output(relative, root_path)
    if target is not None and path != _resolve_target(target, root_path):
        raise CodeWriteError(f"FILE path does not match target: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _build_prompt(spec: str, target: Path | None, reference_text: str) -> str:
    target_text = "<new file>"
    if target is not None and target.is_file():
        target_text = target.read_text(encoding="utf-8")
    prompt = (
        "Write one file. The first line is FILE <relative-path>. Then the file body.\n\n"
        f"{spec}\n\n"
        f"Target:\n{target_text}"
    )
    if reference_text:
        prompt += f"\n\nReference:\n{reference_text}"
    return prompt


def _parse_response(response: str) -> tuple[str, str]:
    first_line, separator, body = response.partition("\n")
    if not separator or not first_line.startswith("FILE "):
        raise CodeWriteError("response must start with FILE <relative-path>")
    relative = first_line[5:].strip()
    if not relative:
        raise CodeWriteError("response must name a file")
    return relative, _strip_fences(body)


def _resolve_output(relative: str, root: Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise CodeWriteError(f"FILE path outside root: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CodeWriteError(f"FILE path outside root: {relative}") from error
    return resolved


def _resolve_target(target: Path, root: Path) -> Path:
    resolved = target.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CodeWriteError(f"target outside root: {target}") from error
    return resolved


def _strip_fences(body: str) -> str:
    lines = body.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines) + ("\n" if lines else "")
    return body
