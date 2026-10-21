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
    relative, body = parse_file_response(response)
    path = _resolve_output(relative, root_path)
    if target is not None and path != _resolve_target(target, root_path):
        raise CodeWriteError(f"FILE path does not match target: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not body.endswith("\n"):
        body += "\n"
    path.write_text(body, encoding="utf-8")
    return path


def parse_file_response(response: str) -> tuple[str, str]:
    text = _unwrap_outer_fence(_normalize_response(response))
    header, body = _split_file_header(text)
    if header is None:
        raise CodeWriteError("response must start with FILE <relative-path>")
    relative = _clean_file_path(header)
    if not relative:
        raise CodeWriteError("response must name a file")
    return relative, _strip_fences(body)


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


def _normalize_response(response: str) -> str:
    return response.lstrip("\ufeff").strip()


def _unwrap_outer_fence(text: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("```"):
        return text
    inner = lines[1:]
    if inner and inner[-1].strip() == "```":
        inner = inner[:-1]
    return "\n".join(inner)


def _split_file_header(text: str) -> tuple[str | None, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("FILE "):
            return stripped[5:], _body_after_file(lines, index)
    return None, text


def _body_after_file(lines: list[str], index: int) -> str:
    body_lines = lines[index + 1 :]
    if _file_line_is_fenced(lines, index):
        body_lines = _drop_trailing_fence(body_lines)
    return "\n".join(body_lines)


def _file_line_is_fenced(lines: list[str], index: int) -> bool:
    return index > 0 and lines[index - 1].strip().startswith("```")


def _drop_trailing_fence(lines: list[str]) -> list[str]:
    if lines and lines[-1].strip() == "```":
        return lines[:-1]
    return lines


def _clean_file_path(raw: str) -> str:
    path = raw.strip()
    if len(path) >= 2 and path[0] == "`" and path[-1] == "`":
        return path[1:-1].strip()
    return path


def _resolve_output(relative: str, root: Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise CodeWriteError(f"FILE path outside root: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CodeWriteError(f"FILE path outside root: {relative}") from error
    _reject_topology_write(resolved, root, relative)
    return resolved


def _reject_topology_write(resolved: Path, root: Path, relative: str) -> None:
    topology = (root / ".k0-mem" / "Project_topology").resolve()
    try:
        resolved.relative_to(topology)
    except ValueError:
        return
    raise CodeWriteError(f"FILE path is Project_topology: {relative}")


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
