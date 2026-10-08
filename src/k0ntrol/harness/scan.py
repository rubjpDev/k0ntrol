import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol.backends.invoke import set_event_context
from k0ntrol.modespec import ModeSpec, load_mode


Invoke = Callable[[ModeSpec, str], str]
BulkRead = Callable[..., str]

EMPTY_NOTE = {"desc": "", "hace": "", "gotchas": ""}
SCAN_QUESTION = (
    "For each file, write structured bullets in this exact shape:\n"
    "- FILE <path>\n"
    "  - desc: <one sentence>\n"
    "  - hace: <what the file does>\n"
    "  - gotchas: <gotchas, or none>"
)


class ScanError(Exception):
    """Raised when scan cannot list git-tracked files."""


def rescan_changed(
    *,
    root: Path,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> str:
    """Re-scan changed files when code memory already exists."""
    if not _graph_path(root, root.name).is_file():
        return "missing"
    return run_scan(
        root=root,
        config=config,
        bulk_read=bulk_read,
        invoke=invoke,
        read_line=read_line,
        write=write,
    )


def run_scan(
    *,
    root: Path,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> str:
    """Walk git-tracked text files and write file-level code memory."""
    set_event_context(run=uuid4().hex, step="scan")
    repo = root.name
    tracked = _git_tracked_files(root)
    cached = _nodes_by_path(_graph_path(root, repo))
    fresh, reused = _partition_by_hash(tracked, root, cached)
    batches = _batch_paths(fresh, config["threshold_lines"])
    if not _confirm_scan(len(batches), config["scan_confirm_over"], read_line, write):
        return "cancelled"
    notes = _describe_batches(batches, root, repo, config, bulk_read, invoke)
    _write_memory(root, repo, fresh, notes, _merge_nodes(repo, root, tracked, notes, reused))
    return "scanned"


def _confirm_scan(
    calls: int,
    limit: int,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> bool:
    if calls <= limit:
        return True
    write(f"scan needs {calls} model calls (limit {limit}). continue? y/n")
    while True:
        answer = read_line().strip().lower()
        if answer in {"y", "n"}:
            return answer == "y"
        write("y or n")


def _git_tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise ScanError(f"unable to list git files: {error}") from error
    if result.returncode != 0:
        err = result.stderr.decode().strip() or "git ls-files failed"
        raise ScanError(err)
    relative_paths = [item.decode() for item in result.stdout.split(b"\0") if item]
    files = []
    for relative in relative_paths:
        path = (root / relative).resolve()
        if _is_text_file(path):
            files.append(path)
    return files


def _is_text_file(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if b"\0" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _graph_path(root: Path, repo: str) -> Path:
    return root / ".k0-mem" / "Project_context" / repo / "graph.json"


def _nodes_by_path(graph_path: Path) -> dict[str, dict]:
    if not graph_path.is_file():
        return {}
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return {}
    return {
        node["path"]: node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("path"), str)
    }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partition_by_hash(
    paths: list[Path],
    root: Path,
    cached: dict[str, dict],
) -> tuple[list[Path], dict[str, dict]]:
    fresh = []
    reused: dict[str, dict] = {}
    for path in paths:
        relative = _relative(path, root)
        node = cached.get(relative)
        if isinstance(node, dict) and node.get("hash") == _file_digest(path):
            reused[relative] = node
        else:
            fresh.append(path)
    return fresh, reused


def _merge_nodes(
    repo: str,
    root: Path,
    paths: list[Path],
    notes: dict[str, dict[str, str]],
    reused: dict[str, dict],
) -> list[dict]:
    nodes = []
    for path in paths:
        relative = _relative(path, root)
        if relative in reused:
            nodes.append(reused[relative])
            continue
        note = notes.get(relative, EMPTY_NOTE)
        nodes.append(
            {
                "id": f"{repo}:file:{relative}",
                "kind": "File",
                "path": relative,
                "desc": note["desc"],
                "hash": _file_digest(path),
            }
        )
    return nodes


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _batch_paths(paths: list[Path], threshold_lines: int) -> list[list[Path]]:
    batches: list[list[Path]] = []
    current: list[Path] = []
    total = 0
    for path in paths:
        lines = _line_count(path)
        if current and total + lines > threshold_lines:
            batches.append(current)
            current = []
            total = 0
        current.append(path)
        total += lines
    if current:
        batches.append(current)
    return batches


def _describe_batches(
    batches: list[list[Path]],
    root: Path,
    repo: str,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
) -> dict[str, dict[str, str]]:
    notes: dict[str, dict[str, str]] = {}
    mode = load_mode("bulk_reader", config)
    for batch in batches:
        raw = bulk_read(
            SCAN_QUESTION,
            batch,
            root=root,
            repo=repo,
            mode=mode,
            invoke=invoke,
        )
        notes.update(_parse_notes(raw))
    return notes


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root.resolve()).as_posix()


def _parse_notes(text: str) -> dict[str, dict[str, str]]:
    notes: dict[str, dict[str, str]] = {}
    current = ""
    for line in text.splitlines():
        current = _note_line(notes, current, line)
    return notes


def _note_line(notes: dict[str, dict[str, str]], current: str, line: str) -> str:
    if line.startswith("- FILE "):
        current = line[len("- FILE ") :].strip()
        notes[current] = dict(EMPTY_NOTE)
        return current
    if current:
        _note_field(notes[current], line)
    return current


def _note_field(note: dict[str, str], line: str) -> None:
    for key in ("desc", "hace", "gotchas"):
        prefix = f"  - {key}: "
        if line.startswith(prefix):
            note[key] = line[len(prefix) :]
            return


def _write_memory(
    root: Path,
    repo: str,
    described: list[Path],
    notes: dict[str, dict[str, str]],
    nodes: list[dict],
) -> None:
    files_dir = root / ".k0-mem" / "Project_context" / repo / "files"
    keep = {
        node["path"]
        for node in nodes
        if node.get("kind") == "File" and isinstance(node.get("path"), str)
    }
    _remove_stale_file_markdown(files_dir, keep)
    for path in described:
        relative = _relative(path, root)
        note = notes.get(relative, EMPTY_NOTE)
        md_path = files_dir / f"{relative}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_file_markdown(relative, note), encoding="utf-8")
    graph_path = _graph_path(root, repo)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps({"nodes": nodes}, indent=2) + "\n",
        encoding="utf-8",
    )


def _remove_stale_file_markdown(files_dir: Path, keep: set[str]) -> None:
    if not files_dir.is_dir():
        return
    for md_path in files_dir.rglob("*.md"):
        relative = md_path.relative_to(files_dir).as_posix()
        source = relative.removesuffix(".md")
        if source not in keep:
            md_path.unlink()


def _file_markdown(relative: str, note: dict[str, str]) -> str:
    hace = note["hace"] or note["desc"]
    gotchas = note["gotchas"] or "none"
    return f"# {relative}\n\n## hace\n{hace}\n\n## gotchas\n{gotchas}\n"
