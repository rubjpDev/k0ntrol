"""Hash-based code-graph scan over git-tracked text files.

The scan writes the code graph (`graph.json`) plus file notes under
`.k0-mem/Project_context/<repo>/`. When ``invoke`` is given — the
one-shot ``bulk_read`` helper call — changed files go out in
``threshold_lines`` batches and their parsed notes land in the graph's
``desc`` fields and the note markdown the FTS index covers. Without
``invoke`` the scan still refreshes nodes and edges, with empty notes.
"""

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from k0ntrol.memory.python_extract import (
    extract_python,
    module_name,
    resolve_calls,
    resolve_imports,
)
from k0ntrol.tools.bulk_read import bulk_read

Invoke = Callable[[str], str]

EMPTY_NOTE = {"desc": "", "hace": "", "gotchas": ""}
DEFAULT_THRESHOLD_LINES = 350
SCAN_QUESTION = (
    "For each file, write structured bullets in this exact shape:\n"
    "- FILE <path>\n"
    "  - desc: <one sentence>\n"
    "  - hace: <what the file does>\n"
    "  - gotchas: <gotchas, or none>\n"
    "- FUNCTION <module.qualname>\n"
    "  - desc: <one sentence>\n"
    "  - hace: <what the function does>\n"
    "  - gotchas: <gotchas, or none>\n"
    "- CLASS <module.qualname>\n"
    "  - desc: <one sentence>\n"
    "  - hace: <what the class does>\n"
    "  - gotchas: <gotchas, or none>"
)

_NOTE_HEADERS = (
    ("- FILE ", ""),
    ("- FUNCTION ", "fn:"),
    ("- CLASS ", "class:"),
)


class ScanError(Exception):
    """Raised when scan cannot list git-tracked files."""


def run_scan(
    root: Path, *, config: dict | None = None, invoke: Invoke | None = None
) -> str:
    """Walk git-tracked text files and write the code graph + notes.

    ``invoke`` is the ``bulk_read`` helper's one-shot call; ``config``
    supplies ``threshold_lines`` for batching fresh files.
    """
    repo = root.name
    tracked = _git_tracked_files(root)
    previous = _read_graph(_graph_path(root, repo))
    cached = _files_by_path(previous["nodes"])
    cached_ids = _nodes_by_id(previous["nodes"])
    fresh, reused = _partition_by_hash(tracked, root, cached)
    if _unchanged_tracked_files(fresh, tracked, root, cached):
        return "scanned"
    notes = _describe_fresh(fresh, root, repo, config, invoke)
    extracts = _python_extracts(root, tracked)
    file_nodes = _merge_nodes(repo, root, tracked, notes, reused)
    symbol_nodes, edges = _python_graph(repo, extracts, notes, reused, cached_ids)
    _write_memory(
        root,
        repo,
        fresh,
        notes,
        file_nodes + symbol_nodes,
        edges,
        extracts,
        write_notes=invoke is not None,
    )
    return "scanned"


def _describe_fresh(
    fresh: list[Path],
    root: Path,
    repo: str,
    config: dict | None,
    invoke: Invoke | None,
) -> dict[str, dict[str, str]]:
    if invoke is None:
        return {}
    notes: dict[str, dict[str, str]] = {}
    for batch in _batch_paths(fresh, _threshold_lines(config)):
        raw = bulk_read(SCAN_QUESTION, batch, root=root, repo=repo, invoke=invoke)
        notes.update(_parse_notes(raw))
    return notes


def _threshold_lines(config: dict | None) -> int:
    return int((config or {}).get("threshold_lines", DEFAULT_THRESHOLD_LINES))


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


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _parse_notes(text: str) -> dict[str, dict[str, str]]:
    notes: dict[str, dict[str, str]] = {}
    current = ""
    for line in text.splitlines():
        current = _note_line(notes, current, line)
    return notes


def _note_line(notes: dict[str, dict[str, str]], current: str, line: str) -> str:
    header = _note_header(line)
    if header is not None:
        notes[header] = dict(EMPTY_NOTE)
        return header
    if current:
        _note_field(notes[current], line)
    return current


def _note_header(line: str) -> str | None:
    for prefix, key_prefix in _NOTE_HEADERS:
        if line.startswith(prefix):
            return key_prefix + line[len(prefix) :].strip()
    return None


def _note_field(note: dict[str, str], line: str) -> None:
    for key in ("desc", "hace", "gotchas"):
        prefix = f"  - {key}: "
        if line.startswith(prefix):
            note[key] = line[len(prefix) :]
            return


def _git_tracked_files(root: Path) -> list[Path]:
    result = _run_git_ls_files(root)
    _raise_git_ls_error(result)
    return _text_tracked_files(root, result.stdout)


def _run_git_ls_files(root: Path) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise ScanError(f"unable to list git files: {error}") from error


def _raise_git_ls_error(result: subprocess.CompletedProcess[bytes]) -> None:
    if result.returncode == 0:
        return
    err = result.stderr.decode().strip() or "git ls-files failed"
    raise ScanError(err)


def _text_tracked_files(root: Path, stdout: bytes) -> list[Path]:
    files = []
    for relative in _nul_split_paths(stdout):
        path = (root / relative).resolve()
        if _is_text_file(path):
            files.append(path)
    return files


def _nul_split_paths(stdout: bytes) -> list[str]:
    return [item.decode() for item in stdout.split(b"\0") if item]


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


def _read_graph(graph_path: Path) -> dict:
    empty: dict = {"nodes": [], "edges": []}
    if not graph_path.is_file():
        return empty
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    nodes = data.get("nodes")
    edges = data.get("edges")
    return {
        "nodes": nodes if isinstance(nodes, list) else [],
        "edges": edges if isinstance(edges, list) else [],
    }


def _files_by_path(nodes: list) -> dict[str, dict]:
    return {
        node["path"]: node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("path"), str)
    }


def _nodes_by_id(nodes: list) -> dict[str, dict]:
    return {
        node["id"]: node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
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


def _unchanged_tracked_files(
    fresh: list[Path],
    tracked: list[Path],
    root: Path,
    cached: dict[str, dict],
) -> bool:
    if fresh:
        return False
    return {_relative(path, root) for path in tracked} == set(cached)


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


def _python_extracts(root: Path, paths: list[Path]) -> dict[str, dict | None]:
    extracts: dict[str, dict | None] = {}
    for path in paths:
        relative = _relative(path, root)
        if not relative.endswith(".py"):
            continue
        extracts[relative] = extract_python(
            path.read_text(encoding="utf-8"),
            module_name(relative),
        )
    return extracts


def _python_graph(
    repo: str,
    extracts: dict[str, dict | None],
    notes: dict[str, dict[str, str]],
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    ok = {relative: data for relative, data in extracts.items() if data is not None}
    modules = {module_name(relative): relative for relative in extracts}
    nodes, edges, names = _defined_symbols(repo, ok, notes, reused, cached_ids)
    edges.extend(_reference_edges(repo, ok, modules, names))
    return nodes, edges


def _defined_symbols(
    repo: str,
    extracts: dict[str, dict],
    notes: dict[str, dict[str, str]],
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    names: dict[str, list[str]] = {}
    for relative, data in extracts.items():
        file_id = f"{repo}:file:{relative}"
        for qualname in data["functions"]:
            node_id = f"{repo}:fn:{qualname}"
            nodes.append(
                _symbol_node(
                    node_id, "Function", relative, f"fn:{qualname}", notes, reused, cached_ids
                )
            )
            edges.append({"from": file_id, "to": node_id, "kind": "defines"})
            names.setdefault(qualname.rsplit(".", 1)[-1], []).append(node_id)
        for qualname in data["classes"]:
            node_id = f"{repo}:class:{qualname}"
            nodes.append(
                _symbol_node(
                    node_id, "Class", relative, f"class:{qualname}", notes, reused, cached_ids
                )
            )
            edges.append({"from": file_id, "to": node_id, "kind": "defines"})
    return nodes, edges, names


def _reference_edges(
    repo: str,
    extracts: dict[str, dict],
    modules: dict[str, str],
    names: dict[str, list[str]],
) -> list[dict]:
    edges: list[dict] = []
    for relative, data in extracts.items():
        file_id = f"{repo}:file:{relative}"
        for imported in resolve_imports(module_name(relative), data["imports"], modules):
            edges.append({"from": file_id, "to": f"{repo}:file:{imported}", "kind": "imports"})
        for caller, callee in resolve_calls(data["calls"], names):
            edges.append({"from": f"{repo}:fn:{caller}", "to": callee, "kind": "calls"})
    return edges


def _symbol_node(
    node_id: str,
    kind: str,
    relative: str,
    note_key: str,
    notes: dict[str, dict[str, str]],
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> dict:
    return {
        "id": node_id,
        "kind": kind,
        "desc": _symbol_desc(node_id, note_key, relative, notes, reused, cached_ids),
    }


def _symbol_desc(
    node_id: str,
    note_key: str,
    relative: str,
    notes: dict[str, dict[str, str]],
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> str:
    if relative in reused:
        cached = cached_ids.get(node_id, {})
        if isinstance(cached.get("desc"), str):
            return cached["desc"]
        return ""
    return notes.get(note_key, EMPTY_NOTE)["desc"]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root.resolve()).as_posix()


def _write_memory(
    root: Path,
    repo: str,
    described: list[Path],
    notes: dict[str, dict[str, str]],
    nodes: list[dict],
    edges: list[dict],
    extracts: dict[str, dict | None],
    *,
    write_notes: bool,
) -> None:
    base = root / ".k0-mem" / "Project_context" / repo
    if write_notes:
        _write_notes_markdown(root, base, described, notes, extracts)
    _prune_stale_markdown(base, repo, nodes)
    graph_path = _graph_path(root, repo)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_notes_markdown(
    root: Path,
    base: Path,
    described: list[Path],
    notes: dict[str, dict[str, str]],
    extracts: dict[str, dict | None],
) -> None:
    for path in described:
        relative = _relative(path, root)
        note = notes.get(relative, EMPTY_NOTE)
        md_path = base / "files" / f"{relative}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_file_markdown(relative, note), encoding="utf-8")
    _write_symbol_markdown(root, base, described, notes, extracts)


def _write_symbol_markdown(
    root: Path,
    base: Path,
    described: list[Path],
    notes: dict[str, dict[str, str]],
    extracts: dict[str, dict | None],
) -> None:
    for path in described:
        relative = _relative(path, root)
        extract = extracts.get(relative)
        if not extract:
            continue
        _write_kind_markdown(base / "functions", extract["functions"], "fn:", notes)
        _write_kind_markdown(base / "classes", extract["classes"], "class:", notes)


def _write_kind_markdown(
    directory: Path,
    qualnames: list[str],
    prefix: str,
    notes: dict[str, dict[str, str]],
) -> None:
    if not qualnames:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for qualname in qualnames:
        note = notes.get(f"{prefix}{qualname}", EMPTY_NOTE)
        (directory / f"{qualname}.md").write_text(
            _file_markdown(qualname, note),
            encoding="utf-8",
        )


def _prune_stale_markdown(base: Path, repo: str, nodes: list[dict]) -> None:
    _remove_stale_file_markdown(
        base / "files",
        {
            node["path"]
            for node in nodes
            if node.get("kind") == "File" and isinstance(node.get("path"), str)
        },
    )
    _remove_stale_file_markdown(
        base / "functions",
        _symbol_names(nodes, "Function", f"{repo}:fn:"),
    )
    _remove_stale_file_markdown(
        base / "classes",
        _symbol_names(nodes, "Class", f"{repo}:class:"),
    )


def _symbol_names(nodes: list[dict], kind: str, prefix: str) -> set[str]:
    return {
        node["id"].removeprefix(prefix)
        for node in nodes
        if node.get("kind") == kind and isinstance(node.get("id"), str)
    }


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
