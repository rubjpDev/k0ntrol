"""Hash-based code-graph scan over git-tracked text files.

The scan writes the code graph (`graph.json`) under
`.k0-mem/Project_context/<repo>/`. File notes are empty until the
`bulk_read` helper is wired back into the scan in a later ticket.
"""

import hashlib
import json
import subprocess
from pathlib import Path

from k0ntrol.memory.python_extract import (
    extract_python,
    module_name,
    resolve_calls,
    resolve_imports,
)


class ScanError(Exception):
    """Raised when scan cannot list git-tracked files."""


def run_scan(root: Path) -> str:
    """Walk git-tracked text files and write the code graph."""
    repo = root.name
    tracked = _git_tracked_files(root)
    previous = _read_graph(_graph_path(root, repo))
    cached = _files_by_path(previous["nodes"])
    cached_ids = _nodes_by_id(previous["nodes"])
    fresh, reused = _partition_by_hash(tracked, root, cached)
    if _unchanged_tracked_files(fresh, tracked, root, cached):
        return "scanned"
    extracts = _python_extracts(root, tracked)
    file_nodes = _merge_nodes(repo, root, tracked, reused)
    symbol_nodes, edges = _python_graph(repo, extracts, reused, cached_ids)
    _write_graph(root, repo, file_nodes + symbol_nodes, edges)
    return "scanned"


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
    reused: dict[str, dict],
) -> list[dict]:
    nodes = []
    for path in paths:
        relative = _relative(path, root)
        if relative in reused:
            nodes.append(reused[relative])
            continue
        nodes.append(
            {
                "id": f"{repo}:file:{relative}",
                "kind": "File",
                "path": relative,
                "desc": "",
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
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    ok = {relative: data for relative, data in extracts.items() if data is not None}
    modules = {module_name(relative): relative for relative in extracts}
    nodes, edges, names = _defined_symbols(repo, ok, reused, cached_ids)
    edges.extend(_reference_edges(repo, ok, modules, names))
    return nodes, edges


def _defined_symbols(
    repo: str,
    extracts: dict[str, dict],
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
                _symbol_node(node_id, "Function", relative, reused, cached_ids)
            )
            edges.append({"from": file_id, "to": node_id, "kind": "defines"})
            names.setdefault(qualname.rsplit(".", 1)[-1], []).append(node_id)
        for qualname in data["classes"]:
            node_id = f"{repo}:class:{qualname}"
            nodes.append(
                _symbol_node(node_id, "Class", relative, reused, cached_ids)
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
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> dict:
    return {
        "id": node_id,
        "kind": kind,
        "desc": _symbol_desc(node_id, relative, reused, cached_ids),
    }


def _symbol_desc(
    node_id: str,
    relative: str,
    reused: dict[str, dict],
    cached_ids: dict[str, dict],
) -> str:
    if relative in reused:
        cached = cached_ids.get(node_id, {})
        if isinstance(cached.get("desc"), str):
            return cached["desc"]
    return ""


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root.resolve()).as_posix()


def _write_graph(root: Path, repo: str, nodes: list[dict], edges: list[dict]) -> None:
    graph_path = _graph_path(root, repo)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "\n",
        encoding="utf-8",
    )
