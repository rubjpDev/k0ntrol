import ast
from pathlib import Path


def module_name(relative: str) -> str:
    """Return the dotted module name for a repo-relative Python path."""
    parts = list(Path(relative).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_python(source: str, module: str) -> dict | None:
    """Collect functions, classes, imports and calls. None on syntax error."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    extractor = _Extractor(module)
    extractor.visit(tree)
    return {
        "functions": extractor.functions,
        "classes": extractor.classes,
        "imports": extractor.imports,
        "calls": extractor.calls,
    }


def resolve_imports(
    current_module: str,
    raw_imports: list[tuple[int, str, list[str]]],
    modules: dict[str, str],
) -> list[str]:
    """Map import statements to repo-relative paths. Unknown modules are dropped."""
    found: list[str] = []
    seen: set[str] = set()
    for level, module, names in raw_imports:
        _collect_import_paths(
            found, seen, current_module, level, module, names, modules
        )
    return found


def _collect_import_paths(
    found: list[str],
    seen: set[str],
    current_module: str,
    level: int,
    module: str,
    names: list[str],
    modules: dict[str, str],
) -> None:
    for candidate in _import_candidates(current_module, level, module, names):
        _remember_import_path(found, seen, modules.get(candidate))


def _remember_import_path(
    found: list[str], seen: set[str], path: str | None
) -> None:
    if path is None or path in seen:
        return
    seen.add(path)
    found.append(path)


def resolve_calls(
    calls: list[tuple[str, str]],
    functions_by_name: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Resolve call names to in-repo function ids. Unresolved names are dropped."""
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for caller, name in calls:
        for callee in functions_by_name.get(name, []):
            key = (caller, callee)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)
    return edges


def _import_candidates(
    current: str,
    level: int,
    module: str,
    names: list[str],
) -> list[str]:
    base = _import_base(current, level, module)
    if base is None:
        return []
    return _namespaced_import_candidates(base, names)


def _namespaced_import_candidates(base: str, names: list[str]) -> list[str]:
    candidates = _optional_import_base(base)
    for name in names:
        _append_imported_name(candidates, base, name)
    return candidates


def _optional_import_base(base: str) -> list[str]:
    if not base:
        return []
    return [base]


def _append_imported_name(candidates: list[str], base: str, name: str) -> None:
    if name == "*":
        return
    candidates.append(_qualified_import(base, name))


def _qualified_import(base: str, name: str) -> str:
    if not base:
        return name
    return f"{base}.{name}"


def _import_base(current: str, level: int, module: str) -> str | None:
    if level == 0:
        return module
    parts = current.split(".") if current else []
    if len(parts) < level:
        return None
    parent = parts[:-level]
    if module:
        parent.extend(module.split("."))
    return ".".join(parent)


def _called_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _Extractor(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.functions: list[str] = []
        self.classes: list[str] = []
        self.imports: list[tuple[int, str, list[str]]] = []
        self.calls: list[tuple[str, str]] = []
        self._qual: list[str] = []
        self._caller: list[str | None] = [None]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(self._qualify(node.name))
        self._qual.append(node.name)
        for child in node.body:
            self.visit(child)
        self._qual.pop()

    def visit_Call(self, node: ast.Call) -> None:
        caller = self._caller[-1]
        name = _called_name(node.func)
        if caller and name:
            self.calls.append((caller, name))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((0, alias.name, []))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = [alias.name for alias in node.names]
        self.imports.append((node.level, node.module or "", names))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = self._qualify(node.name)
        self.functions.append(qualname)
        self._qual.append(node.name)
        self._caller.append(qualname)
        for child in node.body:
            self.visit(child)
        self._caller.pop()
        self._qual.pop()

    def _qualify(self, name: str) -> str:
        return ".".join(part for part in (self.module, *self._qual, name) if part)
