import difflib
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol.backends.invoke import current_event_context, set_event_context
from k0ntrol.frontends.mentions import MentionError, mention_paths
from k0ntrol.harness.delegate import should_delegate
from k0ntrol.modespec import ModeSpec, load_mode
from k0ntrol.stages.code_write import CodeWriteError, code_write


Invoke = Callable[[ModeSpec, str], str]
BulkRead = Callable[..., str]
RunSuite = Callable[[Path], tuple[int, str]]


def _begin_event_run() -> None:
    set_event_context(run=uuid4().hex, step="")


def _with_event_step(invoke: Invoke, step: str) -> Invoke:
    def invoke_with_step(mode: ModeSpec, prompt: str) -> str:
        run = current_event_context()["run"]
        set_event_context(run=run, step=step)
        return invoke(mode, prompt)

    return invoke_with_step


def run_fast(
    task: str,
    *,
    root: Path,
    config: dict,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    invoke: Invoke,
    bulk_read: BulkRead,
    run_suite: RunSuite | None = None,
) -> str:
    _begin_event_run()
    if not task.strip():
        write("missing task")
        return "stuck"
    try:
        paths = mention_paths(task, root)
    except MentionError as error:
        write(str(error))
        return "stuck"
    question = re.sub(r"@[^\s]+", "", task).strip()
    spec = _literal_spec(question, paths, root)
    (root / "spec.md").write_text(spec, encoding="utf-8")
    write(spec)
    tester_invoke = _with_event_step(invoke, "tester")
    try:
        reference_text = _mention_references(
            question,
            paths,
            root,
            config,
            tester_invoke,
            bulk_read,
        )
        written = code_write(
            spec,
            root=root,
            mode=load_mode("tester", config),
            invoke=_test_invoke(tester_invoke),
            target=None,
            reference_text=reference_text,
        )
    except CodeWriteError as error:
        write(str(error))
        return "stuck"
    _show_file(written, root, write)
    target = _first_non_test_path(paths, root)
    if target is None:
        return "tested"
    try:
        coded, before = _write_coder(
            spec,
            target,
            written,
            reference_text,
            root,
            config,
            _with_event_step(invoke, "coder"),
        )
    except CodeWriteError as error:
        write(str(error))
        return "stuck"
    _show_file(coded, root, write)
    if run_suite is None:
        return "tested"
    return _validate_fast(
        spec,
        before,
        target,
        root,
        config,
        invoke,
        read_line,
        write,
        run_suite,
        written,
        reference_text,
    )


def run_full(
    task: str,
    *,
    root: Path,
    config: dict,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    invoke: Invoke,
    bulk_read: BulkRead,
) -> str:
    _begin_event_run()
    if not task.strip():
        write("missing task")
        return "stuck"
    spec_invoke = _with_event_step(invoke, "spec")
    try:
        prompt = _context_prompt(task, root, config, spec_invoke, bulk_read)
    except MentionError as error:
        write(str(error))
        return "stuck"

    spec = _write_and_show_spec(prompt, root, config, spec_invoke, write)
    while True:
        answer = _read_gate(read_line, write)
        if answer == "y":
            try:
                return _write_tests(spec, root, config, invoke, read_line, write)
            except CodeWriteError as error:
                write(str(error))
                return "stuck"
        prompt = _rejected_prompt(prompt, spec)
        spec = _write_and_show_spec(prompt, root, config, spec_invoke, write)


def _context_prompt(
    task: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
) -> str:
    paths = mention_paths(task, root)
    question = re.sub(r"@[^\s]+", "", task).strip()
    delegated = [path for path in paths if should_delegate(path, config["threshold_lines"])]
    direct = [path for path in paths if path not in delegated]
    sections = [question]
    if delegated:
        mode = load_mode("bulk_reader", config)
        sections.append(
            bulk_read(
                question,
                delegated,
                root=root,
                repo=root.name,
                mode=mode,
                invoke=invoke,
            )
        )
    sections.extend(_file_section(path, root) for path in direct)
    return "\n\n".join(sections)


def _literal_spec(question: str, paths: list[Path], root: Path) -> str:
    files = "\n".join(
        f"- {path.relative_to(root.resolve()).as_posix()}" for path in paths
    )
    return f"{question}\n\n## Files\n{files}\n"


def _first_non_test_path(paths: list[Path], root: Path) -> Path | None:
    for path in paths:
        relative = path.relative_to(root.resolve()).as_posix()
        if not _is_test_path(relative):
            return path
    return None


def _mention_references(
    question: str,
    paths: list[Path],
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
) -> str:
    target = _first_non_test_path(paths, root)
    reference_paths = [path for path in paths if path != target]
    if target is not None and not should_delegate(target, config["threshold_lines"]):
        reference_paths.insert(0, target)
    delegated = [
        path
        for path in reference_paths
        if should_delegate(path, config["threshold_lines"])
    ]
    delegated_summary = ""
    if delegated:
        delegated_summary = bulk_read(
            question,
            delegated,
            root=root,
            repo=root.name,
            mode=load_mode("bulk_reader", config),
            invoke=invoke,
        )
    sections = []
    summary_added = False
    for path in reference_paths:
        if path in delegated:
            if delegated_summary and not summary_added:
                sections.append(delegated_summary)
                summary_added = True
            continue
        sections.append(_file_section(path, root))
    return "\n".join(sections)


def _coder_reference_text(
    tester_reference: str,
    test_path: Path,
    target: Path,
    root: Path,
) -> str:
    target_section = _file_section(target, root)
    references = tester_reference
    if references.startswith(target_section):
        references = references[len(target_section) :].lstrip("\n")
    test_section = _file_section(test_path, root)
    return "\n".join(section for section in (references, test_section) if section)


def _write_coder(
    spec: str,
    target: Path,
    test_path: Path,
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
) -> tuple[Path, str]:
    before = target.read_text(encoding="utf-8") if target.is_file() else ""
    coded = code_write(
        spec,
        root=root,
        mode=load_mode("coder", config),
        invoke=_coder_invoke(invoke, target, root),
        target=target,
        reference_text=reference_text,
    )
    return coded, before


def _validate_fast(
    spec: str,
    before: str,
    target: Path,
    root: Path,
    config: dict,
    invoke: Invoke,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite,
    test_path: Path,
    reference_text: str,
) -> str:
    retries = 0
    validator_invoke = _with_event_step(invoke, "validator")
    coder_invoke = _with_event_step(invoke, "coder")
    while True:
        code, output = run_suite(root)
        if code == 0:
            write(_unified_diff(target, before, root))
            if _read_gate(read_line, write) == "y":
                return "done"
            _append_spec_text(spec, root, "rejected at final gate")
            return "stuck"
        summary = validator_invoke(
            load_mode("validator", config),
            _validator_prompt(spec, output),
        )
        spec = _append_spec_text(spec, root, summary)
        if retries >= config["max_retries"]:
            _append_spec_text(spec, root, "stuck")
            write("stuck")
            return "stuck"
        retries += 1
        try:
            coded, before = _write_coder(
                spec,
                target,
                test_path,
                reference_text,
                root,
                config,
                coder_invoke,
            )
        except CodeWriteError as error:
            write(str(error))
            return "stuck"
        _show_file(coded, root, write)


def _validator_prompt(spec: str, output: str) -> str:
    return f"{spec}\n\nSuite output:\n{output}"


def _append_spec_text(spec: str, root: Path, addition: str) -> str:
    updated = f"{spec.rstrip(chr(10))}\n{addition.strip()}\n"
    (root / "spec.md").write_text(updated, encoding="utf-8")
    return updated


def _unified_diff(path: Path, before: str, root: Path) -> str:
    relative = path.relative_to(root.resolve()).as_posix()
    after = path.read_text(encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


def _file_section(path: Path, root: Path) -> str:
    relative = path.relative_to(root.resolve()).as_posix()
    return f"{relative}\n{path.read_text(encoding='utf-8')}"


def _write_and_show_spec(
    prompt: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
) -> str:
    spec = invoke(load_mode("spec", config), prompt)
    (root / "spec.md").write_text(spec, encoding="utf-8")
    write(spec)
    return spec


def _rejected_prompt(prompt: str, spec: str) -> str:
    return f"{prompt}\n\nPrevious spec:\n{spec}\n\nrejected"


def _write_tests(
    spec: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> str:
    files = _spec_files(spec)
    test_paths = [path for path in files if _is_test_path(path)]
    if not test_paths:
        return "tested"
    reference_text = _reference_text(files, root)
    mode = load_mode("tester", config)
    for attempt in range(2):
        for _path in test_paths:
            written = code_write(
                spec,
                root=root,
                mode=mode,
                invoke=_test_invoke(invoke),
                target=None,
                reference_text=reference_text,
            )
            _show_file(written, root, write)
        if _read_gate(read_line, write) == "y":
            return "done"
        spec = _append_rejection(spec, root)
        if attempt == 1:
            write("stuck")
            return "stuck"
    return "stuck"


def _append_rejection(spec: str, root: Path) -> str:
    updated = spec.rstrip("\n") + "\nrejected by human\n"
    (root / "spec.md").write_text(updated, encoding="utf-8")
    return updated


def _spec_files(spec: str) -> list[str]:
    lines = spec.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "## Files")
    except StopIteration:
        return []
    paths = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        value = line.strip()
        if value:
            paths.append(value[2:].strip() if value.startswith("- ") else value)
    return paths


def _is_test_path(relative: str) -> bool:
    path = Path(relative)
    return (
        path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or "tests" in path.parts
    )


def _reference_text(files: list[str], root: Path) -> str:
    sections = []
    for relative in files:
        if _is_test_path(relative):
            continue
        path = (root / relative).resolve()
        if path.is_file():
            sections.append(f"{path.relative_to(root.resolve()).as_posix()}\n")
            sections[-1] += path.read_text(encoding="utf-8")
    return "\n".join(sections)


def _test_invoke(invoke: Invoke) -> Invoke:
    def guarded(mode: ModeSpec, prompt: str) -> str:
        response = invoke(mode, prompt)
        relative = response.partition("\n")[0].removeprefix("FILE ").strip()
        if not response.startswith("FILE ") or not _is_test_path(relative):
            raise CodeWriteError(f"FILE path is not a test path: {relative}")
        return response

    return guarded


def _coder_invoke(invoke: Invoke, target: Path, root: Path) -> Invoke:
    def guarded(mode: ModeSpec, prompt: str) -> str:
        response = invoke(mode, prompt)
        first_line = response.partition("\n")[0]
        relative = first_line.removeprefix("FILE ").strip()
        candidate = Path(relative)
        if (
            not response.startswith("FILE ")
            or candidate.is_absolute()
            or (root / candidate).resolve() != target.resolve()
        ):
            raise CodeWriteError(f"FILE path is not the coder target: {relative}")
        return response

    return guarded


def _show_file(path: Path, root: Path, write: Callable[[str], None]) -> None:
    relative = path.relative_to(root.resolve()).as_posix()
    write(f"{relative}\n{path.read_text(encoding='utf-8')}")


def _read_gate(read_line: Callable[[], str], write: Callable[[str], None]) -> str:
    while True:
        answer = read_line().strip().lower()
        if answer in {"y", "n"}:
            return answer
        write("y or n")


def run_configured_suite(root: Path, test_cmd: str) -> tuple[int, str]:
    result = subprocess.run(
        shlex.split(test_cmd),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr
