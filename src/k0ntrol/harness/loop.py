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
from k0ntrol.harness.retrieve import agent_memory
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
            invoke=_test_invoke(_memory_invoke(tester_invoke, spec, root, config)),
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
    run_suite: RunSuite | None = None,
) -> str:
    _begin_event_run()
    if not task.strip():
        write("missing task")
        return "stuck"
    spec_invoke = _with_event_step(invoke, "spec")
    prompt = _full_context_prompt(
        task,
        root,
        config,
        spec_invoke,
        bulk_read,
        write,
    )
    if prompt is None:
        return "stuck"

    spec = _write_and_show_spec(prompt, root, config, spec_invoke, write)
    while True:
        status, prompt, spec = _run_full_iteration(
            spec,
            prompt,
            root,
            config,
            invoke,
            bulk_read,
            read_line,
            write,
            run_suite,
            spec_invoke,
        )
        if status != "continue":
            return status


def _run_full_iteration(
    spec: str,
    prompt: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite | None,
    spec_invoke: Invoke,
) -> tuple[str, str, str]:
    if _read_gate(read_line, write) != "y":
        prompt = _rejected_prompt(prompt, spec)
        spec = _write_and_show_spec(prompt, root, config, spec_invoke, write)
        return "continue", prompt, spec
    status = _run_full_pipeline(
        spec,
        root,
        config,
        invoke,
        bulk_read,
        read_line,
        write,
        run_suite,
    )
    if status != "spec":
        return status, prompt, spec
    prompt = _rejected_prompt(
        prompt,
        (root / "spec.md").read_text(encoding="utf-8"),
    )
    spec = _write_and_show_spec(prompt, root, config, spec_invoke, write)
    return "continue", prompt, spec


def _full_context_prompt(
    task: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    write: Callable[[str], None],
) -> str | None:
    try:
        return _context_prompt(task, root, config, invoke, bulk_read)
    except MentionError as error:
        write(str(error))
        return None


def _run_full_pipeline(
    spec: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite | None,
) -> str:
    try:
        return _write_tests(
            spec,
            root,
            config,
            invoke,
            bulk_read,
            read_line,
            write,
            run_suite,
        )
    except CodeWriteError as error:
        write(str(error))
        return "stuck"


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


def _first_spec_non_test_path(paths: list[str], root: Path) -> Path | None:
    for relative in paths:
        if not _is_test_path(relative):
            return (root / relative).resolve()
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
    references = tester_reference
    if target.is_file():
        target_section = _file_section(target, root)
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
        invoke=_coder_invoke(_memory_invoke(invoke, spec, root, config), target, root),
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
    result = _validate_attempt(
        spec,
        before,
        target,
        root,
        config,
        invoke,
        write,
        run_suite,
        test_path,
        reference_text,
    )
    if result is None:
        return "stuck"
    current_spec, current_before = result
    write(_unified_diff(target, current_before, root))
    if _read_gate(read_line, write) == "y":
        return "done"
    _append_spec_text(current_spec, root, "rejected at final gate")
    return "stuck"


def _validate_attempt(
    spec: str,
    before: str,
    target: Path,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
    run_suite: RunSuite,
    test_path: Path,
    reference_text: str,
) -> tuple[str, str] | None:
    retries = 0
    while True:
        code, output = run_suite(root)
        if code == 0:
            return spec, before
        result = _validation_retry(
            spec,
            target,
            test_path,
            reference_text,
            root,
            config,
            invoke,
            write,
            output,
            retries,
        )
        if result is None:
            return None
        spec, before, retries = result


def _validation_retry(
    spec: str,
    target: Path,
    test_path: Path,
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
    output: str,
    retries: int,
) -> tuple[str, str, int] | None:
    summary = _with_event_step(invoke, "validator")(
        load_mode("validator", config),
        _validator_prompt(spec, output),
    )
    updated_spec = _append_spec_text(spec, root, summary)
    if retries >= config["max_retries"]:
        _append_spec_text(updated_spec, root, "stuck")
        write("stuck")
        return None
    updated_before = _retry_coder(
        updated_spec,
        target,
        test_path,
        reference_text,
        root,
        config,
        _with_event_step(invoke, "coder"),
        write,
    )
    if updated_before is None:
        return None
    return updated_spec, updated_before, retries + 1


def _retry_coder(
    spec: str,
    target: Path,
    test_path: Path,
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
) -> str | None:
    try:
        coded, before = _write_coder(
            spec,
            target,
            test_path,
            reference_text,
            root,
            config,
            invoke,
        )
    except CodeWriteError as error:
        write(str(error))
        return None
    _show_file(coded, root, write)
    return before


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
    spec_invoke = _memory_invoke(invoke, prompt, root, config)
    spec = spec_invoke(load_mode("spec", config), prompt)
    (root / "spec.md").write_text(spec, encoding="utf-8")
    write(spec)
    return spec


def _memory_invoke(invoke: Invoke, query: str, root: Path, config: dict) -> Invoke:
    extra = agent_memory(query, root=root, config=config, invoke=invoke)

    def wrapped(mode: ModeSpec, prompt: str) -> str:
        return invoke(mode, _join_memory(prompt, extra))

    return wrapped


def _join_memory(prompt: str, extra: str) -> str:
    if not extra:
        return prompt
    return f"{prompt}\n\n{extra}"


def _rejected_prompt(prompt: str, spec: str) -> str:
    return f"{prompt}\n\nPrevious spec:\n{spec}\n\nrejected"


def _write_tests(
    spec: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite | None,
) -> str:
    files = _spec_files(spec)
    test_paths = [path for path in files if _is_test_path(path)]
    if not test_paths:
        return "tested"
    reference_text = _reference_text(files, root)
    return _test_gate_loop(
        spec,
        files,
        test_paths,
        reference_text,
        root,
        config,
        invoke,
        bulk_read,
        read_line,
        write,
        run_suite,
    )


def _test_gate_loop(
    spec: str,
    files: list[str],
    test_paths: list[str],
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite | None,
) -> str:
    mode = load_mode("tester", config)
    tester_invoke = _with_event_step(invoke, "tester")
    for attempt in range(2):
        _write_test_files(
            spec,
            test_paths,
            reference_text,
            root,
            mode,
            _memory_invoke(tester_invoke, spec, root, config),
            write,
        )
        if _read_gate(read_line, write) == "y":
            return _continue_full_after_tests(
                spec=spec,
                files=files,
                test_paths=[(root / path).resolve() for path in test_paths],
                reference_text=reference_text,
                root=root,
                config=config,
                invoke=invoke,
                bulk_read=bulk_read,
                read_line=read_line,
                write=write,
                run_suite=run_suite,
            )
        spec = _append_rejection(spec, root)
        if attempt == 1:
            write("stuck")
            return "stuck"
    return "stuck"


def _write_test_files(
    spec: str,
    test_paths: list[str],
    reference_text: str,
    root: Path,
    mode: ModeSpec,
    invoke: Invoke,
    write: Callable[[str], None],
) -> None:
    for relative in test_paths:
        expected = (root / relative).resolve()
        written = code_write(
            spec,
            root=root,
            mode=mode,
            invoke=_test_invoke(invoke, expected, root),
            target=None,
            reference_text=reference_text,
        )
        _show_file(written, root, write)


def _continue_full_after_tests(
    *,
    spec: str,
    files: list[str],
    test_paths: list[Path],
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite | None,
) -> str:
    if run_suite is None:
        return "done"
    test_path = test_paths[0]
    target = _first_spec_non_test_path(files, root)
    if target is None:
        return "done"
    coder_reference = _coder_reference_text(
        reference_text,
        test_path,
        target,
        root,
    )
    coded, before = _write_coder(
        spec,
        target,
        test_path,
        coder_reference,
        root,
        config,
        _with_event_step(invoke, "coder"),
    )
    _show_file(coded, root, write)
    return _validate_full(
        spec,
        before,
        target,
        root,
        config,
        invoke,
        bulk_read,
        read_line,
        write,
        run_suite,
        test_paths,
        coder_reference,
    )


def _validate_full(
    spec: str,
    before: str,
    target: Path,
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
    run_suite: RunSuite,
    test_paths: list[Path],
    reference_text: str,
) -> str:
    while True:
        result = _validate_attempt(
            spec,
            before,
            target,
            root,
            config,
            invoke,
            write,
            run_suite,
            test_paths[0],
            reference_text,
        )
        if result is None:
            return "stuck"
        spec, before = result
        write(_unified_diff(target, before, root))
        decision = _review_decision(
            spec,
            before,
            target,
            reference_text,
            test_paths,
            root,
            config,
            _with_event_step(invoke, "coder"),
            invoke,
            bulk_read,
            read_line,
            write,
        )
        if isinstance(decision, str):
            return decision
        spec, before = decision


def _review_decision(
    spec: str,
    before: str,
    target: Path,
    reference_text: str,
    test_paths: list[Path],
    root: Path,
    config: dict,
    coder_invoke: Invoke,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> str | tuple[str, str]:
    review_path = _write_review(
        spec,
        before,
        target,
        test_paths,
        root,
        config,
        invoke,
        bulk_read,
        write,
    )
    review = review_path.read_text(encoding="utf-8")
    choice = _read_review_gate(read_line, write)
    if choice == "ok":
        return "done"
    updated_spec = _append_review(spec, root, review)
    if choice == "spec":
        return "spec"
    coded, updated_before = _write_coder(
        updated_spec,
        target,
        test_paths[0],
        reference_text,
        root,
        config,
        coder_invoke,
    )
    _show_file(coded, root, write)
    return updated_spec, updated_before


def _write_review(
    spec: str,
    before: str,
    target: Path,
    test_paths: list[Path],
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
    write: Callable[[str], None],
) -> Path:
    review_path = root / "review.md"
    reviewer_invoke = _with_event_step(invoke, "reviewer")
    written = code_write(
        spec,
        root=root,
        mode=load_mode("reviewer", config),
        invoke=_memory_invoke(reviewer_invoke, spec, root, config),
        target=review_path,
        reference_text=_review_reference(
            before,
            target,
            test_paths,
            root,
            config,
            bulk_read,
            invoke,
        ),
    )
    _show_file(written, root, write)
    write("Final review: ok / spec / code")
    return written


def _review_reference(
    before: str,
    target: Path,
    test_paths: list[Path],
    root: Path,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
) -> str:
    paths = [*test_paths, target]
    paths = list(dict.fromkeys(paths))
    delegated = [
        path for path in paths if should_delegate(path, config["threshold_lines"])
    ]
    summary = ""
    if delegated:
        summary = bulk_read(
            "Review the completed implementation against spec.md.",
            delegated,
            root=root,
            repo=root.name,
            mode=load_mode("bulk_reader", config),
            invoke=_with_event_step(invoke, "reviewer"),
        )
    diff = _unified_diff(target, before, root)
    if len(diff.splitlines()) > config["threshold_lines"]:
        diff = bulk_read(
            "Summarize this implementation diff for review:\n" + diff,
            [target],
            root=root,
            repo=root.name,
            mode=load_mode("bulk_reader", config),
            invoke=_with_event_step(invoke, "reviewer"),
        )
    test_context = "\n\n".join(
        _review_path_context(path, root, delegated, summary)
        for path in test_paths
    )
    return "\n\n".join(
        (
            "Tests:\n" + test_context,
            "Implementation:\n"
            + _review_path_context(target, root, delegated, summary),
            "Implementation diff:\n" + diff,
        )
    )


def _review_path_context(
    path: Path,
    root: Path,
    delegated: list[Path],
    summary: str,
) -> str:
    if path in delegated:
        return summary
    return _file_section(path, root)


def _append_review(spec: str, root: Path, review: str) -> str:
    return _append_spec_text(spec, root, f"Review:\n{review}")


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


def _test_invoke(
    invoke: Invoke,
    expected: Path | None = None,
    root: Path | None = None,
) -> Invoke:
    def guarded(mode: ModeSpec, prompt: str) -> str:
        response = invoke(mode, prompt)
        relative = response.partition("\n")[0].removeprefix("FILE ").strip()
        candidate = Path(relative)
        matches_expected = (
            expected is None
            or not candidate.is_absolute()
            and ((root or Path.cwd()) / candidate).resolve() == expected
        )
        if (
            not response.startswith("FILE ")
            or not _is_test_path(relative)
            or not matches_expected
        ):
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


def _read_review_gate(
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> str:
    while True:
        answer = read_line().strip().lower()
        if answer in {"ok", "spec", "code"}:
            return answer
        write("ok / spec / code")


def run_configured_suite(root: Path, test_cmd: str) -> tuple[int, str]:
    result = subprocess.run(
        shlex.split(test_cmd),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr
