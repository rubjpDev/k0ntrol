import difflib
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol.backends.invoke import (
    current_event_context,
    set_event_context,
    update_last_event,
)
from k0ntrol.frontends.mentions import MentionError, mention_paths
from k0ntrol.harness.delegate import should_delegate
from k0ntrol.harness.retrieve import agent_memory, custom_topology_text
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
    targets = _mention_non_test_paths(paths, root)
    if not targets:
        return "tested"
    try:
        befores = _write_coder_targets(
            spec,
            targets,
            written,
            reference_text,
            root,
            config,
            _with_event_step(invoke, "coder"),
            write,
        )
    except CodeWriteError as error:
        write(str(error))
        return "stuck"
    if run_suite is None:
        return "tested"
    return _validate_fast(
        spec,
        befores,
        targets,
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
    accepted = _accept_files_contract(
        spec, prompt, root, config, spec_invoke, write
    )
    if accepted is None:
        return "stuck", prompt, spec
    spec, files, prompt = accepted
    _show_files_contract(files, write)
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
    targets = _mention_non_test_paths(paths, root)
    return targets[0] if targets else None


def _mention_non_test_paths(paths: list[Path], root: Path) -> list[Path]:
    return [
        path
        for path in paths
        if not _is_test_path(path.relative_to(root.resolve()).as_posix())
    ]


def _spec_non_test_paths(paths: list[str], root: Path) -> list[Path]:
    return [
        (root / relative).resolve()
        for relative in paths
        if not _is_test_path(relative)
    ]


def _mention_references(
    question: str,
    paths: list[Path],
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
) -> str:
    threshold = config["threshold_lines"]
    reference_paths = _tester_reference_paths(paths, root, threshold)
    delegated = _delegated_paths(reference_paths, threshold)
    summary = _delegated_summary(
        question, delegated, root, config, invoke, bulk_read
    )
    return _join_reference_sections(reference_paths, delegated, summary, root)


def _tester_reference_paths(
    paths: list[Path], root: Path, threshold: int
) -> list[Path]:
    target = _first_non_test_path(paths, root)
    others = [path for path in paths if path != target]
    if _include_target_in_refs(target, threshold):
        others.insert(0, target)
    return others


def _include_target_in_refs(target: Path | None, threshold: int) -> bool:
    return target is not None and not should_delegate(target, threshold)


def _delegated_paths(paths: list[Path], threshold: int) -> list[Path]:
    return [path for path in paths if should_delegate(path, threshold)]


def _delegated_summary(
    question: str,
    delegated: list[Path],
    root: Path,
    config: dict,
    invoke: Invoke,
    bulk_read: BulkRead,
) -> str:
    if not delegated:
        return ""
    return bulk_read(
        question,
        delegated,
        root=root,
        repo=root.name,
        mode=load_mode("bulk_reader", config),
        invoke=invoke,
    )


def _join_reference_sections(
    paths: list[Path],
    delegated: list[Path],
    summary: str,
    root: Path,
) -> str:
    sections: list[str] = []
    inserted = False
    for path in paths:
        inserted = _add_mention_section(
            sections, path, delegated, summary, inserted, root
        )
    return "\n".join(sections)


def _add_mention_section(
    sections: list[str],
    path: Path,
    delegated: list[Path],
    summary: str,
    inserted: bool,
    root: Path,
) -> bool:
    if path not in delegated:
        sections.append(_file_section(path, root))
        return inserted
    return _add_delegated_summary(sections, summary, inserted)


def _add_delegated_summary(
    sections: list[str], summary: str, inserted: bool
) -> bool:
    if inserted or not summary:
        return inserted
    sections.append(summary)
    return True


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


def _write_coder_targets(
    spec: str,
    targets: list[Path],
    test_path: Path,
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
    kept: list[Path] | None = None,
) -> dict[Path, str]:
    befores: dict[Path, str] = {}
    written: list[Path] = []
    for target in targets:
        refs = _coder_call_reference(
            reference_text, test_path, target, written, root, kept or []
        )
        coded, before = _write_coder(
            spec, target, test_path, refs, root, config, invoke
        )
        _show_file(coded, root, write)
        befores[target] = before
        written.append(coded)
    return befores


def _coder_call_reference(
    reference_text: str,
    test_path: Path,
    target: Path,
    written: list[Path],
    root: Path,
    kept: list[Path],
) -> str:
    refs = _coder_reference_text(reference_text, test_path, target, root)
    for path in written:
        refs = _append_written_reference(refs, path, root)
    for path in kept:
        if path != target and path.is_file():
            refs = _append_written_reference(refs, path, root)
    return refs


def _append_written_reference(reference_text: str, path: Path, root: Path) -> str:
    section = _file_section(path, root)
    if not reference_text:
        return section
    return f"{reference_text}\n{section}"


def _validate_fast(
    spec: str,
    befores: dict[Path, str],
    targets: list[Path],
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
        befores,
        targets,
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
    current_spec, befores = result
    write(_combined_diff(targets, befores, root))
    if _read_gate(read_line, write) == "y":
        return "done"
    _append_spec_text(current_spec, root, "rejected at final gate")
    return "stuck"


def _validate_attempt(
    spec: str,
    befores: dict[Path, str],
    targets: list[Path],
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
    run_suite: RunSuite,
    test_path: Path,
    reference_text: str,
) -> tuple[str, dict[Path, str]] | None:
    retries = 0
    while True:
        code, output = _run_suite_and_gates(run_suite, root, config)
        if code == 0:
            return spec, befores
        result = _validation_retry(
            spec,
            targets,
            test_path,
            reference_text,
            root,
            config,
            invoke,
            write,
            output,
            retries,
            befores,
        )
        if result is None:
            return None
        spec, befores, retries = result


def _run_suite_and_gates(
    run_suite: RunSuite, root: Path, config: dict
) -> tuple[int, str]:
    code, output = run_suite(root)
    if code != 0:
        return code, output
    return _run_gates(root, config.get("gates") or [])


def _run_gates(root: Path, gates: list) -> tuple[int, str]:
    for gate in gates:
        code, output = run_configured_suite(root, gate["cmd"])
        if code != 0:
            return code, _gate_failure(gate["name"], output)
    return 0, ""


def _gate_failure(name: str, output: str) -> str:
    return f"gate {name}\n{output}"


def _validation_retry(
    spec: str,
    targets: list[Path],
    test_path: Path,
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
    output: str,
    retries: int,
    befores: dict[Path, str],
) -> tuple[str, dict[Path, str], int] | None:
    summary = _with_event_step(invoke, "validator")(
        load_mode("validator", config),
        _validator_prompt(spec, output),
    )
    updated_spec = _append_spec_text(spec, root, summary)
    if retries >= config["max_retries"]:
        _append_spec_text(updated_spec, root, "stuck")
        write("stuck")
        return None
    redo, kept = _plan_redo(
        summary,
        targets,
        root,
        write,
        last_attempt=retries + 1 >= config["max_retries"],
    )
    updated_befores = _retry_coder(
        updated_spec,
        redo,
        test_path,
        reference_text,
        root,
        config,
        _with_event_step(invoke, "coder"),
        write,
        kept=kept,
    )
    if updated_befores is None:
        return None
    return updated_spec, {**befores, **updated_befores}, retries + 1


def _parse_redo_paths(raw: str) -> list[str] | None:
    paths: list[str] = []
    collecting = False
    for line in raw.splitlines():
        collecting = _advance_redo(line, collecting, paths)
    return None if not collecting else paths


def _advance_redo(line: str, collecting: bool, paths: list[str]) -> bool:
    stripped = line.strip()
    if stripped.lower().startswith("redo:"):
        return True
    if collecting:
        _append_redo_bullet(stripped, paths)
    return collecting


def _append_redo_bullet(stripped: str, paths: list[str]) -> None:
    if stripped.startswith("- "):
        paths.append(stripped[2:].strip().strip("`"))


def _filter_redo_targets(
    proposed: list[str] | None, targets: list[Path], root: Path
) -> list[Path]:
    selected = _picked_redo_targets(proposed, targets, root)
    if not selected:
        return list(targets)
    return selected


def _picked_redo_targets(
    proposed: list[str] | None, targets: list[Path], root: Path
) -> list[Path]:
    if not proposed:
        return []
    allowed = _redo_allowed(targets, root)
    picked = _proposed_redo_paths(proposed, allowed)
    return [path for path in targets if path in picked]


def _redo_allowed(targets: list[Path], root: Path) -> dict[str, Path]:
    allowed: dict[str, Path] = {}
    for path in targets:
        allowed[path.relative_to(root.resolve()).as_posix()] = path
    return allowed


def _proposed_redo_paths(
    proposed: list[str], allowed: dict[str, Path]
) -> set[Path]:
    picked: set[Path] = set()
    for rel in proposed:
        if rel in allowed:
            picked.add(allowed[rel])
    return picked


def _select_redo_targets(
    raw: str,
    targets: list[Path],
    root: Path,
    last_attempt: bool = False,
) -> list[Path]:
    if last_attempt:
        return list(targets)
    return _filter_redo_targets(_parse_redo_paths(raw), targets, root)


def _plan_redo(
    raw: str,
    targets: list[Path],
    root: Path,
    write: Callable[[str], None],
    last_attempt: bool = False,
) -> tuple[list[Path], list[Path]]:
    redo = _select_redo_targets(raw, targets, root, last_attempt=last_attempt)
    _record_redo(redo, root, write)
    return redo, _kept_targets(targets, redo)


def _kept_targets(targets: list[Path], redo: list[Path]) -> list[Path]:
    chosen = set(redo)
    return [path for path in targets if path not in chosen]


def _relative_targets(targets: list[Path], root: Path) -> list[str]:
    return [path.relative_to(root.resolve()).as_posix() for path in targets]


def _show_redo(
    targets: list[Path], root: Path, write: Callable[[str], None]
) -> None:
    write("↻ coder: " + " ".join(_relative_targets(targets, root)))


def _record_redo(
    targets: list[Path], root: Path, write: Callable[[str], None]
) -> None:
    rels = _relative_targets(targets, root)
    _show_redo(targets, root, write)
    update_last_event(root, redo=rels)


def _retry_coder(
    spec: str,
    targets: list[Path],
    test_path: Path,
    reference_text: str,
    root: Path,
    config: dict,
    invoke: Invoke,
    write: Callable[[str], None],
    kept: list[Path] | None = None,
) -> dict[Path, str] | None:
    try:
        return _write_coder_targets(
            spec,
            targets,
            test_path,
            reference_text,
            root,
            config,
            invoke,
            write,
            kept=kept,
        )
    except CodeWriteError as error:
        write(str(error))
        return None


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


def _combined_diff(targets: list[Path], befores: dict[Path, str], root: Path) -> str:
    return "".join(_unified_diff(path, befores[path], root) for path in targets)


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
    spec = _trim_spec_to_goal(spec_invoke(load_mode("spec", config), prompt))
    (root / "spec.md").write_text(spec, encoding="utf-8")
    write(spec)
    return spec


def _trim_spec_to_goal(spec: str) -> str:
    index = spec.find("## Goal")
    if index < 0:
        return spec
    return spec[index:]


def _memory_invoke(invoke: Invoke, query: str, root: Path, config: dict) -> Invoke:
    extra = "\n\n".join(
        part
        for part in (
            custom_topology_text(root),
            agent_memory(query, root=root, config=config, invoke=invoke),
        )
        if part
    )

    def wrapped(mode: ModeSpec, prompt: str) -> str:
        return invoke(mode, _join_memory(prompt, extra))

    return wrapped


def _join_memory(prompt: str, extra: str) -> str:
    if not extra:
        return prompt
    return f"{prompt}\n\n{extra}"


def _rejected_prompt(prompt: str, spec: str) -> str:
    return _retry_prompt(prompt, spec, "rejected")


def _retry_prompt(prompt: str, spec: str, reason: str) -> str:
    return f"{prompt}\n\nPrevious spec:\n{spec}\n\n{reason}"


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
    targets = _spec_non_test_paths(files, root)
    if not targets:
        return "done"
    befores = _write_coder_targets(
        spec,
        targets,
        test_path,
        reference_text,
        root,
        config,
        _with_event_step(invoke, "coder"),
        write,
    )
    return _validate_full(
        spec,
        befores,
        targets,
        root,
        config,
        invoke,
        bulk_read,
        read_line,
        write,
        run_suite,
        test_paths,
        reference_text,
    )


def _validate_full(
    spec: str,
    befores: dict[Path, str],
    targets: list[Path],
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
            befores,
            targets,
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
        spec, befores = result
        write(_combined_diff(targets, befores, root))
        decision = _review_decision(
            spec,
            befores,
            targets,
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
        spec, befores = decision


def _review_decision(
    spec: str,
    befores: dict[Path, str],
    targets: list[Path],
    reference_text: str,
    test_paths: list[Path],
    root: Path,
    config: dict,
    coder_invoke: Invoke,
    invoke: Invoke,
    bulk_read: BulkRead,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> str | tuple[str, dict[Path, str]]:
    review_path = _write_review(
        spec,
        befores,
        targets,
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
    redo, kept = _plan_redo(review, targets, root, write)
    updated_befores = _write_coder_targets(
        updated_spec,
        redo,
        test_paths[0],
        reference_text,
        root,
        config,
        coder_invoke,
        write,
        kept=kept,
    )
    return updated_spec, {**befores, **updated_befores}


def _write_review(
    spec: str,
    befores: dict[Path, str],
    targets: list[Path],
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
            befores,
            targets,
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
    befores: dict[Path, str],
    targets: list[Path],
    test_paths: list[Path],
    root: Path,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
) -> str:
    paths = list(dict.fromkeys([*test_paths, *targets]))
    delegated = [
        path for path in paths if should_delegate(path, config["threshold_lines"])
    ]
    summary = _review_delegated_summary(delegated, root, config, bulk_read, invoke)
    diff = _review_diff(targets, befores, root, config, bulk_read, invoke)
    test_context = "\n\n".join(
        _review_path_context(path, root, delegated, summary)
        for path in test_paths
    )
    implementation = "\n\n".join(
        _review_path_context(path, root, delegated, summary) for path in targets
    )
    return "\n\n".join(
        (
            "Tests:\n" + test_context,
            "Implementation:\n" + implementation,
            "Implementation diff:\n" + diff,
        )
    )


def _review_delegated_summary(
    delegated: list[Path],
    root: Path,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
) -> str:
    if not delegated:
        return ""
    return bulk_read(
        "Review the completed implementation against spec.md.",
        delegated,
        root=root,
        repo=root.name,
        mode=load_mode("bulk_reader", config),
        invoke=_with_event_step(invoke, "reviewer"),
    )


def _review_diff(
    targets: list[Path],
    befores: dict[Path, str],
    root: Path,
    config: dict,
    bulk_read: BulkRead,
    invoke: Invoke,
) -> str:
    diff = _combined_diff(targets, befores, root)
    if len(diff.splitlines()) <= config["threshold_lines"]:
        return diff
    return bulk_read(
        "Summarize this implementation diff for review:\n" + diff,
        targets,
        root=root,
        repo=root.name,
        mode=load_mode("bulk_reader", config),
        invoke=_with_event_step(invoke, "reviewer"),
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


_FILE_LINE = re.compile(r"^- +(?:`([^`]+)`|(\S+)) +\((create|change)\)$")


def _accept_files_contract(
    spec: str,
    prompt: str,
    root: Path,
    config: dict,
    spec_invoke: Invoke,
    write: Callable[[str], None],
) -> tuple[str, list[tuple[str, str]], str] | None:
    retries = 0
    while True:
        entries = _parse_files_entries(spec)
        error = _files_contract_error(spec, root, entries)
        if error is None:
            return spec, entries or [], prompt
        spec = _append_spec_text(spec, root, error)
        write(error)
        if retries >= config["max_retries"]:
            write("stuck")
            return None
        prompt = _retry_prompt(prompt, spec, error)
        spec = _write_and_show_spec(prompt, root, config, spec_invoke, write)
        retries += 1


def _files_contract_error(
    spec: str,
    root: Path,
    entries: list[tuple[str, str]] | None,
) -> str | None:
    if entries is None:
        return _invalid_files_line(spec)
    return _files_entries_error(entries, root)


def _invalid_files_line(spec: str) -> str:
    bad = next(
        line for line in _files_section_lines(spec) if _parse_files_line(line) is None
    )
    return f"invalid ## Files line: {bad}"


def _files_entries_error(
    entries: list[tuple[str, str]], root: Path
) -> str | None:
    for path, action in entries:
        error = _files_path_error(path, action, root)
        if error:
            return error
    return _files_mix_error(entries)


def _files_path_error(path: str, action: str, root: Path) -> str | None:
    resolved = _resolved_in_repo(path, root)
    if resolved is None:
        return f"path is outside the repo: {path}"
    relative = resolved.relative_to(root.resolve())
    if relative.parts[:1] == (".k0-mem",):
        return f"path is inside .k0-mem/: {path}"
    if action == "change" and not resolved.is_file():
        return f"change path does not exist: {path}"
    if action == "create" and resolved.exists():
        return f"create path already exists: {path}"
    return None


def _resolved_in_repo(relative: str, root: Path) -> Path | None:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _files_mix_error(entries: list[tuple[str, str]]) -> str | None:
    if not any(_is_test_path(path) for path, _action in entries):
        return "## Files needs a test path"
    if not any(not _is_test_path(path) for path, _action in entries):
        return "## Files needs a non-test path"
    return None


def _show_files_contract(
    entries: list[tuple[str, str]], write: Callable[[str], None]
) -> None:
    for path, action in entries:
        write(f"{path} ({action})")


def _parse_files_line(line: str) -> tuple[str, str] | None:
    match = _FILE_LINE.fullmatch(line.strip())
    if match is None:
        return None
    return match.group(1) or match.group(2), match.group(3)


def _files_section_lines(spec: str) -> list[str]:
    lines = spec.splitlines()
    start = _files_section_start(lines)
    if start is None:
        return []
    return _nonempty_until_heading(lines[start + 1 :])


def _files_section_start(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == "## Files":
            return index
    return None


def _nonempty_until_heading(lines: list[str]) -> list[str]:
    values = []
    for line in lines:
        if line.startswith("## "):
            break
        value = line.strip()
        if value:
            values.append(value)
    return values


def _parse_files_entries(spec: str) -> list[tuple[str, str]] | None:
    entries = []
    for line in _files_section_lines(spec):
        parsed = _parse_files_line(line)
        if parsed is None:
            return None
        entries.append(parsed)
    return entries


def _spec_files(spec: str) -> list[str]:
    entries = _parse_files_entries(spec)
    if entries is None:
        return []
    return [path for path, _action in entries]


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
    write("y/n")
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
