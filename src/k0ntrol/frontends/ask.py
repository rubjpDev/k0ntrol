import re
from collections.abc import Callable
from pathlib import Path

from k0ntrol.harness.delegate import should_delegate
from k0ntrol.harness.retrieve import has_code_memory, memory_context, rules_text
from k0ntrol.modespec import ModeSpec, load_mode
from k0ntrol.frontends.mentions import MentionError, mention_paths


def answer_ask(
    text: str,
    *,
    root: Path,
    config: dict,
    invoke: Callable[[ModeSpec, str], str],
    bulk_read: Callable,
) -> str:
    paths = mention_paths(text, root)
    if not paths and not has_code_memory(root):
        raise MentionError("ask requires at least one @file mention")
    question = re.sub(r"@[^\s]+", "", text).strip()
    prompt_question = _question_with_memory(question, root, config, invoke)
    if not paths:
        return invoke(load_mode("bulk_reader", config), prompt_question)
    return _ask_with_files(prompt_question, paths, root, config, invoke, bulk_read)


def _question_with_memory(
    question: str,
    root: Path,
    config: dict,
    invoke: Callable[[ModeSpec, str], str],
) -> str:
    sections = [question]
    rules = rules_text(root)
    if rules:
        sections.extend(["", rules])
    if not has_code_memory(root):
        return "\n".join(sections)
    retrieved = memory_context(question, root=root, config=config, invoke=invoke)
    if retrieved:
        sections.extend(["", retrieved])
    return "\n".join(sections)


def _ask_with_files(
    question: str,
    paths: list[Path],
    root: Path,
    config: dict,
    invoke: Callable[[ModeSpec, str], str],
    bulk_read: Callable,
) -> str:
    delegated = [path for path in paths if should_delegate(path, config["threshold_lines"])]
    direct = [path for path in paths if path not in delegated]
    mode = load_mode("bulk_reader", config)
    results = []
    if delegated:
        results.append(
            bulk_read(
                question,
                delegated,
                root=root,
                repo=root.name,
                mode=mode,
                invoke=invoke,
            )
        )
    if direct:
        results.append(invoke(mode, _direct_prompt(question, direct, root)))
    return "".join(results)


def _direct_prompt(question: str, paths: list[Path], root: Path) -> str:
    sections = [_file_section(path, root) for path in paths]
    return "\n".join([question, "", *sections])


def _file_section(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return f"{relative}\n{path.read_text(encoding='utf-8')}"
