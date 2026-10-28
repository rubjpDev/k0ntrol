from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from langchain.chat_models import init_chat_model

from k0ntrol import __version__
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.banner import render_banner
from k0ntrol.memory.reflect import Reflector, offer_mem_proposals
from k0ntrol.workflow import Outcome, Pending, RunRefused
from k0ntrol.workflow import resume as resume_run
from k0ntrol.workflow import start as start_run


def run_session(
    read_line: Callable[[], str],
    write: Callable[[str], None],
    *,
    root: Path,
    config: dict,
    color: bool = False,
    echo_input: bool = True,
    ask: Callable = answer_ask,
    start: Callable = start_run,
    resume: Callable = resume_run,
    init: Callable = init_chat_model,
    reflector=None,
) -> int:
    transcript = _new_transcript(root)
    show = _transcript_writer(write, transcript)
    if reflector is None:
        reflector = Reflector(
            root=root, transcript=transcript, cfg=config, init=init, write=show
        )
    show(
        render_banner(
            version=__version__,
            provider=_chatty_provider(config),
            cwd=str(root),
            color=color,
        )
    )
    try:
        return _session_loop(
            read_line,
            show,
            transcript,
            echo_input=echo_input,
            root=root,
            ask=lambda question: _run_ask(
                question,
                ask=ask,
                root=root,
                config=config,
                write=write,
                show=show,
                transcript=transcript,
                reflector=reflector,
            ),
            work=lambda kind, task: _run_work(
                kind,
                task,
                read_line=read_line,
                show=show,
                root=root,
                config=config,
                start=start,
                resume=resume,
                reflector=reflector,
            ),
        )
    finally:
        reflector.finish()


def _chatty_provider(config: dict) -> str | None:
    chatty = (config.get("agents") or {}).get("chatty") or {}
    return chatty.get("provider")


def _session_loop(
    read_line: Callable[[], str],
    show: Callable[[str], None],
    transcript: Path,
    *,
    echo_input: bool,
    root: Path,
    ask: Callable[[str], None],
    work: Callable[[str, str], None],
) -> int:
    while True:
        if _session_should_quit(
            read_line,
            show,
            transcript,
            echo_input=echo_input,
            root=root,
            ask=ask,
            work=work,
        ):
            return 0


def _session_should_quit(
    read_line: Callable[[], str],
    show: Callable[[str], None],
    transcript: Path,
    *,
    echo_input: bool,
    root: Path,
    ask: Callable[[str], None],
    work: Callable[[str, str], None],
) -> bool:
    line = _read_session_line(read_line, show, root)
    if line is None:
        return True
    _echo_session_input(line, show, transcript, echo_input)
    if not line:
        return False
    return _handle_command(line, show, ask=ask, work=work)


def _run_ask(
    question: str,
    *,
    ask: Callable,
    root: Path,
    config: dict,
    write: Callable[[str], None],
    show: Callable[[str], None],
    transcript: Path,
    reflector,
) -> None:
    """Stream CHATTY's answer live; the transcript keeps one entry."""
    deltas: list[str] = []
    result = ask(
        question,
        root=root,
        config=config,
        write=show,
        on_text=lambda delta: (write(delta), deltas.append(delta)),
    )
    if result is not None:
        reflector.ask_turn()
    if deltas:
        _append_transcript(transcript, "".join(deltas))


def _run_work(
    kind: str,
    task: str,
    *,
    read_line: Callable[[], str],
    show: Callable[[str], None],
    root: Path,
    config: dict,
    start: Callable,
    resume: Callable,
    reflector,
) -> None:
    """Drive one run: start, then Pending → answer → resume until it ends."""
    result = _try_start(kind, task, show=show, root=root, config=config, start=start)
    if result is None:
        return
    while isinstance(result, Pending):
        _show_pending(result, show)
        answer = _gate_answer(read_line, show, result)
        if answer is None:
            return
        result = _try_resume(result, answer, show=show, root=root, config=config, resume=resume)
        if result is None:
            return
    show(_outcome_text(result))
    reflector.after_run(result.run)


def _try_start(kind, task, *, show, root, config, start):
    try:
        return start(kind, task, root=root, cfg=config, write=show)
    except RunRefused as error:
        _show_refusal(error, show)
        return None
    except Exception as error:
        # A mid-run failure leaves a resumable checkpoint, never stuck.
        show(f"run failed: {error}")
        return None


def _try_resume(result: Pending, answer: str, *, show, root, config, resume):
    try:
        return resume(result.run, answer, root=root, cfg=config, write=show)
    except RunRefused as error:
        _show_refusal(error, show)
        return result  # bad answer: re-ask the same gate
    except Exception as error:
        show(f"run {result.run} failed: {error}")
        return None


def _show_pending(pending: Pending, show: Callable[[str], None]) -> None:
    if pending.detail:
        show(pending.detail)
    answers = "/".join(pending.answers)
    show(f"{pending.question} [{answers}]")


def _gate_answer(
    read_line: Callable[[], str],
    show: Callable[[str], None],
    pending: Pending,
) -> str | None:
    try:
        return read_line().strip()
    except (KeyboardInterrupt, EOFError):
        show(
            f"run {pending.run} is paused at gate {pending.gate}; "
            "resume it later"
        )
        return None


def _show_refusal(error: RunRefused, show: Callable[[str], None]) -> None:
    for line in error.lines:
        show(line)


def _outcome_text(result: Outcome) -> str:
    text = f"run {result.outcome}"
    if result.reason:
        text += f" ({result.reason}"
        if result.node:
            text += f" at {result.node}"
        text += ")"
    return text


def _read_session_line(
    read_line: Callable[[], str],
    show: Callable[[str], None],
    root: Path,
) -> str | None:
    try:
        offer_mem_proposals(root, read_line, show)
        return read_line()
    except (KeyboardInterrupt, EOFError):
        return None


def _echo_session_input(
    line: str,
    show: Callable[[str], None],
    transcript: Path,
    echo_input: bool,
) -> None:
    input_line = f"› {line}"
    if echo_input:
        show(input_line)
        return
    _append_transcript(transcript, input_line)


def _new_transcript(root: Path) -> Path:
    directory = root / ".k0-mem" / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}.txt"
    path.touch()
    return path


def _transcript_writer(
    write: Callable[[str], None],
    transcript: Path,
) -> Callable[[str], None]:
    def show(text: str) -> None:
        write(text)
        _append_transcript(transcript, text)

    return show


def _append_transcript(transcript: Path, text: str) -> None:
    with transcript.open("a", encoding="utf-8") as file:
        file.write(text)
        if not text.endswith("\n"):
            file.write("\n")


def _handle_command(
    line: str,
    write: Callable[[str], None],
    *,
    ask: Callable[[str], None],
    work: Callable[[str, str], None],
) -> bool:
    command, _, argument = line.partition(" ")
    if command == "/quit":
        return True
    if command == "/help":
        write("commands: /ask, /fast, /full, /quit")
        return False
    if command == "/ask":
        ask(argument)
        return False
    if command == "/fast":
        work("fast", argument)
        return False
    if command == "/full":
        work("full", argument)
        return False
    write(f"unknown command: {command}")
    return False
