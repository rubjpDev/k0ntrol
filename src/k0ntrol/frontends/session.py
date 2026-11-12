import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain.chat_models import init_chat_model

from k0ntrol import __version__
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.config import ConfigError, load_config
from k0ntrol.events import render_event
from k0ntrol.frontends.banner import render_banner
from k0ntrol.frontends.klog import Klog
from k0ntrol.memory.reflect import (
    Reflector,
    apply_mem_decision,
    list_pending_mem_proposals,
    offer_mem_proposals,
)
from k0ntrol.workflow import Outcome, Pending, RunRefused, Unfinished
from k0ntrol.workflow import abandon as abandon_run
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
    abandon: Callable = abandon_run,
    init: Callable = init_chat_model,
    reflector=None,
    stream: Callable[[str], None] | None = None,
    ui=None,
    klog: Callable[[Path], str] | None = None,
) -> int:
    """The REPL. ``write`` prints whole lines; ``stream``, when given,
    prints ``/ask`` answer deltas without line breaks.

    ``ui`` is the terminal front (``frontends.terminal.Terminal``): when
    given it reads the prompt, renders progress and asks the gates, and
    ``read_line``/``write``/``stream`` are not used.
    """
    if ui is not None:
        read_line, write, stream = ui.read, ui.line, ui.stream
    elif stream is None:
        stream = write
    else:
        write, stream = _terminal(write, stream)
    transcript = _new_transcript(root)
    sitting = transcript.stem  # one id per sitting: Langfuse session, run_start
    chat: list[tuple[str, str]] = []  # this session's questions and answers
    show = _transcript_writer(write, transcript, events=getattr(ui, "event", None))
    front = _ui_front(ui, show, transcript) if ui else _plain_front(read_line, show)
    if reflector is None:
        reflector = Reflector(
            root=root, transcript=transcript, cfg=config, init=init,
            write=_reflect_writer(ui, show, transcript),
        )
    _welcome(ui, show, config=config, root=root, color=color)
    try:
        return _session_loop(
            read_line,
            show,
            transcript,
            echo_input=echo_input,
            root=root,
            front=front,
            commands=_commands(ui, show, config, root, klog or Klog().open),
            ask=lambda question: _run_ask(
                question,
                chat=chat,
                ask=ask,
                root=root,
                config=config,
                stream=stream,
                show=show,
                transcript=transcript,
                reflector=reflector,
                session=sitting,
            ),
            work=lambda kind, task: _run_work(
                kind,
                task,
                front=front,
                show=show,
                root=root,
                config=config,
                start=start,
                resume=resume,
                abandon=abandon,
                reflector=reflector,
                session=sitting,
            ),
        )
    finally:
        _closing(ui)
        reflector.finish()


@dataclass(frozen=True)
class _Front:
    """How the session asks the human: gates, resume, y/n, outcomes."""

    answer_gate: Callable[[Pending], str]
    confirm_resume: Callable[[Unfinished], str | None]
    offer_mem: Callable[[Path], None]
    report_outcome: Callable[[Outcome], None]


def _plain_front(read_line: Callable[[], str], show: Callable[[str], None]) -> _Front:
    def answer_gate(pending: Pending) -> str:
        _show_pending(pending, show)
        return read_line().strip()

    return _Front(
        answer_gate=answer_gate,
        confirm_resume=lambda unfinished: _ask_resume(unfinished, read_line, show),
        offer_mem=lambda root: offer_mem_proposals(root, read_line, show),
        report_outcome=lambda result: show(_outcome_text(result)),
    )


def _ui_front(ui, show: Callable[[str], None], transcript: Path) -> _Front:
    def confirm_resume(unfinished: Unfinished) -> str:
        question = f"resume run {unfinished.run} at {unfinished.node}?"
        _append_transcript(transcript, question)
        return ui.confirm(question)

    return _Front(
        answer_gate=ui.gate,
        confirm_resume=confirm_resume,
        offer_mem=lambda root: _offer_mem_with_ui(ui, root),
        report_outcome=lambda result: _append_transcript(transcript, _outcome_text(result)),
    )


def _offer_mem_with_ui(ui, root: Path) -> None:
    """Reflect's Mem.md proposals as yes/no pickers, one per proposal."""
    for item in list_pending_mem_proposals(root):
        approved = ui.confirm(f"Add to Mem.md: {item['body']}") == "y"
        apply_mem_decision(root, int(str(item["id"])), approved)


def _reflect_writer(ui, show: Callable[[str], None], transcript: Path) -> Callable[[str], None]:
    """Reflect runs in the background while the prompt waits: with a UI
    its lines go to the transcript only, never across the input line."""
    if ui is None:
        return show
    return lambda text: _append_transcript(transcript, text)


def _welcome(ui, show: Callable[[str], None], *, config: dict, root: Path, color: bool) -> None:
    provider = _chatty_provider(config)
    if ui is not None:
        ui.welcome(version=__version__, provider=provider, cwd=str(root),
                   vcs=config.get("vcs"), tracker=config.get("tracker"))
        return
    show(render_banner(version=__version__, provider=provider, cwd=str(root), color=color))


def _closing(ui) -> None:
    if ui is not None:
        ui.line("saving session memory…")


def _commands(
    ui, show: Callable[[str], None], config: dict, root: Path, klog: Callable[[Path], str]
) -> dict:
    """The slash commands besides /ask, /fast, /full and /quit."""
    help_ = ui.help if ui is not None else lambda: show(_HELP)
    rows = lambda: _status_rows(config, root)  # noqa: E731
    status = (lambda: ui.status(rows())) if ui is not None else (lambda: _show_rows(show, rows()))
    commands = {"/help": help_, "?": help_, "/status": status}
    commands["/klog"] = lambda: _show_klog(show, klog, root)
    if ui is not None:
        commands["/config"] = lambda: _reconfigure(ui, show, config, root)
    return commands


def _show_klog(show: Callable[[str], None], klog: Callable[[Path], str], root: Path) -> None:
    try:
        show(f"event log: {klog(root)}")
    except OSError as error:
        show(f"klog failed: {error}")


def _reconfigure(ui, show: Callable[[str], None], config: dict, root: Path) -> None:
    """Run the setup screens, then reload the config in place: every
    command holds this same dict, so the next run uses the new models."""
    try:
        ui.configure(root, Path.home())
        fresh = load_config(root)
    except (KeyboardInterrupt, EOFError):
        show("setup cancelled; nothing changed")
        return
    except ConfigError as error:
        show(f"config error: {error}")
        return
    config.clear()
    config.update(fresh)


_HELP = "commands: /ask, /fast, /full, /status, /klog, /quit"


def _status_rows(config: dict, root: Path) -> list[tuple[str, str]]:
    rows = [("cwd", str(root)), ("vcs", str(config.get("vcs") or "github"))]
    rows.append(("tracker", str(config.get("tracker") or "none")))
    rows.append(("test_cmd", str(config.get("test_cmd", ""))))
    for kind in ("agents", "helpers"):
        rows += [(name, _model_text(entry)) for name, entry in (config.get(kind) or {}).items()]
    return rows


def _model_text(entry: dict) -> str:
    """``provider/model``, with ``· effort`` when one is set."""
    effort = f" · {entry['effort']}" if entry.get("effort") else ""
    return f"{entry.get('provider')}/{entry.get('model')}{effort}"


def _show_rows(show: Callable[[str], None], rows: list[tuple[str, str]]) -> None:
    for name, value in rows:
        show(f"{name}: {value}")


def _terminal(
    write: Callable[[str], None], stream: Callable[[str], None]
) -> tuple[Callable[[str], None], Callable[[str], None]]:
    """Line and delta writers sharing one terminal: a line written after
    streamed text starts on a line of its own."""
    mid_line = False

    def line(text: str) -> None:
        nonlocal mid_line
        if mid_line:
            stream("\n")
            mid_line = False
        write(text)

    def delta(text: str) -> None:
        nonlocal mid_line
        stream(text)
        mid_line = not text.endswith("\n")

    return line, delta


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
    front: _Front,
    commands: dict,
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
            front=front,
            commands=commands,
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
    front: _Front,
    commands: dict,
    ask: Callable[[str], None],
    work: Callable[[str, str], None],
) -> bool:
    line = _read_session_line(read_line, front.offer_mem, root)
    if line is None:
        return True
    _echo_session_input(line, show, transcript, echo_input)
    if not line:
        return False
    return _handle_command(line, show, ask=ask, work=work, commands=commands)


# The last exchanges an answer reads, and how much of each old answer.
_CHAT_TURNS = 4
_CHAT_ANSWER_CHARS = 1500


def _chat_history(chat: list[tuple[str, str]]) -> str:
    """The conversation so far as plain text for CHATTY's next question."""
    if not chat:
        return ""
    turns = [
        f"Q: {question}\nA: {answer[:_CHAT_ANSWER_CHARS]}"
        for question, answer in chat[-_CHAT_TURNS:]
    ]
    return "Earlier in this conversation:\n\n" + "\n\n".join(turns) + "\n\nNow:"


def _run_ask(
    question: str,
    *,
    chat: list[tuple[str, str]],
    ask: Callable,
    root: Path,
    config: dict,
    stream: Callable[[str], None],
    show: Callable[[str], None],
    transcript: Path,
    reflector,
    session: str | None = None,
) -> None:
    """Stream CHATTY's answer live; the transcript keeps one entry."""
    deltas: list[str] = []

    def _on_text(delta: str) -> None:
        stream(delta)
        deltas.append(delta)

    try:
        result = ask(
            question,
            root=root,
            config=config,
            write=show,
            on_text=_on_text,
            history=_chat_history(chat),
            session=session,
        )
    except KeyboardInterrupt:
        show("answer interrupted")
        result = None
    if result is not None:
        reflector.ask_turn()
        chat.append((question, "".join(deltas)))
    if deltas:
        _append_transcript(transcript, "".join(deltas))


def _run_work(
    kind: str,
    task: str,
    *,
    front: _Front,
    show: Callable[[str], None],
    root: Path,
    config: dict,
    start: Callable,
    resume: Callable,
    abandon: Callable,
    reflector,
    session: str | None = None,
) -> None:
    """Drive one run: start, then Pending → answer → resume until it ends."""
    result = _start_or_resume(
        kind, task, front=front, show=show, root=root, config=config,
        start=start, resume=resume, abandon=abandon, session=session,
    )
    if result is None:
        return
    while isinstance(result, Pending):
        answer = _gate_answer(front, show, result)
        if answer is None:
            return
        result = _try_resume(
            result, answer, show=show, root=root, config=config, resume=resume,
            session=session,
        )
        if result is None:
            return
    front.report_outcome(result)
    reflector.after_run(result.run)


def _start_or_resume(
    kind: str,
    task: str,
    *,
    front: _Front,
    show: Callable[[str], None],
    root: Path,
    config: dict,
    start: Callable,
    resume: Callable,
    abandon: Callable,
    session: str | None = None,
):
    """Start the run; an unfinished one is resumed or abandoned first."""
    result = _try_start(
        kind, task, show=show, root=root, config=config, start=start, session=session
    )
    if not isinstance(result, Unfinished):
        return result
    answer = front.confirm_resume(result)
    if answer == "y":
        resumed = _try_resume(
            result, None, show=show, root=root, config=config, resume=resume,
            session=session,
        )
        return None if resumed is result else resumed
    if answer != "n":
        return None
    if not _try_abandon(result, kind, task, show=show, root=root, abandon=abandon):
        return None
    return _try_start(
        kind, task, show=show, root=root, config=config, start=start, session=session
    )


def _try_abandon(
    unfinished: Unfinished,
    kind: str,
    task: str,
    *,
    show: Callable[[str], None],
    root: Path,
    abandon: Callable,
) -> bool:
    """Drop the run; a refusal (dirty tree) keeps it and says how to go on."""
    try:
        abandon(unfinished.run, root=root)
    except RunRefused as error:
        _show_refusal(error, show)
        show(
            f"run {unfinished.run} is kept: clean the tree yourself, or run "
            f"/{kind} {task} again and answer y to resume it"
        )
        return False
    return True


def _ask_resume(
    unfinished: Unfinished,
    read_line: Callable[[], str],
    show: Callable[[str], None],
) -> str | None:
    """``resume run <id> at <node>? [y/n]`` until a y or n; None on EOF."""
    while True:
        show(f"resume run {unfinished.run} at {unfinished.node}? [y/n]")
        answer = _paused_on_interrupt(read_line, show, unfinished)
        if answer is None or answer.lower() in ("y", "n"):
            return answer and answer.lower()


_PAUSED = "run paused · run the same command again to resume it"


def _try_start(kind, task, *, show, root, config, start, session=None):
    try:
        return start(
            kind, task, root=root, cfg=config, write=show, session=session
        )
    except RunRefused as error:
        _show_refusal(error, show)
        return None
    except KeyboardInterrupt:
        # Ctrl-C mid-run: the checkpoint holds the last finished node.
        show(_PAUSED)
        return None
    except Exception as error:
        # A mid-run failure leaves a resumable checkpoint, never stuck.
        show(f"run failed: {error}")
        return None


def _try_resume(
    result: Pending | Unfinished,
    answer: str | None,
    *,
    show,
    root,
    config,
    resume,
    session=None,
):
    try:
        return resume(
            result.run, answer, root=root, cfg=config, write=show, session=session
        )
    except RunRefused as error:
        _show_refusal(error, show)
        return result  # bad answer: re-ask the same gate
    except KeyboardInterrupt:
        show(_PAUSED)
        return None
    except Exception as error:
        show(f"run {result.run} failed: {error}")
        return None


def _show_pending(pending: Pending, show: Callable[[str], None]) -> None:
    for part in (pending.detail, pending.diff):
        if part:
            show(part)
    answers = "/".join(pending.answers)
    show(f"{pending.question} [{answers}]")


def _gate_answer(
    front: _Front, show: Callable[[str], None], pending: Pending
) -> str | None:
    return _paused_on_interrupt(lambda: front.answer_gate(pending), show, pending)


def _paused_on_interrupt(
    read: Callable[[], str],
    show: Callable[[str], None],
    pending: Pending | Unfinished,
) -> str | None:
    """The human's answer, or None (and a pause note) on Ctrl-C / Ctrl-D."""
    try:
        return read().strip()
    except (KeyboardInterrupt, EOFError):
        where = pending.gate if isinstance(pending, Pending) else pending.node
        show(f"run {pending.run} is paused at {where}; resume it later")
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
    offer_mem: Callable[[Path], None],
    root: Path,
) -> str | None:
    try:
        offer_mem(root)
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


def _route_text(line: str, *, ask: Callable[[str], None], work: Callable[[str, str], None]) -> None:
    """Plain text is a question for CHATTY; ``#12`` alone works issue 12."""
    text = line.strip()
    if re.fullmatch(r"#\d+", text):
        work("full", text)
        return
    ask(text)


def _new_transcript(root: Path) -> Path:
    directory = root / ".k0-mem" / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}.txt"
    path.touch()
    return path


def _transcript_writer(
    write: Callable[[str], None],
    transcript: Path,
    *,
    events: Callable[[dict], None] | None = None,
) -> Callable[[str], None]:
    if events is not None:
        return _RichShow(write, transcript, events)

    def show(text: str) -> None:
        write(text)
        _append_transcript(transcript, text)

    return show


class _RichShow:
    """``show`` for the terminal UI: lines as usual, and progress events
    rendered by the UI while the transcript keeps the plain line."""

    def __init__(self, write, transcript: Path, events: Callable[[dict], None]) -> None:
        self._write = write
        self._transcript = transcript
        self._events = events

    def __call__(self, text: str) -> None:
        self._write(text)
        _append_transcript(self._transcript, text)

    def event(self, event: dict) -> None:
        self._events(event)
        line = render_event(event)
        if line is not None:
            _append_transcript(self._transcript, line)


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
    commands: dict,
) -> bool:
    command, _, argument = line.partition(" ")
    if command == "/quit":
        return True
    if not command.startswith("/") and command != "?":
        _route_text(line, ask=ask, work=work)
        return False
    handlers = {
        "/ask": ask,
        "/fast": lambda task: work("fast", task),
        "/full": lambda task: work("full", task),
    }
    handlers.update({name: lambda _arg, f=f: f() for name, f in commands.items()})
    handler = handlers.get(command)
    if handler is None:
        write(f"unknown command: {command} · /help lists them")
        return False
    handler(argument)
    return False
