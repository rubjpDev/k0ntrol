from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol import __version__
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.banner import render_banner
from k0ntrol.memory.reflect import offer_mem_proposals


def run_session(
    read_line: Callable[[], str],
    write: Callable[[str], None],
    *,
    root: Path,
    config: dict,
    color: bool = False,
    echo_input: bool = True,
    ask: Callable = answer_ask,
) -> int:
    transcript = _new_transcript(root)
    show = _transcript_writer(write, transcript)
    show(
        render_banner(
            version=__version__,
            provider=_chatty_provider(config),
            cwd=str(root),
            color=color,
        )
    )
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
        ),
    )


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
) -> int:
    while True:
        if _session_should_quit(
            read_line,
            show,
            transcript,
            echo_input=echo_input,
            root=root,
            ask=ask,
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
) -> bool:
    line = _read_session_line(read_line, show, root)
    if line is None:
        return True
    _echo_session_input(line, show, transcript, echo_input)
    if not line:
        return False
    return _handle_command(line, show, ask=ask)


def _run_ask(
    question: str,
    *,
    ask: Callable,
    root: Path,
    config: dict,
    write: Callable[[str], None],
    show: Callable[[str], None],
    transcript: Path,
) -> None:
    """Stream CHATTY's answer live; the transcript keeps one entry."""
    deltas: list[str] = []
    ask(
        question,
        root=root,
        config=config,
        write=show,
        on_text=lambda delta: (write(delta), deltas.append(delta)),
    )
    if deltas:
        _append_transcript(transcript, "".join(deltas))


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
    line: str, write: Callable[[str], None], *, ask: Callable[[str], None]
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
    if command in {"/fast", "/full"}:
        write(f"{command} is not available yet")
        return False
    write(f"unknown command: {command}")
    return False
