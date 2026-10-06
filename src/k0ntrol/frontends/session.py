from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol.backends.invoke import BackendError
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.banner import render_banner
from k0ntrol.frontends.mentions import MentionError
from k0ntrol.harness.loop import run_fast, run_full
from k0ntrol import __version__


def run_session(
    read_line: Callable[[], str],
    write: Callable[[str], None],
    *,
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
    run_suite: Callable[[Path], tuple[int, str]] | None = None,
    color: bool = False,
    echo_input: bool = True,
) -> int:
    transcript = _new_transcript(root)
    show = _transcript_writer(write, transcript)
    show(
        render_banner(
            version=__version__,
            backend=config["backend"],
            cwd=str(root),
            color=color,
        )
    )
    while True:
        try:
            line = read_line()
        except (KeyboardInterrupt, EOFError):
            return 0
        input_line = f"› {line}"
        if echo_input:
            show(input_line)
        else:
            _append_transcript(transcript, input_line)
        if not line:
            continue
        if _handle_command(
            line,
            show,
            read_line=read_line,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
            run_suite=run_suite,
        ):
            return 0


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
    read_line: Callable[[], str],
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
    run_suite: Callable[[Path], tuple[int, str]] | None,
) -> bool:
    parts = line.split(maxsplit=1)
    command = parts[0]
    if command == "/quit":
        return True
    if command == "/help":
        write("commands: /ask, /fast, /full, /quit")
    elif command == "/fast":
        run_fast(
            parts[1] if len(parts) == 2 else "",
            root=root,
            config=config,
            read_line=read_line,
            write=write,
            invoke=invoke,
            bulk_read=bulk_read,
            run_suite=run_suite,
        )
    elif command == "/full":
        run_full(
            parts[1] if len(parts) == 2 else "",
            root=root,
            config=config,
            read_line=read_line,
            write=write,
            invoke=invoke,
            bulk_read=bulk_read,
        )
    elif command == "/ask":
        _handle_ask(
            parts[1] if len(parts) == 2 else "",
            write,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
        )
    else:
        write(f"unknown command: {command}")
    return False


def _handle_ask(
    text: str,
    write: Callable[[str], None],
    *,
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
) -> None:
    try:
        answer = answer_ask(
            text,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
        )
    except MentionError as error:
        write(str(error))
    except BackendError as error:
        write(f"backend error: {error}")
    else:
        write(answer)
