from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.banner import render_banner
from k0ntrol.frontends.mentions import MentionError
from k0ntrol import __version__


def run_session(
    read_line: Callable[[], str],
    write: Callable[[str], None],
    *,
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
    color: bool = False,
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
        show(f"› {line}")
        if not line:
            continue
        if _handle_command(
            line,
            show,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
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
        with transcript.open("a", encoding="utf-8") as file:
            file.write(text)
            if not text.endswith("\n"):
                file.write("\n")

    return show


def _handle_command(
    line: str,
    write: Callable[[str], None],
    *,
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
) -> bool:
    parts = line.split(maxsplit=1)
    command = parts[0]
    if command == "/quit":
        return True
    if command == "/help":
        write("commands: /ask, /fast, /full, /quit")
    elif command == "/fast":
        write("/fast is not in 0.0.1")
    elif command == "/full":
        write("/full is not in 0.0.1")
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
    else:
        write(answer)
