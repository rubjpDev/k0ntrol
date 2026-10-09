import threading
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from k0ntrol.backends.invoke import BackendError, set_event_context
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.banner import render_banner
from k0ntrol.frontends.mentions import MentionError
from k0ntrol.harness.loop import run_fast, run_full
from k0ntrol.harness.reflect import run_reflect
from k0ntrol.harness.scan import rescan_changed
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
    session_run = uuid4().hex
    set_event_context(run=session_run, step="")
    transcript = _new_transcript(root)
    show = _transcript_writer(write, transcript)
    hinted = {"scan": False}
    reflect = _ReflectScheduler(
        root=root,
        config=config,
        invoke=invoke,
        transcript=transcript,
        run=session_run,
    )
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
            session_run=session_run,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
            run_suite=run_suite,
            hinted=hinted,
            reflect=reflect,
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
    session_run: str,
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
    run_suite: Callable[[Path], tuple[int, str]] | None,
    hinted: dict[str, bool],
    reflect: "_ReflectScheduler",
) -> bool:
    parts = line.split(maxsplit=1)
    command = parts[0]
    rest = parts[1] if len(parts) == 2 else ""
    if command == "/quit":
        reflect.flush()
        return True
    if command == "/help":
        write("commands: /ask, /fast, /full, /quit")
        return False
    if command in {"/ask", "/fast", "/full"}:
        _refresh_memory(
            write,
            read_line=read_line,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
            hinted=hinted,
        )
        status = _run_work_command(
            command,
            rest,
            write,
            read_line=read_line,
            session_run=session_run,
            root=root,
            config=config,
            invoke=invoke,
            bulk_read=bulk_read,
            run_suite=run_suite,
        )
        reflect.note_turn(status)
        return False
    write(f"unknown command: {command}")
    return False


def _refresh_memory(
    write: Callable[[str], None],
    *,
    read_line: Callable[[], str],
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
    hinted: dict[str, bool],
) -> None:
    status = rescan_changed(
        root=root,
        config=config,
        bulk_read=bulk_read,
        invoke=invoke,
        read_line=read_line,
        write=write,
    )
    if status != "missing" or hinted["scan"]:
        return
    write("memory not built; run k0 scan")
    hinted["scan"] = True


def _run_work_command(
    command: str,
    rest: str,
    write: Callable[[str], None],
    *,
    read_line: Callable[[], str],
    session_run: str,
    root: Path,
    config: dict,
    invoke: Callable,
    bulk_read: Callable,
    run_suite: Callable[[Path], tuple[int, str]] | None,
) -> str | None:
    if command == "/fast":
        status = run_fast(
            rest,
            root=root,
            config=config,
            read_line=read_line,
            write=write,
            invoke=invoke,
            bulk_read=bulk_read,
            run_suite=run_suite,
        )
        write(_terminal_status(status))
        return status
    if command == "/full":
        status = run_full(
            rest,
            root=root,
            config=config,
            read_line=read_line,
            write=write,
            invoke=invoke,
            bulk_read=bulk_read,
            run_suite=run_suite,
        )
        write(_terminal_status(status))
        return status
    set_event_context(run=session_run, step="ask")
    _handle_ask(
        rest,
        write,
        root=root,
        config=config,
        invoke=invoke,
        bulk_read=bulk_read,
    )
    return None


def _terminal_status(status: str) -> str:
    return "done" if status == "tested" else status


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


class _ReflectScheduler:
    def __init__(
        self,
        *,
        root: Path,
        config: dict,
        invoke: Callable,
        transcript: Path,
        run: str,
    ) -> None:
        self._root = root
        self._config = config
        self._invoke = invoke
        self._transcript = transcript
        self._run = run
        self._every = config.get("reflect_every", 10)
        self._pending = 0
        self._threads: list[threading.Thread] = []

    def note_turn(self, status: str | None) -> None:
        self._pending += 1
        if status == "done" or self._pending >= self._every:
            self._pending = 0
            self._spawn()

    def flush(self) -> None:
        if self._pending:
            self._pending = 0
            self._spawn()
        for thread in self._threads:
            thread.join()

    def _spawn(self) -> None:
        thread = threading.Thread(target=self._run_reflect, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _run_reflect(self) -> None:
        run_reflect(
            root=self._root,
            config=self._config,
            invoke=self._invoke,
            transcript=self._transcript,
            run=self._run,
        )
