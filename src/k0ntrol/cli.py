import argparse
import sys
from pathlib import Path

from langchain.chat_models import init_chat_model

from k0ntrol import __version__
from k0ntrol.config import ConfigError, load_config
from k0ntrol.credentials import load_credentials
from k0ntrol.events import (
    EventWriter,
    K0EventHandler,
    ProgressRenderer,
    events_path,
    new_run_id,
)
from k0ntrol.frontends.banner import color_enabled
from k0ntrol.frontends.session import run_session
from k0ntrol.frontends.terminal import Terminal, needs_project_setup, needs_provider_setup
from k0ntrol.helpers import helper_invoker
from k0ntrol.init_flow import run_init
from k0ntrol.memory.refresh import refresh_memory
from k0ntrol.otel import build_run_trace, trace_callbacks
from k0ntrol.providers import free_checks, has_errors, live_checks, report_lines
from k0ntrol.workflow.change import current_branch


def main(argv: list[str] | None = None, *, init=init_chat_model) -> int:
    parser = argparse.ArgumentParser(
        prog="k0",
        description="Local harness for coding agents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"k0 {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check the provider setup without spending tokens",
    )
    doctor_parser.add_argument(
        "--live",
        action="store_true",
        help="make one tiny tool-call round trip per configured model",
    )
    init_parser = subparsers.add_parser(
        "init",
        help="create the project memory layout",
    )
    init_parser.add_argument("directory")
    init_parser.add_argument(
        "--tracker",
        help="work-item tracker for the project (github; others not in 0.1.0)",
    )
    init_parser.add_argument(
        "--provider",
        help="one provider for every agent and helper (user config)",
    )
    subparsers.add_parser("scan", help="write the code graph")
    args = parser.parse_args(argv)
    # Keys saved by the setup screen; variables already set always win.
    load_credentials(Path.home())

    if args.command == "doctor":
        return _run_doctor(live=args.live)
    if args.command == "init":
        return _run_init(args.directory, tracker=args.tracker, provider=args.provider)
    if args.command == "scan":
        return _run_scan(init=init)
    if args.command is None:
        return _run_session(init=init)
    return 0


def _run_init(
    directory: str, *, tracker: str | None = None, provider: str | None = None
) -> int:
    return run_init(Path(directory), tracker=tracker, provider=provider)


def _run_doctor(live: bool) -> int:
    try:
        cfg = load_config(Path.cwd())
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1

    checks = free_checks(cfg, root=Path.cwd())
    if live:
        checks += live_checks(cfg)
    print("\n".join(report_lines(checks)))
    return 1 if has_errors(checks) else 0


def _run_session(*, init=init_chat_model) -> int:
    root = Path.cwd()
    ui = _terminal_ui()
    if ui is not None and not _first_run(ui, root):
        return 1
    try:
        config = load_config(root)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1
    return run_session(
        lambda: input("› "),
        print,
        root=root,
        config=config,
        color=color_enabled(sys.stdout),
        echo_input=False,
        init=init,
        stream=lambda text: print(text, end="", flush=True),
        ui=ui,
    )


def _first_run(ui: Terminal, root: Path) -> bool:
    """The setup screens on a new repo or machine; False when cancelled."""
    if needs_project_setup(root) or needs_provider_setup(Path.home()):
        ui.welcome(version=__version__, provider=None, cwd=str(root))
    try:
        ui.first_run(root, Path.home())
    except (KeyboardInterrupt, EOFError):
        print("\nsetup cancelled; run k0 again to finish it")
        return False
    return True


def _terminal_ui() -> Terminal | None:
    """The Claude Code-style front in a real terminal; plain lines otherwise."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    return Terminal.open(history=Path.home() / ".k0ntrol" / "history")


def _run_scan(*, init=init_chat_model, tracing=build_run_trace) -> int:
    """Rescan in place: changed files get notes from the bulk_read helper,
    then the FTS index is rebuilt — run-scoped events as it works.

    The scan is its own Langfuse/OTel trace (id = its run id, kind ``scan``).
    """
    root = Path.cwd()
    run_id = new_run_id()
    trace = tracing(run_id)
    code = _scan(root, init, run_id, trace)
    trace.finish(
        kind="scan",
        outcome="done" if code == 0 else "failed",
        metadata=_scan_metadata(root),
    )
    return code


def _scan(root: Path, init, run_id: str, trace) -> int:
    try:
        config = load_config(root)
        callbacks = [_scan_handler(root, run_id), *trace_callbacks(trace)]
        invoke = helper_invoker(
            config, "bulk_read", init=init, config={"callbacks": callbacks}
        )
        refresh_memory(root, config=config, invoke=invoke)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"scan error: {error}", file=sys.stderr)
        return 1
    return 0


def _scan_metadata(root: Path) -> dict:
    """The scan trace's metadata: branch and k0 version."""
    metadata = {"k0_version": __version__}
    branch = current_branch(root)
    if branch:
        metadata["branch"] = branch
    return metadata


def _scan_handler(root: Path, run_id: str) -> K0EventHandler:
    path = events_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(path, run=run_id, progress=ProgressRenderer(print))
    return K0EventHandler(writer)
