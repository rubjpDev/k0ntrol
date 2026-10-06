import argparse
import sys
from pathlib import Path

import yaml

import k0ntrol.backends.cursor
import k0ntrol.backends.invoke
from k0ntrol import __version__
from k0ntrol.config import ConfigError, load_config
from k0ntrol.frontends.banner import color_enabled
from k0ntrol.frontends.session import run_session
from k0ntrol.modespec import load_mode
from k0ntrol.stages.bulk_read import BulkReadError, bulk_read


def main(argv: list[str] | None = None) -> int:
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
    subparsers.add_parser("doctor", help="print the effective configuration")
    bulk_read_parser = subparsers.add_parser(
        "bulk-read",
        help="ask the bulk-reader mode about files",
    )
    bulk_read_parser.add_argument("--question", required=True)
    bulk_read_parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _run_doctor()
    if args.command == "bulk-read":
        return _run_bulk_read(args.question, args.paths)
    if args.command is None:
        return _run_session()
    return 0


def _run_doctor() -> int:
    try:
        cfg = load_config(Path.cwd())
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1

    print(yaml.safe_dump(cfg, sort_keys=False), end="")

    problems = k0ntrol.backends.cursor.cursor_problems(cfg)
    if problems:
        for problem in problems:
            print(f"cursor: {problem}", file=sys.stderr)
        return 1

    return 0


def _run_session() -> int:
    root = Path.cwd()
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
        invoke=k0ntrol.backends.invoke.invoke,
        bulk_read=bulk_read,
        color=color_enabled(sys.stdout),
    )


def _run_bulk_read(question: str, paths: list[str]) -> int:
    root = Path.cwd()
    try:
        config = load_config(root)
        mode = load_mode("bulk_reader", config)
        result = bulk_read(
            question,
            [root / path for path in paths],
            root=root,
            repo=root.name,
            mode=mode,
            invoke=k0ntrol.backends.invoke.invoke,
        )
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1
    except BulkReadError as error:
        print(f"bulk-read error: {error}", file=sys.stderr)
        return 1
    except k0ntrol.backends.invoke.BackendError as error:
        print(f"backend error: {error}", file=sys.stderr)
        return 1

    print(result, end="" if result.endswith("\n") else "\n")
    return 0
