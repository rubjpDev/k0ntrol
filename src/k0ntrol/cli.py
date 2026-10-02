import argparse
import sys
from pathlib import Path

import yaml

import k0ntrol.backends.cursor
from k0ntrol import __version__
from k0ntrol.config import ConfigError, load_config


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
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _run_doctor()
    if args.command is None:
        parser.print_help()
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
