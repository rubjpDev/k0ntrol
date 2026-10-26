import argparse
import sys
from pathlib import Path

import yaml

from k0ntrol import __version__
from k0ntrol.config import ConfigError, load_config
from k0ntrol.frontends.banner import color_enabled
from k0ntrol.frontends.session import run_session
from k0ntrol.memory.layout import init_project
from k0ntrol.memory.scan import ScanError, run_scan


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
    init_parser = subparsers.add_parser(
        "init",
        help="create the project memory layout",
    )
    init_parser.add_argument("directory")
    subparsers.add_parser("scan", help="write the code graph")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _run_doctor()
    if args.command == "init":
        return _run_init(args.directory)
    if args.command == "scan":
        return _run_scan()
    if args.command is None:
        return _run_session()
    return 0


def _run_init(directory: str) -> int:
    for line in init_project(Path(directory)):
        print(line)
    return 0


def _run_doctor() -> int:
    try:
        cfg = load_config(Path.cwd())
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 1

    print(yaml.safe_dump(cfg, sort_keys=False), end="")
    print(_gates_report(cfg.get("gates") or []))
    return 0


def _gates_report(gates: list) -> str:
    if not gates:
        return "active gates: none"
    lines = ["active gates:"]
    for gate in gates:
        lines.append(f"  {gate['name']}: {gate['cmd']}")
    return "\n".join(lines)


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
        color=color_enabled(sys.stdout),
        echo_input=False,
    )


def _run_scan() -> int:
    try:
        run_scan(Path.cwd())
    except ScanError as error:
        print(f"scan error: {error}", file=sys.stderr)
        return 1
    return 0
