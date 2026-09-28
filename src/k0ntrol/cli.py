import argparse

from k0ntrol import __version__


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
    parser.add_subparsers(dest="command")
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
    return 0
