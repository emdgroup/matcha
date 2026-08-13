"""Command-line interface for MATCHA molecular property prediction.

Dispatches subcommands (train, evaluate, predict, autotune, etc.) to their
respective modules. Each subcommand accepts a ``--config`` argument pointing
to a YAML configuration file validated against Pydantic schemas in
:mod:`matcha.utils.schemas.cli`.
"""

import sys
import importlib

COMMANDS = {
    "train": "matcha.cli.train",
    "predict": "matcha.cli.predict",
    "evaluate": "matcha.cli.evaluate",
    "baseline": "matcha.cli.baseline",
    "autotune": "matcha.cli.autotune",
    "stitch": "matcha.cli.stitch",
    "summarize": "matcha.cli.summarize",
    "prepare_dataset": "matcha.cli.prepare_dataset",
    "pretrain_multitask": "matcha.cli.pretrain_multitask",
    "pretrain_encoder": "matcha.cli.pretrain_encoder",
}


def show_help():
    """Display help information"""
    print("Usage: matcha <command> [options]")
    print("\nAvailable commands:")
    for cmd in sorted(COMMANDS.keys()):
        print(f"  {cmd}")
    print("\nFor command-specific help, use: matcha <command> --help")


def main():
    """Main entry point for the CLI"""
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        show_help()
        return 0

    command = sys.argv[1]

    if command not in COMMANDS:
        print(f"Error: Unknown command '{command}'")
        show_help()
        return 1

    # Remove the command from argv so the subcommand sees the correct arguments
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Import and run the appropriate module
    try:
        module = importlib.import_module(COMMANDS[command])
        return module.main()
    except Exception as e:
        print(f"Error running command '{command}': {e}")
        return 1


if __name__ == "main":
    sys.exit(main())
