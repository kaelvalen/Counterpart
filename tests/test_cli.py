"""CLI smoke tests: every command must at least register and render help.

This guards against typos in option definitions that only surface at import time
(found the hard way: ``exist_ok`` vs ``exists`` broke the whole CLI).
"""

from __future__ import annotations

from typer.testing import CliRunner

from counterpart.cli import app

runner = CliRunner()

COMMANDS = [
    "demo",
    "prepare",
    "generate",
    "evaluate",
    "score",
    "baselines",
    "select",
    "viz",
]


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_each_command_help() -> None:
    for command in COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{command}: {result.output}"
