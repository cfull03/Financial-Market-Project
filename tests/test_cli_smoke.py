from __future__ import annotations

import subprocess


def test_cli_help_runs() -> None:
    # Just a smoke test to ensure entrypoint is installed and runnable
    cp = subprocess.run(["dsproj", "--help"], capture_output=True, text=True)
    assert cp.returncode == 0
    assert "Commands" in cp.stdout or "Usage" in cp.stdout
