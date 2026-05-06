import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "immunoforge.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_top_level_help_accepts_h_alias():
    result = _run_cli("--h")

    assert result.returncode == 0
    assert "usage: immunoforge" in result.stdout
    assert "{run,qc,targets,codon}" in result.stdout
    assert "-h, --help, --h" in result.stdout


def test_subcommand_help_accepts_h_alias():
    result = _run_cli("run", "--h")

    assert result.returncode == 0
    assert "usage: immunoforge run" in result.stdout
    assert "--config" in result.stdout
    assert "--steps" in result.stdout
