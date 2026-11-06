"""
Environment and dependency bootstrapper for antigravity-cloud-render.
Auto-checks Python packages from requirements.txt on first run and validates
that required external CLI binaries (ffmpeg, ffprobe, gh) are available on PATH.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_python_dependencies(project_root: Path) -> None:
    """
    Check if third-party packages listed in requirements.txt are installed.
    If any package is missing, install requirements.txt automatically via pip
    so first-run execution requires zero manual environment setup.
    """
    req_file = project_root / "requirements.txt"
    if not req_file.exists():
        return

    missing = []
    for raw_line in req_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        # Extract base package name before version specifiers
        pkg_name = (
            line.split(">=")[0]
            .split("==")[0]
            .split("<=")[0]
            .split("~=")[0]
            .split(">")[0]
            .split("<")[0]
            .strip()
        )
        if pkg_name and importlib.util.find_spec(pkg_name.replace("-", "_")) is None:
            missing.append(pkg_name)

    if missing:
        print(f"[bootstrap] Installing missing Python dependencies: {', '.join(missing)}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
            check=True,
        )


def ensure_cli_binaries() -> None:
    """
    Verify that ffmpeg, ffprobe, and GitHub CLI (gh) are accessible on system PATH.
    Raises RuntimeError with actionable instructions if any binary is missing.
    """
    required_bins = ["ffmpeg", "ffprobe", "gh"]
    missing_bins = [b for b in required_bins if shutil.which(b) is None]
    if missing_bins:
        raise RuntimeError(
            f"Missing required CLI tools on PATH: {', '.join(missing_bins)}. "
            "Install FFmpeg and GitHub CLI (gh) before running cloud-render."
        )
