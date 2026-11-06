"""
Configuration persistence module for antigravity-cloud-render.
Loads and saves default settings (repository name, workflow file, chunk count,
default FFmpeg arguments, and polling timeouts) using a local config.json file.
"""

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "repo": "Yeamin-Sheikh/antigravity-cloud-render",
    "workflow": "parallel-render.yml",
    "default_chunks": 4,
    "default_ffmpeg_args": "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k",
    "poll_interval_seconds": 5,
    "workflow_find_timeout_seconds": 60,
    "output_dir": "output",
    "cleanup_release": True,
}


def get_config_path(project_root: Path) -> Path:
    """Return the path to config.json in the project root."""
    return project_root / "config.json"


def load_config(project_root: Path) -> Dict[str, Any]:
    """
    Load config.json from the project root, merging with DEFAULT_CONFIG so newly
    added keys always have safe fallbacks. Creates config.json on first run if absent.
    """
    cfg_path = get_config_path(project_root)
    config = dict(DEFAULT_CONFIG)

    if cfg_path.exists():
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[config] Warning: could not parse {cfg_path} ({exc}), using defaults.")
    else:
        save_config(project_root, config)

    return config


def save_config(project_root: Path, config: Dict[str, Any]) -> None:
    """Persist configuration dictionary to config.json with readable formatting."""
    cfg_path = get_config_path(project_root)
    cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
