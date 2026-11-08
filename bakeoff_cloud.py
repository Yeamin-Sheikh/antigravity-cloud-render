#!/usr/bin/env python3
"""
CLI entry point for `cloud-bakeoff` (Multi-VM 9:16 15-second Influencer Video Bake-Off).
Launches 6 parallel GitHub Actions VMs simultaneously to compare LTX-Video-2B-Distilled,
Wan 2.1 1.3B, and SkyReels-V2-DF-1.3B at both 720p (720x1280) and 1080p (1080x1920)
9:16 vertical aspect ratios for a 15-second influencer clip, downloads all 6 outputs into
`output/bakeoff/`, verifies every file with ffprobe, and cleans up the temporary release.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_cli_binaries, ensure_python_dependencies
from src.config_loader import load_config
from src.github_cluster import cleanup_temp_release, watch_workflow_run
from src.ltx_cluster import create_empty_temp_release
from src.splitter import probe_video_info

DEFAULT_INFLUENCER_PROMPT = (
    "Vertical 9:16 close-up portrait video of a charismatic South Asian lifestyle influencer "
    "talking to the camera with expressive natural hand gestures in a sunlit modern studio, "
    "soft rim lighting, shallow depth of field, sharp facial details, photorealistic 4k footage"
)

DEFAULT_BAKEOFF_MATRIX = [
    {
        "id": "1_ltx2b_distilled_720p",
        "engine": "ltx-2b-distilled",
        "quality": "720p (720x1280)",
        "base_w": 192,
        "base_h": 320,
        "frames": 9,
        "steps": 4,
        "target_w": 720,
        "target_h": 1280,
        "seed": 101,
    },
    {
        "id": "2_ltx2b_distilled_1080p",
        "engine": "ltx-2b-distilled",
        "quality": "1080p (1080x1920)",
        "base_w": 192,
        "base_h": 320,
        "frames": 9,
        "steps": 6,
        "target_w": 1080,
        "target_h": 1920,
        "seed": 101,
    },
    {
        "id": "3_wan21_13b_720p",
        "engine": "wan-2.1-1.3b",
        "quality": "720p (720x1280)",
        "base_w": 192,
        "base_h": 320,
        "frames": 9,
        "steps": 4,
        "target_w": 720,
        "target_h": 1280,
        "seed": 202,
    },
    {
        "id": "4_wan21_13b_1080p",
        "engine": "wan-2.1-1.3b",
        "quality": "1080p (1080x1920)",
        "base_w": 192,
        "base_h": 320,
        "frames": 9,
        "steps": 4,
        "target_w": 1080,
        "target_h": 1920,
        "seed": 202,
    },
    {
        "id": "5_skyreels_v2_13b_720p",
        "engine": "skyreels-v2-1.3b",
        "quality": "720p (720x1280)",
        "base_w": 192,
        "base_h": 320,
        "frames": 9,
        "steps": 4,
        "target_w": 720,
        "target_h": 1280,
        "seed": 303,
    },
    {
        "id": "6_skyreels_v2_13b_1080p",
        "engine": "skyreels-v2-1.3b",
        "quality": "1080p (1080x1920)",
        "base_w": 192,
        "base_h": 320,
        "frames": 9,
        "steps": 4,
        "target_w": 1080,
        "target_h": 1920,
        "seed": 303,
    },
]


def find_bakeoff_run_id(repo: str, workflow: str, job_tag: str, timeout_seconds: int = 60) -> int:
    """Find the workflow run ID matching `Bakeoff 9x16 <job_tag>`."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        cmd = [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--limit",
            "10",
            "--json",
            "databaseId,displayTitle,url",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for run in json.loads(proc.stdout or "[]"):
            if job_tag in (run.get("displayTitle") or ""):
                run_id = int(run["databaseId"])
                print(f"[bakeoff] Found workflow run ID {run_id}: {run.get('url', '')}")
                return run_id
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for bake-off workflow run '{job_tag}'.")


def main(argv=None) -> int:
    ensure_python_dependencies(PROJECT_ROOT)
    ensure_cli_binaries()
    config = load_config(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Run 6-VM 9:16 15s influencer bake-off on GitHub Actions.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_INFLUENCER_PROMPT, help="9:16 influencer prompt.")
    parser.add_argument("--duration", type=float, default=15.0, help="Target clip duration in seconds (default 15.0).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "output" / "bakeoff"),
        help="Directory to download all bake-off MP4 outputs.",
    )
    parser.add_argument("--keep-release", action="store_true", help="Do not delete temporary release.")
    args = parser.parse_args(argv)

    repo = config["repo"]
    workflow = "bakeoff-9x16.yml"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    short_uid = uuid.uuid4().hex[:6]
    job_tag = f"bakeoff-{timestamp}-{short_uid}"
    configs_json = json.dumps(DEFAULT_BAKEOFF_MATRIX, separators=(",", ":"))

    start_time = time.time()
    release_created = False

    try:
        create_empty_temp_release(repo=repo, job_tag=job_tag)
        release_created = True

        print(f"[bakeoff] Dispatching {len(DEFAULT_BAKEOFF_MATRIX)} parallel VMs on {repo}...")
        cmd_dispatch = [
            "gh",
            "workflow",
            "run",
            workflow,
            "--repo",
            repo,
            "-f",
            f"job_tag={job_tag}",
            "-f",
            f"prompt={args.prompt}",
            "-f",
            f"configs_json={configs_json}",
            "-f",
            f"duration={args.duration}",
        ]
        subprocess.run(cmd_dispatch, check=True)

        run_id = find_bakeoff_run_id(repo=repo, workflow=workflow, job_tag=job_tag)
        watch_workflow_run(repo=repo, run_id=run_id, poll_interval=5)

        print(f"[bakeoff] Downloading all generated 9:16 MP4 files into {out_dir}...")
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                job_tag,
                "--repo",
                repo,
                "--pattern",
                "bakeoff_*.mp4",
                "--dir",
                str(out_dir),
                "--clobber",
            ],
            check=True,
        )

        downloaded_files = sorted(out_dir.glob("bakeoff_*.mp4"))
        print(f"\n[bakeoff] Verified {len(downloaded_files)} output video(s) in {out_dir}:")
        for mp4 in downloaded_files:
            info = probe_video_info(mp4)
            print(
                f"  - {mp4.name}: {info['duration']:.2f}s | "
                f"{info['width']}x{info['height']} (9:16) @ {info['r_frame_rate']}fps | "
                f"v:{info['video_codec']} a:{info['audio_codec']} | "
                f"{info['size_bytes'] / 1024:.1f} KB"
            )

        print(f"[bakeoff] Completed {len(downloaded_files)}-VM bake-off in {time.time() - start_time:.1f}s.")
        return 0

    finally:
        if release_created and not args.keep_release:
            cleanup_temp_release(repo=repo, job_tag=job_tag)


if __name__ == "__main__":
    sys.exit(main())
