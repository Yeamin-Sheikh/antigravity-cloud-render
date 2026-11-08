"""
GitHub Actions orchestration module for LTX-Video-2B-Distilled cloud generation.
Manages temporary release creation, multi-scene matrix workflow_dispatch,
live run watching via `gh run watch`, final stitched video download, and release cleanup.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from src.github_cluster import cleanup_temp_release, watch_workflow_run


def create_empty_temp_release(repo: str, job_tag: str) -> None:
    """
    Create a temporary GitHub release tagged `job_tag` to receive `final_ltx.mp4`
    once the GitHub Actions LTX-Video runners complete.
    """
    print(f"[ltx-cluster] Creating temporary release '{job_tag}' in {repo}...")
    cmd = [
        "gh",
        "release",
        "create",
        job_tag,
        "--repo",
        repo,
        "--title",
        f"LTX Cloud Gen {job_tag}",
        "--notes",
        "Temporary release for LTX-Video-2B-Distilled cloud generation. Auto-deleted on completion.",
    ]
    subprocess.run(cmd, check=True)


def trigger_ltx_workflow(
    repo: str,
    workflow: str,
    job_tag: str,
    scenes: List[Dict[str, Any]],
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    steps: int,
) -> str:
    """
    Dispatch `.github/workflows/ltx-generate.yml` with the JSON array of scene prompts.
    Each item in `scenes` runs on its own parallel 4-vCPU GitHub Actions VM.
    """
    scenes_json = json.dumps(scenes, separators=(",", ":"))
    print(
        f"[ltx-cluster] Dispatching '{workflow}' on {repo} across {len(scenes)} runner(s)...\n"
        f"              job_tag    = {job_tag}\n"
        f"              resolution = {width}x{height}, {num_frames} frames @ {fps}fps ({steps} steps)"
    )
    cmd = [
        "gh",
        "workflow",
        "run",
        workflow,
        "--repo",
        repo,
        "-f",
        f"job_tag={job_tag}",
        "-f",
        f"scenes_json={scenes_json}",
        "-f",
        f"width={width}",
        "-f",
        f"height={height}",
        "-f",
        f"num_frames={num_frames}",
        "-f",
        f"fps={fps}",
        "-f",
        f"steps={steps}",
    ]
    subprocess.run(cmd, check=True)
    return scenes_json


def find_ltx_run_id(
    repo: str,
    workflow: str,
    job_tag: str,
    poll_interval: int = 3,
    timeout_seconds: int = 60,
) -> int:
    """
    Locate the exact GitHub Actions workflow run ID triggered for `LTX Cloud Gen <job_tag>`.
    """
    print(f"[ltx-cluster] Waiting for workflow run 'LTX Cloud Gen {job_tag}' to register...")
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
            "databaseId,displayTitle,status,conclusion,url",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        runs = json.loads(proc.stdout or "[]")
        for run in runs:
            title = run.get("displayTitle") or ""
            if job_tag in title:
                run_id = int(run["databaseId"])
                run_url = run.get("url", "")
                print(f"[ltx-cluster] Found workflow run ID {run_id}: {run_url}")
                return run_id

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for LTX workflow run '{job_tag}'."
    )


def download_ltx_video(
    repo: str,
    job_tag: str,
    temp_dir: Path,
    output_video: Path,
) -> Path:
    """
    Download `final_ltx.mp4` from the temporary release `job_tag` and move it to `output_video`.
    """
    print(f"[ltx-cluster] Downloading final_ltx.mp4 from release '{job_tag}'...")
    cmd = [
        "gh",
        "release",
        "download",
        job_tag,
        "--repo",
        repo,
        "--pattern",
        "final_ltx.mp4",
        "--dir",
        str(temp_dir),
        "--clobber",
    ]
    subprocess.run(cmd, check=True)

    downloaded = temp_dir / "final_ltx.mp4"
    if not downloaded.exists():
        raise FileNotFoundError(f"Expected generated file not found at {downloaded}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded), str(output_video))
    print(
        f"[ltx-cluster] Saved generated video to '{output_video}' "
        f"({output_video.stat().st_size / 1024:.1f} KB)."
    )
    return output_video
