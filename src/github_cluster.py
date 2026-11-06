"""
GitHub Actions cluster orchestration module for antigravity-cloud-render.
Manages temporary GitHub release creation, chunk upload, workflow_dispatch
triggering, live run watching via `gh run watch`, final stitched video download,
and guaranteed release/tag cleanup.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional


def create_temp_release_and_upload(
    repo: str,
    job_tag: str,
    chunk_files: List[Path],
) -> None:
    """
    Create a temporary GitHub release tagged `job_tag` and upload all `in_XXXX.mp4`
    chunk files in a single `gh release create` invocation.
    """
    if not chunk_files:
        raise ValueError("No chunk files provided to upload.")

    print(
        f"[cluster] Creating temporary release '{job_tag}' in {repo} "
        f"and uploading {len(chunk_files)} chunk(s)..."
    )
    cmd = [
        "gh",
        "release",
        "create",
        job_tag,
        *[str(f) for f in chunk_files],
        "--repo",
        repo,
        "--title",
        f"Cloud Render Job {job_tag}",
        "--notes",
        "Temporary chunk payload for parallel GitHub Actions FFmpeg rendering. Auto-deleted on completion.",
    ]
    subprocess.run(cmd, check=True)
    print(f"[cluster] Uploaded {len(chunk_files)} chunk(s) to release '{job_tag}'.")


def trigger_workflow_dispatch(
    repo: str,
    workflow: str,
    job_tag: str,
    chunk_indices: List[int],
    ffmpeg_args: str,
) -> str:
    """
    Trigger `.github/workflows/parallel-render.yml` via `gh workflow run` with
    `job_tag`, `chunks_json`, and `ffmpeg_args`.
    Returns the serialized JSON array string passed as `chunks_json`.
    """
    chunks_json = json.dumps(chunk_indices, separators=(",", ":"))
    print(
        f"[cluster] Dispatching workflow '{workflow}' on {repo}...\n"
        f"          job_tag     = {job_tag}\n"
        f"          chunks_json = {chunks_json}\n"
        f"          ffmpeg_args = {ffmpeg_args}"
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
        f"chunks_json={chunks_json}",
        "-f",
        f"ffmpeg_args={ffmpeg_args}",
    ]
    subprocess.run(cmd, check=True)
    return chunks_json


def find_workflow_run_id(
    repo: str,
    workflow: str,
    job_tag: str,
    poll_interval: int = 3,
    timeout_seconds: int = 60,
) -> int:
    """
    Locate the exact GitHub Actions workflow run ID triggered for `job_tag`.
    Because `parallel-render.yml` sets `run-name: "Cloud Render ${{ inputs.job_tag }}"`,
    matching `job_tag` inside `displayTitle` uniquely identifies this run even if
    multiple renders are dispatched concurrently.
    """
    print(f"[cluster] Waiting for workflow run 'Cloud Render {job_tag}' to register...")
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
                print(f"[cluster] Found workflow run ID {run_id}: {run_url}")
                return run_id

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for workflow run with tag '{job_tag}'."
    )


def watch_workflow_run(
    repo: str,
    run_id: int,
    poll_interval: int = 5,
) -> None:
    """
    Watch the GitHub Actions workflow run until completion using `gh run watch --exit-status`.
    Raises subprocess.CalledProcessError if any matrix chunk or stitch job fails.
    """
    print(f"[cluster] Watching workflow run {run_id} until completion...")
    cmd = [
        "gh",
        "run",
        "watch",
        str(run_id),
        "--repo",
        repo,
        "--interval",
        str(poll_interval),
        "--exit-status",
    ]
    subprocess.run(cmd, check=True)
    print(f"[cluster] Workflow run {run_id} completed successfully.")


def download_rendered_video(
    repo: str,
    job_tag: str,
    temp_dir: Path,
    output_video: Path,
) -> Path:
    """
    Download `final_rendered.mp4` from the temporary GitHub release `job_tag`
    and move it to `output_video`.
    """
    print(f"[cluster] Downloading final_rendered.mp4 from release '{job_tag}'...")
    cmd = [
        "gh",
        "release",
        "download",
        job_tag,
        "--repo",
        repo,
        "--pattern",
        "final_rendered.mp4",
        "--dir",
        str(temp_dir),
        "--clobber",
    ]
    subprocess.run(cmd, check=True)

    downloaded_file = temp_dir / "final_rendered.mp4"
    if not downloaded_file.exists():
        raise FileNotFoundError(
            f"Expected downloaded file not found at {downloaded_file}"
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_file), str(output_video))
    print(
        f"[cluster] Saved rendered video to '{output_video}' "
        f"({output_video.stat().st_size / 1024:.1f} KB)."
    )
    return output_video


def cleanup_temp_release(repo: str, job_tag: str) -> None:
    """
    Delete the temporary GitHub release and its associated git tag so repository
    storage stays clean after the render completes or aborts.
    """
    print(f"[cluster] Deleting temporary release and tag '{job_tag}' from {repo}...")
    cmd = [
        "gh",
        "release",
        "delete",
        job_tag,
        "--repo",
        repo,
        "--cleanup-tag",
        "--yes",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"[cluster] Cleaned up release and tag '{job_tag}'.")
    else:
        print(
            f"[cluster] Warning: could not delete release '{job_tag}': "
            f"{proc.stderr.strip()}"
        )
