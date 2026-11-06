#!/usr/bin/env python3
"""
CLI entry point for `cloud-render` (antigravity-cloud-render).
Splits a local video losslessly at keyframes into N chunks, uploads them to a
temporary GitHub Release, dispatches a parallel GitHub Actions matrix render,
watches execution live, downloads the stitched output, and deletes the temporary
release/tag.

Usage:
    cloud-render <input_video> <output_video> [--chunks N] [--args "..."]
"""

import argparse
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Resolve project root so imports work regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_cli_binaries, ensure_python_dependencies
from src.config_loader import load_config
from src.github_cluster import (
    cleanup_temp_release,
    create_temp_release_and_upload,
    download_rendered_video,
    find_workflow_run_id,
    trigger_workflow_dispatch,
    watch_workflow_run,
)
from src.splitter import split_video_lossless
from src.verifier import verify_rendered_output


def resolve_safe_output_path(input_path: Path, requested_output: Path, default_subdir: str) -> Path:
    """
    Ensure that processed media files are never dumped directly into the same directory
    as the source media file when only a bare filename or identical folder is given.
    If the resolved parent folder matches `input_path.parent`, route the file into a
    dedicated subfolder (`output/`) unless the user already placed it in a distinct folder.
    """
    resolved_in = input_path.resolve()
    resolved_out = requested_output.resolve()

    if resolved_out.parent == resolved_in.parent:
        safe_dir = resolved_out.parent / default_subdir
        safe_dir.mkdir(parents=True, exist_ok=True)
        routed = safe_dir / resolved_out.name
        print(
            f"[cloud-render] Routing output into dedicated subfolder to keep source folder clean: {routed}"
        )
        return routed

    return resolved_out


def normalize_cli_argv(argv: list[str]) -> list[str]:
    """
    Normalize CLI arguments so `--args "-c:v libx264 ..."` works seamlessly with Python's
    argparse even when the FFmpeg argument string begins with a leading hyphen (`-`).
    Also handles cases where a shell wrapper splits tokens after `--args`.
    """
    known_top_flags = {"--chunks", "-c", "--repo", "--keep-release", "-h", "--help"}
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("--args", "-a") and i + 1 < len(argv):
            # Collect the next token; if the shell split an unquoted FFmpeg flag sequence,
            # gather subsequent tokens until we hit a known cloud-render top-level flag.
            collected = [argv[i + 1]]
            j = i + 2
            # If argv[i+1] has no spaces and starts with '-', check if subsequent tokens
            # are part of the FFmpeg args rather than cloud-render flags.
            if " " not in argv[i + 1] and argv[i + 1].startswith("-"):
                while j < len(argv) and argv[j] not in known_top_flags:
                    collected.append(argv[j])
                    j += 1
            merged_val = " ".join(collected)
            normalized.append(f"--args={merged_val}")
            i = j
        else:
            normalized.append(token)
            i += 1
    return normalized


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line flags for cloud-render."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    clean_argv = normalize_cli_argv(raw_argv)

    parser = argparse.ArgumentParser(
        prog="cloud-render",
        description="Offload FFmpeg video renders to parallel 4-vCPU GitHub Actions runners.",
    )
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to the source input video file.",
    )
    parser.add_argument(
        "output_video",
        type=str,
        help="Destination path for the rendered output video file.",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=int,
        default=None,
        help="Number of parallel keyframe chunks to split into (default from config.json: 4).",
    )
    parser.add_argument(
        "--args",
        "-a",
        type=str,
        default=None,
        help="Exact FFmpeg encoding arguments applied to each chunk.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Override GitHub repository (owner/name) from config.json.",
    )
    parser.add_argument(
        "--keep-release",
        action="store_true",
        help="Do not delete the temporary GitHub release/tag after completion (for debugging).",
    )
    return parser.parse_args(clean_argv)


def main(argv=None) -> int:
    """Run the end-to-end parallel cloud render pipeline."""
    ensure_python_dependencies(PROJECT_ROOT)
    ensure_cli_binaries()

    config = load_config(PROJECT_ROOT)
    args = parse_args(argv)

    input_video = Path(args.input_video).resolve()
    if not input_video.exists():
        print(f"[error] Input video not found: {input_video}", file=sys.stderr)
        return 1

    requested_out = Path(args.output_video)
    output_video = resolve_safe_output_path(
        input_video,
        requested_out,
        config.get("output_dir", "output"),
    )

    repo = args.repo or config["repo"]
    workflow = config["workflow"]
    num_chunks = args.chunks if args.chunks is not None else int(config["default_chunks"])
    ffmpeg_args = args.args if args.args is not None else config["default_ffmpeg_args"]
    poll_interval = int(config.get("poll_interval_seconds", 5))
    find_timeout = int(config.get("workflow_find_timeout_seconds", 60))

    # Generate a unique temporary release tag for this render job
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    short_uid = uuid.uuid4().hex[:6]
    job_tag = f"render-{timestamp}-{short_uid}"

    start_time = time.time()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"cloud_render_{short_uid}_"))
    release_created = False

    try:
        # Step 1: Split input video losslessly at keyframes into N chunks
        chunk_files = split_video_lossless(
            input_video=input_video,
            chunks_dir=temp_dir,
            target_chunks=num_chunks,
        )
        chunk_indices = list(range(len(chunk_files)))

        # Step 2: Create temporary GitHub Release and upload in_XXXX.mp4 segments
        create_temp_release_and_upload(
            repo=repo,
            job_tag=job_tag,
            chunk_files=chunk_files,
        )
        release_created = True

        # Step 3: Trigger parallel-render.yml workflow_dispatch
        trigger_workflow_dispatch(
            repo=repo,
            workflow=workflow,
            job_tag=job_tag,
            chunk_indices=chunk_indices,
            ffmpeg_args=ffmpeg_args,
        )

        # Step 4: Find the workflow run ID and watch until completion
        run_id = find_workflow_run_id(
            repo=repo,
            workflow=workflow,
            job_tag=job_tag,
            poll_interval=3,
            timeout_seconds=find_timeout,
        )
        watch_workflow_run(
            repo=repo,
            run_id=run_id,
            poll_interval=poll_interval,
        )

        # Step 5: Download final_rendered.mp4 from the release to output_video
        download_rendered_video(
            repo=repo,
            job_tag=job_tag,
            temp_dir=temp_dir,
            output_video=output_video,
        )

        # Step 6: Verify output streams and duration with ffprobe
        verify_rendered_output(
            input_video=input_video,
            output_video=output_video,
        )

        elapsed = time.time() - start_time
        print(f"[cloud-render] Completed end-to-end parallel render in {elapsed:.1f}s -> {output_video}")
        return 0

    finally:
        # Always clean up the temporary GitHub release/tag and local scratch folder
        if release_created and not args.keep_release and config.get("cleanup_release", True):
            cleanup_temp_release(repo=repo, job_tag=job_tag)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
