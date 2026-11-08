#!/usr/bin/env python3
"""
CLI entry point for `cloud-ltx` (LTX-Video-2B-Distilled parallel cloud generator).
Dispatches one or more text-to-video scene prompts to parallel GitHub Actions VMs,
runs the two-stage memory-optimized LTX-Video-2B-Distilled pipeline, stitches scenes
losslessly, downloads the rendered MP4, verifies streams with ffprobe, and cleans up
the temporary GitHub release.

Usage:
    cloud-ltx "<prompt>" <output_video> [--width 512] [--height 320] [--frames 25] [--fps 24] [--steps 8] [--seed 42]
    cloud-ltx --prompt "Scene 1..." --prompt "Scene 2..." <output_video>
"""

import argparse
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import ensure_cli_binaries, ensure_python_dependencies
from src.config_loader import load_config
from src.github_cluster import cleanup_temp_release, watch_workflow_run
from src.ltx_cluster import (
    create_empty_temp_release,
    download_ltx_video,
    find_ltx_run_id,
    trigger_ltx_workflow,
)
from src.splitter import probe_video_info


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for cloud-ltx."""
    parser = argparse.ArgumentParser(
        prog="cloud-ltx",
        description="Generate AI videos on parallel GitHub Actions VMs using LTX-Video-2B-Distilled.",
    )
    parser.add_argument(
        "positional",
        nargs="*",
        help="Either `<prompt> <output_video>` or `<output_video>` when `--prompt` is used.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        action="append",
        dest="prompts",
        default=[],
        help="Scene prompt (can be specified multiple times for parallel multi-scene generation).",
    )
    parser.add_argument("--width", "-W", type=int, default=512, help="Video width (multiple of 32, default: 512).")
    parser.add_argument("--height", "-H", type=int, default=320, help="Video height (multiple of 32, default: 320).")
    parser.add_argument("--frames", "-f", type=int, default=25, help="Frame count 8*k+1 (default: 25).")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate (default: 24).")
    parser.add_argument("--steps", "-s", type=int, default=8, help="Distilled denoising steps (default: 8).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42).")
    parser.add_argument("--repo", type=str, default=None, help="Override GitHub repository (owner/name).")
    parser.add_argument("--keep-release", action="store_true", help="Do not delete temporary GitHub release.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Execute end-to-end LTX-Video-2B-Distilled cloud generation."""
    ensure_python_dependencies(PROJECT_ROOT)
    ensure_cli_binaries()

    config = load_config(PROJECT_ROOT)
    args = parse_args(argv)

    prompts = list(args.prompts)
    if len(args.positional) == 2 and not prompts:
        prompts = [args.positional[0]]
        output_arg = args.positional[1]
    elif len(args.positional) == 1 and prompts:
        output_arg = args.positional[0]
    elif len(args.positional) >= 2 and prompts:
        prompts.extend(args.positional[:-1])
        output_arg = args.positional[-1]
    else:
        print(
            "Usage: cloud-ltx \"<prompt>\" <output_video> [--width 512] [--height 320] [--frames 25]",
            file=sys.stderr,
        )
        return 1

    output_video = Path(output_arg).resolve()
    # Route bare filenames in project root into output/ subfolder
    if output_video.parent == PROJECT_ROOT:
        output_video = (PROJECT_ROOT / config.get("output_dir", "output") / output_video.name).resolve()

    repo = args.repo or config["repo"]
    workflow = config.get("ltx_workflow", "ltx-generate.yml")
    poll_interval = int(config.get("poll_interval_seconds", 5))
    find_timeout = int(config.get("workflow_find_timeout_seconds", 60))

    scenes = [
        {"index": idx, "prompt": p, "seed": args.seed + idx}
        for idx, p in enumerate(prompts)
    ]

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    short_uid = uuid.uuid4().hex[:6]
    job_tag = f"ltx-{timestamp}-{short_uid}"

    start_time = time.time()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"cloud_ltx_{short_uid}_"))
    release_created = False

    try:
        create_empty_temp_release(repo=repo, job_tag=job_tag)
        release_created = True

        trigger_ltx_workflow(
            repo=repo,
            workflow=workflow,
            job_tag=job_tag,
            scenes=scenes,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            fps=args.fps,
            steps=args.steps,
        )

        run_id = find_ltx_run_id(
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

        download_ltx_video(
            repo=repo,
            job_tag=job_tag,
            temp_dir=temp_dir,
            output_video=output_video,
        )

        info = probe_video_info(output_video)
        elapsed = time.time() - start_time
        print(
            f"[cloud-ltx] Verified output: {info['duration']:.2f}s | "
            f"{info['width']}x{info['height']} @ {info['r_frame_rate']}fps | "
            f"v:{info['video_codec']} a:{info['audio_codec']} | "
            f"{info['size_bytes'] / 1024:.1f} KB"
        )
        print(f"[cloud-ltx] Completed LTX-Video cloud generation in {elapsed:.1f}s -> {output_video}")
        return 0

    finally:
        if release_created and not args.keep_release and config.get("cleanup_release", True):
            cleanup_temp_release(repo=repo, job_tag=job_tag)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
