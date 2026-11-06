"""
Post-render media verification module for antigravity-cloud-render.
Compares the downloaded stitched video against the original source video using
ffprobe to verify stream integrity, codec presence, and duration accuracy.
"""

from pathlib import Path
from typing import Any, Dict

from src.splitter import probe_video_info


def verify_rendered_output(
    input_video: Path,
    output_video: Path,
    tolerance_seconds: float = 0.5,
) -> Dict[str, Any]:
    """
    Probe both `input_video` and `output_video` with ffprobe and verify that:
    1. `output_video` exists and is non-empty.
    2. `output_video` has a valid video stream (and audio stream if source had audio).
    3. `output_video` duration matches `input_video` duration within `tolerance_seconds`.
    """
    src_info = probe_video_info(input_video)
    out_info = probe_video_info(output_video)

    duration_diff = abs(out_info["duration"] - src_info["duration"])
    duration_ok = duration_diff <= tolerance_seconds

    print(
        f"[verifier] Source  : {src_info['duration']:.3f}s | "
        f"{src_info['width']}x{src_info['height']} | "
        f"v:{src_info['video_codec']} a:{src_info['audio_codec']} | "
        f"{src_info['size_bytes'] / 1024:.1f} KB"
    )
    print(
        f"[verifier] Rendered: {out_info['duration']:.3f}s | "
        f"{out_info['width']}x{out_info['height']} | "
        f"v:{out_info['video_codec']} a:{out_info['audio_codec']} | "
        f"{out_info['size_bytes'] / 1024:.1f} KB "
        f"(delta={duration_diff:.3f}s)"
    )

    if out_info["video_codec"] == "unknown" or out_info["width"] <= 0 or out_info["height"] <= 0:
        raise RuntimeError(
            f"Rendered video is missing a valid video stream: {out_info}"
        )

    if src_info["has_audio"] and not out_info["has_audio"]:
        raise RuntimeError(
            "Source video contained an audio stream, but rendered video has no audio stream."
        )

    if not duration_ok:
        raise RuntimeError(
            f"Rendered video duration ({out_info['duration']:.3f}s) deviates from "
            f"source ({src_info['duration']:.3f}s) by {duration_diff:.3f}s "
            f"(tolerance={tolerance_seconds}s)."
        )

    return {
        "source": src_info,
        "output": out_info,
        "duration_diff": duration_diff,
        "verified": True,
    }
