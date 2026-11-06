"""
Lossless keyframe video splitter module for antigravity-cloud-render.
Probes input media duration via ffprobe and uses FFmpeg's segment muxer
(-f segment -c copy -reset_timestamps 1) to divide the input video into N
independent chunks (in_0000.mp4, in_0001.mp4, ...) without re-encoding.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def probe_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Inspect a video file with ffprobe and return structured metadata including
    duration, video codec, width, height, frame rate, and stream count.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # Prefer video stream duration if available to avoid slight audio padding skew
    duration = float(video_stream.get("duration") or fmt.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"Could not determine positive duration for {video_path}")

    return {
        "duration": duration,
        "format_duration": float(fmt.get("duration") or duration),
        "size_bytes": int(fmt.get("size") or 0),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "r_frame_rate": video_stream.get("r_frame_rate", "0/0"),
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "streams_count": len(streams),
    }


def split_video_lossless(
    input_video: Path,
    chunks_dir: Path,
    target_chunks: int,
) -> List[Path]:
    """
    Split `input_video` losslessly at keyframes into `in_XXXX.mp4` segments inside `chunks_dir`.

    Why we subtract a small 20ms epsilon from (duration / target_chunks):
    FFmpeg's `-f segment` muxer only splits when a keyframe's PTS is >= the next
    target segment boundary. If a 12.0s video has keyframes at 0.0s, 3.0s, 6.0s, 9.0s
    and audio padding makes container duration 12.01s, then `12.01 / 4 = 3.0025s`,
    causing FFmpeg to miss the 3.0000s keyframe and wait until 4.0s. Subtracting a
    small epsilon (0.02s, less than 1 frame at 30fps) ensures exact keyframe hits
    at 3.0s, 6.0s, and 9.0s while keeping `-f segment -c copy -reset_timestamps 1`.
    """
    if target_chunks < 1:
        raise ValueError("target_chunks must be >= 1")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    # Remove any stale in_*.mp4 chunks from prior runs in this folder
    for old_file in chunks_dir.glob("in_*.mp4"):
        old_file.unlink()

    info = probe_video_info(input_video)
    duration = info["duration"]

    # Calculate segment duration with a half-frame epsilon so exact keyframe intervals
    # (e.g. 3.000s on a 12.000s clip) trigger a split instead of overshooting to the next GOP.
    raw_seg_time = duration / float(target_chunks)
    seg_time = max(0.2, raw_seg_time - 0.02) if target_chunks > 1 else duration + 10.0

    output_pattern = str(chunks_dir / "in_%04d.mp4")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-f",
        "segment",
        "-segment_time",
        f"{seg_time:.4f}",
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        output_pattern,
    ]

    print(
        f"[splitter] Splitting '{input_video.name}' ({duration:.2f}s, "
        f"{info['width']}x{info['height']}) into ~{target_chunks} keyframe chunks "
        f"(segment_time={seg_time:.3f}s)..."
    )
    subprocess.run(cmd, check=True)

    chunk_files = sorted(chunks_dir.glob("in_*.mp4"))
    if not chunk_files:
        raise RuntimeError(f"FFmpeg segment split produced zero chunks in {chunks_dir}")

    print(f"[splitter] Created {len(chunk_files)} lossless chunk(s):")
    for cf in chunk_files:
        cf_info = probe_video_info(cf)
        print(
            f"  - {cf.name}: {cf_info['duration']:.2f}s "
            f"({cf.stat().st_size / 1024:.1f} KB)"
        )

    return chunk_files
