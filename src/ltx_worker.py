#!/usr/bin/env python3
"""
GitHub Actions runner worker for LTX-Video-2B-Distilled generation.
Executes a two-stage memory-optimized pipeline so peak RAM stays well below
the 16 GB limit of a standard GitHub Actions ubuntu-latest VM:
  Stage 1: Load T5-XXL bf16 text encoder (city96/t5-v1_1-xxl-encoder-bf16),
           compute prompt embeddings, then immediately free RAM and disk cache.
  Stage 2: Load LTX-Video-2B-0.9.6-Distilled transformer + v0.9.5 timestep-aware VAE
           in bfloat16, run 7 guidance-distilled steps (guidance_scale=1.0),
           decode with VAE tiling, and encode a clean yuv420p H.264 MP4 via FFmpeg.
"""

import argparse
import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXConditionPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, T5TokenizerFast

# Enable fast Rust-based Hugging Face Hub downloads if hf_transfer is installed
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

T5_BF16_REPO = "city96/t5-v1_1-xxl-encoder-bf16"
DISTILLED_2B_REPO = "multimodalart/ltxv-2b-0.9.6-distilled"
BASE_095_REPO = "Lightricks/LTX-Video-0.9.5"

# Official 8-step distilled schedule for LTX-Video distilled checkpoints
DISTILLED_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]


def align_dimensions(width: int, height: int, num_frames: int) -> tuple[int, int, int]:
    """
    Ensure width and height are multiples of 32 and (num_frames - 1) is a multiple of 8
    as required by LTX-Video's 32x32x8 spatiotemporal VAE compression ratios.
    """
    aligned_w = max(64, width - (width % 32))
    aligned_h = max(64, height - (height % 32))
    rem = (num_frames - 1) % 8
    aligned_f = num_frames if rem == 0 else max(9, num_frames - rem)
    return aligned_w, aligned_h, aligned_f


def purge_hf_repo_cache(repo_id: str) -> None:
    """
    Delete a specific Hugging Face hub cached repository folder from disk after its
    tensors are no longer needed, keeping runner SSD usage well below 14 GB.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub"
    folder_name = "models--" + repo_id.replace("/", "--")
    target = hub_dir / folder_name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print(f"[ltx-worker] Reclaimed disk cache for {repo_id}")


def encode_prompt_stage1(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1: Load only the T5-XXL text encoder in bfloat16, encode the text prompt,
    and immediately unload the 4.7B text encoder from memory and disk.
    Because LTX-Video-2B-Distilled uses guidance_scale=1.0 (no classifier-free guidance),
    negative prompt embeddings are not needed, cutting prompt encoding work in half.
    """
    t0 = time.time()
    print(f"[ltx-worker] Stage 1: Loading bf16 T5-XXL encoder from {T5_BF16_REPO}...")
    tokenizer = T5TokenizerFast.from_pretrained(T5_BF16_REPO)
    text_encoder = T5EncoderModel.from_pretrained(
        T5_BF16_REPO,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    pipe_text = LTXConditionPipeline(
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
    )

    print(f"[ltx-worker] Encoding prompt: {prompt!r}")
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, _, _ = pipe_text.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=128,
            device="cpu",
            dtype=torch.bfloat16,
        )

    # Detach and clone tensors before freeing the 9 GB text encoder
    prompt_embeds = prompt_embeds.detach().clone()
    prompt_attention_mask = prompt_attention_mask.detach().clone()

    del pipe_text, text_encoder, tokenizer
    gc.collect()
    purge_hf_repo_cache(T5_BF16_REPO)
    print(f"[ltx-worker] Stage 1 complete in {time.time() - t0:.1f}s (T5 unloaded).")
    return prompt_embeds, prompt_attention_mask


def generate_video_stage2(
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    steps: int,
    output_path: Path,
) -> None:
    """
    Stage 2: Load the LTX-Video-2B-Distilled transformer and timestep-conditioned VAE
    in bfloat16 (~4.9 GB RAM total), run distilled denoising steps with guidance_scale=1.0,
    and mux a standard H.264 yuv420p MP4 file.
    """
    t0 = time.time()
    print(f"[ltx-worker] Stage 2: Loading LTX-Video-2B-Distilled transformer from {DISTILLED_2B_REPO}...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        DISTILLED_2B_REPO,
        subfolder="scheduler",
    )
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        DISTILLED_2B_REPO,
        subfolder="transformer",
        variant="bf16",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"[ltx-worker] Loading LTX-Video 0.9.5 timestep-aware VAE from {BASE_095_REPO}...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        BASE_095_REPO,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    pipe = LTXConditionPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )
    pipe.vae.enable_tiling()

    # Select custom distilled timesteps if using default 8 steps, else use step count
    custom_timesteps = DISTILLED_TIMESTEPS if steps in (7, 8) else None
    num_inference_steps = None if custom_timesteps is not None else steps

    print(
        f"[ltx-worker] Running distilled inference: {width}x{height}, "
        f"{num_frames} frames @ {fps}fps, steps={steps}, seed={seed}..."
    )
    gen_start = time.time()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        frames = pipe(
            conditions=None,
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=num_inference_steps,
            timesteps=custom_timesteps,
            guidance_scale=1.0,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            image_cond_noise_scale=0.0,
            generator=generator,
            output_type="pil",
        ).frames[0]

    print(f"[ltx-worker] Diffusion + VAE decode finished in {time.time() - gen_start:.1f}s.")

    raw_mp4 = output_path.with_suffix(".raw.mp4")
    export_to_video(frames, str(raw_mp4), fps=fps)

    # Re-mux with FFmpeg to standard H.264 yuv420p + silent stereo AAC track so all
    # players and lossless concat jobs handle the output seamlessly.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(raw_mp4),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=48000",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    raw_mp4.unlink(missing_ok=True)
    print(
        f"[ltx-worker] Saved {output_path} ({output_path.stat().st_size / 1024:.1f} KB) "
        f"in {time.time() - t0:.1f}s total."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-Video-2B-Distilled worker for GitHub Actions.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--output", type=str, required=True, help="Output MP4 file path.")
    parser.add_argument("--width", type=int, default=512, help="Video width (multiple of 32).")
    parser.add_argument("--height", type=int, default=320, help="Video height (multiple of 32).")
    parser.add_argument("--num-frames", type=int, default=25, help="Frame count (8*k + 1, e.g. 25, 49, 81).")
    parser.add_argument("--fps", type=int, default=24, help="Output frame rate.")
    parser.add_argument("--steps", type=int, default=7, help="Distilled denoising steps (default: 7).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Use all vCPUs available on the GitHub Actions VM
    cpu_threads = os.cpu_count() or 4
    torch.set_num_threads(cpu_threads)
    print(f"[ltx-worker] Configured PyTorch with {cpu_threads} CPU threads.")

    width, height, num_frames = align_dimensions(args.width, args.height, args.num_frames)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_embeds, prompt_attention_mask = encode_prompt_stage1(args.prompt)
    generate_video_stage2(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=args.fps,
        seed=args.seed,
        steps=args.steps,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
