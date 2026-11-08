#!/usr/bin/env python3
"""
Multi-model 9:16 vertical influencer video bake-off worker for GitHub Actions VMs.
Supports three distinct open-source video diffusion architectures on 16 GB RAM CPU VMs:
  1. ltx-2b-distilled (multimodalart/ltxv-2b-0.9.6-distilled + Lightricks/LTX-Video-0.9.5 VAE)
  2. wan-2.1-1.3b     (Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
  3. skyreels-v2-1.3b (Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)

Uses two-stage memory offloading (Stage 1 text encoder -> free RAM & SSD -> Stage 2
native AVX2 float32 compute) followed by FFmpeg 8.0 Lanczos + CAS mastering to produce
exact 15.0-second 9:16 vertical outputs in 720p (720x1280) or 1080p (1080x1920).
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
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    LTXConditionPipeline,
    LTXVideoTransformer3DModel,
    SkyReelsV2DiffusionForcingPipeline,
    SkyReelsV2Transformer3DModel,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast, UMT5EncoderModel

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

T5_BF16_REPO = "city96/t5-v1_1-xxl-encoder-bf16"
LTX_096_DISTILLED_REPO = "multimodalart/ltxv-2b-0.9.6-distilled"
LTX_095_BASE_REPO = "Lightricks/LTX-Video-0.9.5"
WAN_13B_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SKYREELS_13B_REPO = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"


def purge_hf_repo_cache(repo_id: str) -> None:
    """Remove cached Hugging Face repo files from disk once tensors are unloaded."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub"
    target = hub_dir / ("models--" + repo_id.replace("/", "--"))
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print(f"[bakeoff-worker] Reclaimed SSD cache for {repo_id}")


def run_ltx_engine(
    prompt: str,
    base_w: int,
    base_h: int,
    base_frames: int,
    steps: int,
    seed: int,
    raw_mp4: Path,
) -> None:
    """
    Execute Stage 1 (T5-XXL prompt encoding) and Stage 2 (LTX-Video-2B-Distilled
    diffusion in native AVX2 float32) to generate the base 9:16 vertical clip.
    """
    t0 = time.time()
    print(f"[bakeoff-worker] [ltx-2b-distilled] Stage 1: Loading T5-XXL from {T5_BF16_REPO}...")
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
    # Cast embeddings to float32 so Stage 2 runs on native AVX2 FP32 FMA units
    prompt_embeds = prompt_embeds.detach().to(torch.float32).clone()
    prompt_attention_mask = prompt_attention_mask.detach().clone()

    del pipe_text, text_encoder, tokenizer
    gc.collect()
    purge_hf_repo_cache(T5_BF16_REPO)
    print(f"[bakeoff-worker] [ltx-2b-distilled] Stage 1 finished in {time.time() - t0:.1f}s.")

    t1 = time.time()
    print(f"[bakeoff-worker] [ltx-2b-distilled] Stage 2: Loading {LTX_096_DISTILLED_REPO} transformer + VAE...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        LTX_096_DISTILLED_REPO,
        subfolder="scheduler",
    )
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        LTX_096_DISTILLED_REPO,
        subfolder="transformer",
        variant="bf16",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(torch.float32)
    vae = AutoencoderKLLTXVideo.from_pretrained(
        LTX_095_BASE_REPO,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(torch.float32)

    pipe = LTXConditionPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )
    pipe.vae.enable_tiling()

    if steps <= 4:
        timesteps = [1000, 981, 909, 0.03]
    elif steps <= 6:
        timesteps = [1000, 993, 981, 909, 725, 0.03]
    else:
        timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03]

    print(
        f"[bakeoff-worker] [ltx-2b-distilled] Running {len(timesteps)} distilled steps "
        f"in native FP32 ({base_w}x{base_h}, {base_frames} frames)..."
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        frames = pipe(
            conditions=None,
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            width=base_w,
            height=base_h,
            num_frames=base_frames,
            frame_rate=24,
            num_inference_steps=None,
            timesteps=timesteps,
            guidance_scale=1.0,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            image_cond_noise_scale=0.0,
            generator=generator,
            output_type="pil",
        ).frames[0]

    export_to_video(frames, str(raw_mp4), fps=12)
    print(f"[bakeoff-worker] [ltx-2b-distilled] Stage 2 finished in {time.time() - t1:.1f}s.")


def run_wan_engine(
    prompt: str,
    base_w: int,
    base_h: int,
    base_frames: int,
    steps: int,
    seed: int,
    raw_mp4: Path,
) -> None:
    """
    Execute Stage 1 (UMT5-XXL prompt encoding) and Stage 2 (Wan 2.1 1.3B diffusion
    in native AVX2 float32) to generate the base 9:16 vertical clip.
    """
    t0 = time.time()
    print(f"[bakeoff-worker] [wan-2.1-1.3b] Stage 1: Loading UMT5-XXL from {WAN_13B_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(WAN_13B_REPO, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        WAN_13B_REPO,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe_text = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=None,
        scheduler=None,
        transformer=None,
    )
    with torch.no_grad():
        prompt_embeds, _ = pipe_text.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device="cpu",
            dtype=torch.bfloat16,
        )
    prompt_embeds = prompt_embeds.detach().to(torch.float32).clone()

    del pipe_text, text_encoder, tokenizer
    gc.collect()
    purge_hf_repo_cache(WAN_13B_REPO)
    print(f"[bakeoff-worker] [wan-2.1-1.3b] Stage 1 finished in {time.time() - t0:.1f}s.")

    t1 = time.time()
    print(f"[bakeoff-worker] [wan-2.1-1.3b] Stage 2: Loading 1.3B transformer + VAE from {WAN_13B_REPO}...")
    scheduler = UniPCMultistepScheduler.from_pretrained(WAN_13B_REPO, subfolder="scheduler")
    transformer = WanTransformer3DModel.from_pretrained(
        WAN_13B_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        WAN_13B_REPO,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    pipe = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
    )

    wan_frames = max(9, base_frames - ((base_frames - 1) % 4))
    wan_w = max(64, base_w - (base_w % 16))
    wan_h = max(64, base_h - (base_h % 16))

    print(
        f"[bakeoff-worker] [wan-2.1-1.3b] Running {steps} steps in native FP32 "
        f"({wan_w}x{wan_h}, {wan_frames} frames)..."
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        frames = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            height=wan_h,
            width=wan_w,
            num_frames=wan_frames,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
            output_type="pil",
        ).frames[0]

    export_to_video(frames, str(raw_mp4), fps=12)
    print(f"[bakeoff-worker] [wan-2.1-1.3b] Stage 2 finished in {time.time() - t1:.1f}s.")


def run_skyreels_engine(
    prompt: str,
    base_w: int,
    base_h: int,
    base_frames: int,
    steps: int,
    seed: int,
    raw_mp4: Path,
) -> None:
    """
    Execute Stage 1 (UMT5-XXL prompt encoding) and Stage 2 (SkyReels-V2-DF-1.3B
    diffusion forcing in native AVX2 float32) to generate the base 9:16 vertical clip.
    """
    t0 = time.time()
    print(f"[bakeoff-worker] [skyreels-v2-1.3b] Stage 1: Loading UMT5-XXL from {SKYREELS_13B_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(SKYREELS_13B_REPO, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        SKYREELS_13B_REPO,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe_text = SkyReelsV2DiffusionForcingPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=None,
        scheduler=None,
        transformer=None,
    )
    with torch.no_grad():
        prompt_embeds, _ = pipe_text.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device="cpu",
            dtype=torch.bfloat16,
        )
    prompt_embeds = prompt_embeds.detach().to(torch.float32).clone()

    del pipe_text, text_encoder, tokenizer
    gc.collect()
    purge_hf_repo_cache(SKYREELS_13B_REPO)
    print(f"[bakeoff-worker] [skyreels-v2-1.3b] Stage 1 finished in {time.time() - t0:.1f}s.")

    t1 = time.time()
    print(f"[bakeoff-worker] [skyreels-v2-1.3b] Stage 2: Loading 1.3B DF transformer + VAE from {SKYREELS_13B_REPO}...")
    scheduler = UniPCMultistepScheduler.from_pretrained(SKYREELS_13B_REPO, subfolder="scheduler")
    transformer = SkyReelsV2Transformer3DModel.from_pretrained(
        SKYREELS_13B_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        SKYREELS_13B_REPO,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    pipe = SkyReelsV2DiffusionForcingPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
    )

    sr_frames = max(9, base_frames - ((base_frames - 1) % 4))
    sr_w = max(64, base_w - (base_w % 16))
    sr_h = max(64, base_h - (base_h % 16))

    print(
        f"[bakeoff-worker] [skyreels-v2-1.3b] Running {steps} steps in native FP32 "
        f"({sr_w}x{sr_h}, {sr_frames} frames)..."
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        frames = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            height=sr_h,
            width=sr_w,
            num_frames=sr_frames,
            base_num_frames=97,
            ar_step=0,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
            output_type="pil",
        ).frames[0]

    export_to_video(frames, str(raw_mp4), fps=12)
    print(f"[bakeoff-worker] [skyreels-v2-1.3b] Stage 2 finished in {time.time() - t1:.1f}s.")


def master_to_9x16_15s(
    raw_mp4: Path,
    output_mp4: Path,
    target_w: int,
    target_h: int,
    target_duration: float = 15.0,
    target_fps: int = 24,
) -> None:
    """
    Master the base 9:16 vertical AI clip to exact 720p (720x1280) or 1080p (1080x1920)
    at 24 fps and 15.0 seconds using FFmpeg 8.0:
      1. Creates a seamless forward-reverse boomerang cycle (`[0:v]reverse[r];[0:v][r]concat=n=2:v=1:a=0`)
         interpolated to smooth 24 fps so motion flows continuously without hard cuts.
      2. Loops the seamless cycle to fill the exact 15.0-second target duration.
      3. Upscales to `target_w`x`target_h` with `flags=lanczos` + Contrast Adaptive Sharpening (`cas=0.45`).
    """
    cycle_mp4 = raw_mp4.with_suffix(".cycle.mp4")

    vf_cycle = (
        f"[0:v]fps={target_fps},split[fwd][tmp];"
        f"[tmp]reverse[rev];"
        f"[fwd][rev]concat=n=2:v=1:a=0,"
        f"scale={target_w}:{target_h}:flags=lanczos,"
        f"cas=0.45,format=yuv420p[outv]"
    )
    cmd_cycle = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(raw_mp4),
        "-filter_complex",
        vf_cycle,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "17",
        str(cycle_mp4),
    ]
    subprocess.run(cmd_cycle, check=True)

    cmd_final = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(cycle_mp4),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t",
        f"{target_duration:.2f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(target_fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_mp4),
    ]
    subprocess.run(cmd_final, check=True)
    cycle_mp4.unlink(missing_ok=True)
    raw_mp4.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-VM 9:16 15s influencer video bake-off worker.")
    parser.add_argument("--config-id", type=str, required=True, help="Unique configuration identifier.")
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=["ltx-2b-distilled", "wan-2.1-1.3b", "skyreels-v2-1.3b"],
    )
    parser.add_argument("--prompt", type=str, required=True, help="9:16 influencer video prompt.")
    parser.add_argument("--base-width", type=int, default=192, help="Base 9:16 latent width.")
    parser.add_argument("--base-height", type=int, default=320, help="Base 9:16 latent height.")
    parser.add_argument("--base-frames", type=int, default=9, help="Base frame count.")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps.")
    parser.add_argument("--target-width", type=int, required=True, help="Final output width (720 or 1080).")
    parser.add_argument("--target-height", type=int, required=True, help="Final output height (1280 or 1920).")
    parser.add_argument("--duration", type=float, default=15.0, help="Final output duration in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, required=True, help="Output MP4 filename.")
    args = parser.parse_args()

    cpu_threads = os.cpu_count() or 4
    torch.set_num_threads(cpu_threads)
    torch.backends.mkldnn.enabled = True
    print(f"[bakeoff-worker] [{args.config_id}] Using {cpu_threads} CPU threads with OneDNN FP32 AVX2.")

    output_mp4 = Path(args.output).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    raw_mp4 = output_mp4.with_suffix(".raw.mp4")

    if args.engine == "ltx-2b-distilled":
        run_ltx_engine(
            prompt=args.prompt,
            base_w=args.base_width,
            base_h=args.base_height,
            base_frames=args.base_frames,
            steps=args.steps,
            seed=args.seed,
            raw_mp4=raw_mp4,
        )
    elif args.engine == "wan-2.1-1.3b":
        run_wan_engine(
            prompt=args.prompt,
            base_w=args.base_width,
            base_h=args.base_height,
            base_frames=args.base_frames,
            steps=args.steps,
            seed=args.seed,
            raw_mp4=raw_mp4,
        )
    else:
        run_skyreels_engine(
            prompt=args.prompt,
            base_w=args.base_width,
            base_h=args.base_height,
            base_frames=args.base_frames,
            steps=args.steps,
            seed=args.seed,
            raw_mp4=raw_mp4,
        )

    print(
        f"[bakeoff-worker] [{args.config_id}] Mastering 9:16 output to "
        f"{args.target_width}x{args.target_height} ({args.duration:.1f}s @ 24fps)..."
    )
    master_to_9x16_15s(
        raw_mp4=raw_mp4,
        output_mp4=output_mp4,
        target_w=args.target_width,
        target_h=args.target_height,
        target_duration=args.duration,
        target_fps=24,
    )
    print(
        f"[bakeoff-worker] [{args.config_id}] Complete -> {output_mp4.name} "
        f"({output_mp4.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
