# Antigravity cloud render

Distributed FFmpeg video rendering and AI video generation cluster powered by GitHub Actions matrix runners.

1. `cloud-render` splits source videos losslessly at keyframes, uploads chunks to a temporary GitHub release, encodes them in parallel across 10 to 20 4-vCPU `ubuntu-latest` runners using static FFmpeg 8.0 GPL builds, stitches the rendered segments losslessly, downloads the output, and deletes the temporary release tag.
2. `cloud-ltx` dispatches one or more text-to-video prompts to parallel `ubuntu-latest` VMs running Lightricks `LTX-Video-2B-Distilled` via a two-stage memory-optimized `bfloat16` pipeline (`city96/t5-v1_1-xxl-encoder-bf16` followed by `multimodalart/ltxv-2b-0.9.6-distilled` and `Lightricks/LTX-Video-0.9.5` VAE), stitches generated scenes in order, downloads `final_ltx.mp4`, and cleans up the release tag.

## Project structure

- `.github/workflows/parallel-render.yml`, GitHub Actions workflow with `render-chunk` matrix job and `stitch` job.
- `.github/workflows/ltx-generate.yml`, GitHub Actions workflow for parallel `LTX-Video-2B-Distilled` text-to-video generation and scene stitching.
- `cloud_render.py`, main CLI entry point invoked by the `cloud-render` global command.
- `ltx_cloud.py`, main CLI entry point invoked by the `cloud-ltx` global command.
- `src/ltx_worker.py`, runner worker implementing two-stage memory-optimized `LTX-Video-2B-Distilled` generation inside the 16 GB GitHub Actions VM.
- `src/ltx_cluster.py`, GitHub release and workflow orchestration for `ltx-generate.yml`.
- `src/bootstrap.py`, auto-checks Python dependencies and verifies `ffmpeg`, `ffprobe`, and `gh` on PATH.
- `src/config_loader.py`, loads and persists default settings in `config.json`.
- `src/splitter.py`, probes source media with `ffprobe` and splits losslessly at keyframes via `ffmpeg -f segment -c copy -reset_timestamps 1`.
- `src/github_cluster.py`, handles temporary release creation, chunk upload, `workflow_dispatch`, `gh run watch`, result download, and release cleanup.
- `src/verifier.py`, verifies the stitched output streams, codecs, and duration against the source file using `ffprobe`.
- `config.json`, persistent configuration file.
- `samples/`, input test media directory.
- `output/`, dedicated output media directory kept separate from source files.

## Setup

Install dependencies in one command if not relying on auto-bootstrap:

```powershell
pip install -r requirements.txt
```

Global CLI shims are installed in `C:\Users\Yeamin-Sheikh\.local\bin\cloud-render.ps1`, `cloud-render.cmd`, `cloud-ltx.ps1`, and `cloud-ltx.cmd`.

## Usage

### Parallel FFmpeg rendering

```powershell
cloud-render <input_video> <output_video> [--chunks N] [--args "..."]
```

Example with 4 parallel runners and custom x264 flags:

```powershell
cloud-render samples/test_1080p_12s.mp4 output/rendered_1080p_12s.mp4 --chunks 4 --args "-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k"
```

### Parallel LTX-Video-2B-Distilled AI video generation

Single-scene generation:

```powershell
cloud-ltx "A cinematic close-up of golden sunlight filtering through mountain pine trees, natural motion" output/ltx_sample.mp4 --width 512 --height 320 --frames 25 --steps 8
```

Multi-scene parallel generation across multiple runners:

```powershell
cloud-ltx --prompt "Aerial shot of a coastal highway at sunset" --prompt "Close-up of ocean waves crashing against dark rocks" output/ltx_two_scenes.mp4 --width 512 --height 320 --frames 25
```

### Multi-VM 9:16 15-second influencer bake-off

Launch 6 parallel VMs comparing `LTX-Video-2B-Distilled`, `Wan 2.1 1.3B`, and `SkyReels-V2-DF-1.3B` in both 720p (`720x1280`) and 1080p (`1080x1920`):

```powershell
cloud-bakeoff --duration 15.0 --out-dir output/bakeoff
```
