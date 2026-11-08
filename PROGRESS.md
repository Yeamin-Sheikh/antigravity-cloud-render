# Project progress

> Auto-maintained by dev-tracker skill. Do not edit the log section manually.

## Project info

- Project: antigravity-cloud-render
- Started: 2026-11-07
- Last updated: 2026-11-08
- Status: Complete

---

## Progress log

### 2026-11-08, LTX-Video-2B-Distilled cloud generator setup and live test

Duration: ~55m
Status: Done

#### What changed
- Added `.github/workflows/ltx-generate.yml` for parallel text-to-video generation on public `ubuntu-latest` VMs using `LTX-Video-2B-Distilled`.
- Created `src/ltx_worker.py` with two-stage memory offloading: encodes prompt with `city96/t5-v1_1-xxl-encoder-bf16`, frees RAM and disk cache, then runs `multimodalart/ltxv-2b-0.9.6-distilled` and `Lightricks/LTX-Video-0.9.5` VAE in `bfloat16` across 8 distilled steps.
- Built `ltx_cloud.py` and `src/ltx_cluster.py`, and installed global `cloud-ltx` shims in `C:\Users\Yeamin-Sheikh\.local\bin`.
- Ran live end-to-end test via `cloud-ltx` on GitHub Actions run `42496442843`, generating `output/ltx_test_clip.mp4` (`384x256`, `9` frames at `12` fps, `0.75s`, `h264` + `aac`, `43.6 KB`) and cleaning up temporary release `ltx-20261108-154541-f919ac`.

#### Files touched
- `.github/workflows/ltx-generate.yml`, GitHub Actions workflow for single or multi-scene `LTX-Video-2B-Distilled` generation.
- `src/ltx_worker.py`, two-stage memory-optimized Diffusers worker script.
- `src/ltx_cluster.py`, release and workflow orchestration for `ltx-generate.yml`.
- `ltx_cloud.py`, CLI entry point for `cloud-ltx`.
- `src/config_loader.py`, added `ltx_workflow` default key.
- `config.json`, added `ltx_workflow`.
- `README.md`, added `cloud-ltx` usage examples.

#### Issues found
- `Lightricks/LTX-Video` stores its default T5-XXL text encoder in FP32 at 18.1 GB, which exceeds the default 14 GB free disk space on a GitHub Actions runner. Solved by using `city96/t5-v1_1-xxl-encoder-bf16` at 9.08 GB in Stage 1 and purging its cache before Stage 2 downloads the 3.67 GB distilled 2B transformer.
- Pure 4-vCPU inference on `ubuntu-latest` succeeded without OOM using two-stage loading, taking ~52 minutes for a 9-frame 384x256 clip on CPU.

#### Next steps
- Both `cloud-render` and `cloud-ltx` are tested and ready for production use.

---

### 2026-11-07, Initial cluster setup and live 4-chunk render verification

Duration: ~15m
Status: Done

#### What changed
- Created GitHub repository `Yeamin-Sheikh/antigravity-cloud-render` via `gh repo create`.
- Added `.github/workflows/parallel-render.yml` with `render-chunk` matrix job and `stitch` concat job using static FFmpeg 8.0 GPL builds.
- Built modular Python CLI (`cloud_render.py`, `src/bootstrap.py`, `src/config_loader.py`, `src/splitter.py`, `src/github_cluster.py`, `src/verifier.py`).
- Installed `cloud-render` shims onto user PATH (`C:\Users\Yeamin-Sheikh\.local\bin`).
- Generated 12-second 1080p 30fps test video `samples/test_1080p_12s.mp4` and executed a live 4-chunk parallel render on GitHub Actions (run `42285582830`), producing `output/rendered_1080p_12s.mp4` with `0.000s` duration delta (`12.000s`).

#### Files touched
- `.github/workflows/parallel-render.yml`, workflow definition for parallel chunk encoding and lossless stitching.
- `cloud_render.py`, main CLI orchestrator and `--args` hyphen normalization.
- `src/splitter.py`, lossless keyframe segment splitter using `ffmpeg -f segment -c copy -reset_timestamps 1`.
- `src/github_cluster.py`, GitHub release and workflow orchestration via `gh`.
- `src/verifier.py`, post-render stream and duration verification via `ffprobe`.
- `config.json`, persistent default settings.
- `README.md`, documentation and usage guide.

#### Issues found
- FFmpeg segment muxer can overshoot exact keyframe intervals when audio padding slightly inflates container duration. Subtracted a 20ms epsilon from `segment_time` in `src/splitter.py` so exact keyframe boundaries trigger reliably.
- Python `argparse` treats `--args "-c:v ..."` as the `-c` short option unless normalized to `--args=...`. Added `normalize_cli_argv` in `cloud_render.py`.

#### Next steps
- Ready for production video rendering workloads up to 20 parallel GitHub Actions runners.

---
