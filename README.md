# Antigravity cloud render

Distributed FFmpeg video rendering cluster powered by GitHub Actions matrix runners. Splits source videos losslessly at keyframes, uploads chunks to a temporary GitHub release, encodes them in parallel across 10 to 20 4-vCPU `ubuntu-latest` runners using static FFmpeg 8.0 GPL builds, stitches the rendered segments losslessly, downloads the output, and deletes the temporary release tag.

## Project structure

- `.github/workflows/parallel-render.yml`, GitHub Actions workflow with `render-chunk` matrix job and `stitch` job.
- `cloud_render.py`, main CLI entry point invoked by the `cloud-render` global command.
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

Global CLI shims are installed in `C:\Users\Yeamin-Sheikh\.local\bin\cloud-render.cmd` and `cloud-render.ps1`.

## Usage

```powershell
cloud-render <input_video> <output_video> [--chunks N] [--args "..."]
```

Example with 4 parallel runners and custom x264 flags:

```powershell
cloud-render samples/test_1080p_12s.mp4 output/rendered_1080p_12s.mp4 --chunks 4 --args "-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k"
```
