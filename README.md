# Video-to-Text Analyzer (Speaker Diarization)

An all-in-one pipeline that turns a video file into a structured, per-scene text
document combining **speaker-attributed dialogue** with **visual descriptions**.
Dialogue comes from local WhisperX + pyannote speaker diarization; visual
descriptions come from OpenRouter's Qwen3.5-397B-A17B vision model.

Everything runs from a single script: **`analyze_speak.py`**.

## What It Does

`analyze_speak.py` runs a 7-step pipeline (all steps support **resume** —
re-running skips work that's already complete):

1. **Select video** from `Source/` — uses [guessit](https://github.com/guessit-io/guessit)
   to turn the filename into a clean prefix (e.g. `BreakingBad-s02e12`).
2. **Speaker diarization** — extracts the English audio track to 16 kHz mono WAV,
   then runs WhisperX (`large-v3`) + pyannote diarization in an isolated venv and
   splits the transcript into 30-second subtitle chunks.
3. **Split video** — 30-second chunks at 720p / 2 fps using NVENC hardware
   encoding (`-hwaccel cuda`, `h264_nvenc`, `-preset p7`).
4. **Describe each chunk** — sends each clip plus its matching subtitles to Qwen
   via OpenRouter and gets back a ~200-word visual description.
5. **Combine** — merges subtitles + descriptions into per-chunk
   `[DIALOGUE]` + `[VISUAL DESCRIPTION]` sections.
6. **Post-process with Claude Code** *(optional)* — launches the `claude` CLI in a
   new console against `testoutputs/`, which holds a `CLAUDE.md` with
   deduplication instructions; Claude compresses redundant setting/costume
   re-descriptions across consecutive same-scene chunks.
7. **Results** — individual chunk files plus one combined `*_analysis.txt`.

## Requirements

- **Python 3.10+**
- **NVIDIA GPU + CUDA** — required for both diarization (`device="cuda"`) and
  NVENC video encoding.
- **ffmpeg / ffprobe** with NVENC support, on your `PATH`.
- **OpenRouter API key** — `OPENROUTER_API_KEY` env var, or enter it on first run
  (saved to `.openrouter_key`, which is gitignored).
- **Hugging Face token** — `HF_TOKEN` env var, required by pyannote's gated
  `speaker-diarization-community-1` model. Accept the model terms on Hugging Face
  first.
- **Claude Code** *(optional)* — for step 6. If `claude` isn't on your `PATH`, the
  pipeline skips post-processing and raw results remain in `results/`.

## Setup

### 1. Main environment

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

### 2. Diarization environment (isolated)

WhisperX / pyannote live in their own venv at `subtest/venv` to avoid dependency
conflicts with the main environment. `analyze_speak.py` invokes
`subtest/diarize_worker.py` through it automatically.

```bash
python -m venv subtest/venv
subtest/venv/Scripts/python -m pip install -r subtest/requirements.txt
```

For a CUDA build of torch, install the matching wheel from
[pytorch.org](https://pytorch.org) first if the default is CPU-only.

### 3. Run

```bash
# set your keys (PowerShell)
$env:OPENROUTER_API_KEY = "..."
$env:HF_TOKEN = "..."

python analyze_speak.py
```

Put video files in `Source/` (created on first run), then follow the prompts to
select one and optionally provide movie context (cast, year, etc.).

Output directories are created automatically and are all gitignored:

| Folder          | Contents                                            |
|-----------------|-----------------------------------------------------|
| `Source/`       | Input videos (you add these)                        |
| `audio/`        | Extracted full audio tracks (.wav)                  |
| `Outputs/`      | Video chunks (.mp4)                                 |
| `subtitles/`    | Diarized subtitle chunks (.txt)                     |
| `descriptions/` | Per-chunk Qwen descriptions (.txt)                  |
| `results/`      | Combined analysis + per-chunk files (.txt)          |
| `testoutputs/`  | Claude-post-processed chunks + `CLAUDE.md`          |

## Configuration

All settings live at the top of `analyze_speak.py`:

| Variable           | Default                    | Description                          |
|--------------------|----------------------------|--------------------------------------|
| `CHUNK_DURATION`   | `30`                       | Seconds per chunk                    |
| `VIDEO_QUALITY`    | `720`                      | Max output width (px)                |
| `VIDEO_FPS`        | `2`                        | Output framerate                     |
| `NVENC_PRESET`     | `p7`                       | NVENC preset (p1=fastest, p7=best)   |
| `OPENROUTER_MODEL` | `qwen/qwen3.5-397b-a17b`   | OpenRouter model id                  |

## Layout

```
analyze_speak.py          # the pipeline
subtest/diarize_worker.py # WhisperX + pyannote worker (runs in subtest/venv)
testoutputs/CLAUDE.md     # instructions for the optional Claude post-process step
requirements.txt          # main deps
subtest/requirements.txt  # diarization deps
```

## Forking / Running Locally

Built for a specific home setup. If you're forking:

- **No NVIDIA GPU?** Swap the ffmpeg command in `split_video()` to CPU encoding
  (e.g. `-c:v libx264`) and set `device` in `subtest/diarize_worker.py` to `"cpu"`
  (much slower).
- **`testoutputs/CLAUDE.md`** contains two `<REPO_ROOT>/...` placeholder paths —
  replace `<REPO_ROOT>` with your clone's absolute path for the post-process step.

## Cost

At current OpenRouter pricing for Qwen3.5-397B-A17B ($0.55/M input, $3.50/M
output), a 30-second chunk costs roughly $0.005 — a full ~84-chunk movie runs
about $0.40–0.50.
</content>
