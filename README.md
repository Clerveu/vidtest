# Video-to-Text Analyzer

All-in-one pipeline that converts video files into structured text descriptions using OpenRouter's Qwen3.5-397B-A17B vision model. Includes an optional VLC remote tool for real-time interaction with Claude Code while watching.

## What It Does

**analyze.py** runs a 6-step pipeline:

1. **Select video** from `Source/` — uses guessit to parse filenames into clean prefixes (e.g. `AKnightOfTheSevenKingdoms-s01e01`)
2. **Extract/download subtitles** — embedded track extraction (prioritizes English SDH) > external SRT > auto-download via subliminal
3. **Split video** into 30-second chunks at 720p, 2fps using NVENC hardware encoding (`-hwaccel cuda`, `-preset p7`, `-rc vbr`)
4. **Process each chunk** through Qwen 397B via OpenRouter — sends video + matching subtitles, gets back ~200 word visual descriptions
5. **Combine outputs** — merges subtitle chunks + descriptions into `[DIALOGUE]` + `[VISUAL DESCRIPTION]` sections
6. **Generate results** — individual chunk files + one combined analysis file
7. **Condenses descriptions** - runs all chunks through Claude Code to remove redundancies in visual descriptions to considerably reduce final token count.

**vlc_remote.py** bridges VLC and Claude Code for real-time "movie watching":

- **Shift+Enter** — injects VLC's current timestamp into the active window and sends
- **Ctrl+Shift+Enter** — takes a VLC screenshot + timestamp, pastes both, and sends
- **Ctrl+Backslash** — screenshot + paste without sending (compose a message around it)

Together: analyze.py builds the full text "memory" of a movie, vlc_remote.py lets you poke Claude at any moment during playback with a timestamp/screenshot, and Claude can answer with full context of what's happening.

## Requirements

- **Python 3.10+**
- **ffmpeg** with NVENC support (NVIDIA GPU required for hardware encoding)
- **OpenRouter API key** — set `OPENROUTER_API_KEY` env var or enter it on first run (saved to `.openrouter_key`)

### Python Dependencies

```bash
# Required
pip install requests

# Optional (enables auto subtitle download + smart prefix generation)
pip install subliminal guessit babelfish

# For vlc_remote.py only (Windows)
pip install pywin32 Pillow
```

## Setup

### analyze.py

1. Create a `Source/` folder in the project directory and place video files in it
2. Run `python analyze.py`
3. Follow the prompts to select a video and optionally provide movie context (cast, year, etc.)

Output directories are created automatically:
- `Outputs/` — video chunks (.mp4)
- `subtitles/` — subtitle chunk files (.txt)
- `descriptions/` — per-chunk Qwen descriptions (.txt)
- `results/` — combined analysis + individual chunk files (.txt)

All steps support **resume** — re-running skips already completed work.

### vlc_remote.py

Requires VLC running with its HTTP interface enabled on the target machine.

**Things to change:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLC_URL` | `http://192.168.1.150:8080/requests/status.json` | VLC HTTP API endpoint — change the IP to your VLC machine |
| `VLC_PASS` | `password` | VLC HTTP interface password (set in VLC > Preferences > Main interfaces > Lua > HTTP password) |
| `SNAPSHOT_DIR` | `\\HTPC\vlcpic` | Network path where VLC saves snapshots (set in VLC > Preferences > Video > Snapshots directory) |

### analyze.py Configuration

All configuration is at the top of the file:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOURCE_DIR` | `Source/` | Where to look for input video files |
| `OUTPUTS_DIR` | `Outputs/` | Where video chunks are saved |
| `CHUNK_DURATION` | `30` | Seconds per chunk |
| `VIDEO_QUALITY` | `720` | Max output width in pixels |
| `VIDEO_FPS` | `2` | Output framerate |
| `NVENC_PRESET` | `p7` | NVENC quality preset (p1=fastest, p7=best quality) |
| `OPENROUTER_MODEL` | `qwen/qwen3.5-397b-a17b` | OpenRouter model identifier |

## Other Scripts

- **slice.py** — standalone video slicer (superseded by analyze.py's built-in splitting)
- **test_clip.py** — single-clip tester for debugging OpenRouter API calls

## Cost

At current OpenRouter pricing for Qwen3.5-397B-A17B ($0.55/M input, $3.50/M output), a typical 30-second chunk costs ~$0.005. A full movie (~84 chunks) runs roughly $0.40-0.50.
