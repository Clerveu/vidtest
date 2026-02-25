import os
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).parent / "Source"
OUTPUT = Path(__file__).parent / "Outputs"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


def list_videos():
    videos = sorted(
        f for f in SOURCE.iterdir() if f.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print("No video files found in Source/")
        sys.exit(1)
    return videos


def get_video_info(path):
    """Get duration and resolution via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    for s in data["streams"]:
        if s["codec_type"] == "video":
            w, h = int(s["width"]), int(s["height"])
            return duration, w, h
    return duration, None, None


def prompt_choice(prompt_text, options):
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            idx = int(input(prompt_text))
            if 1 <= idx <= len(options):
                return idx - 1
        except (ValueError, EOFError):
            pass
        print(f"  Enter 1-{len(options)}")


def prompt_number(prompt_text, default=None, min_val=None, max_val=None, is_int=False):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt_text}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            val = int(raw) if is_int else float(raw)
            if min_val is not None and val < min_val:
                print(f"  Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Must be <= {max_val}")
                continue
            return val
        except (ValueError, EOFError):
            print("  Enter a number")


def main():
    OUTPUT.mkdir(exist_ok=True)

    videos = list_videos()
    print("\n=== Videos in Source/ ===")
    idx = prompt_choice(
        "Pick a video: ",
        [f"{v.name}" for v in videos]
    )
    video = videos[idx]

    duration, orig_w, orig_h = get_video_info(video)
    print(f"\n  File: {video.name}")
    print(f"  Duration: {duration:.1f}s  |  Resolution: {orig_w}x{orig_h}")

    fps = prompt_number("\nFPS", default=2.0, min_val=0.1, max_val=60.0)
    width = prompt_number("Output width (height auto-scaled)", default=orig_w, min_val=64, is_int=True)
    clip_len = prompt_number("Clip length in seconds", default=min(duration, 14.0), min_val=0.1, max_val=duration)

    # Calculate number of clips
    num_clips = int(duration // clip_len)
    remainder = duration - num_clips * clip_len
    if remainder > 0.5:
        num_clips += 1

    print(f"\n  Will produce {num_clips} clip(s) of ~{clip_len}s each")
    print(f"  Output: {fps}fps, {width}px wide")

    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("Cancelled.")
        return

    stem = video.stem
    scale_filter = f"scale={width}:-2"

    for i in range(num_clips):
        start = i * clip_len
        remaining = duration - start
        actual_len = min(clip_len, remaining)
        if actual_len < 1.0 / fps:
            print(f"  [{i+1}/{num_clips}] Skipping — remainder too short ({remaining:.1f}s)")
            continue
        out_name = f"{stem}_{width}p_{fps}fps_clip{i+1:03d}.mp4"
        out_path = OUTPUT / out_name

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video),
            "-t", str(actual_len),
            "-vf", f"scale={width}:-2,fps={fps}",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-an",
            str(out_path)
        ]

        print(f"  [{i+1}/{num_clips}] {out_name} (start={start:.1f}s) ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            size_kb = out_path.stat().st_size / 1024
            print(f"OK ({size_kb:.0f} KB)")
        else:
            print("FAILED")
            print(result.stderr[-500:] if result.stderr else "")

    print(f"\nDone. Clips saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
