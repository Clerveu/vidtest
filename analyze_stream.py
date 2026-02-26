"""
VIDEO ANALYZER - All-in-One Pipeline (OpenRouter Qwen3.5-397B-A17B)

Complete pipeline in one script:
1. Select video from Source/ with guessit-generated clean prefix
2. Extract/download subtitles and split into 30s chunk files
3. Split video into 30s chunks at 720p, 2fps, NVENC p7
4. Process each chunk through OpenRouter Qwen
5. Combine subtitle chunks + descriptions into [DIALOGUE] + [VISUAL DESCRIPTION]
6. Generate combined analysis file
"""

import subprocess
import json
import os
import re
import sys
import time
import base64
import requests
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue, Empty

try:
    from babelfish import Language
    import subliminal
    import guessit
    HAS_SUBTITLE_DEPS = True
except ImportError:
    HAS_SUBTITLE_DEPS = False

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "Source"
OUTPUTS_DIR = BASE_DIR / "Outputs"           # video chunks
SUBTITLES_DIR = BASE_DIR / "subtitles"
DESCRIPTIONS_DIR = BASE_DIR / "descriptions"
RESULTS_DIR = BASE_DIR / "results"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
POSTPROCESS_DIR = BASE_DIR / "testoutputs"      # Claude post-processor workdir + CLAUDE.md

# Video processing
CHUNK_DURATION = 30   # seconds
VIDEO_QUALITY = 720   # 720p max width
VIDEO_FPS = 2         # frames per second for output chunks
NVENC_PRESET = "p7"

# OpenRouter
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3.5-397b-a17b"

# Qwen sampling FPS (tells the model how to interpret the video)
SAMPLING_FPS = 2.0

SYSTEM_PROMPT = """You are narrating a portion of a film. Your goal is to very briefly describe the visuals happening on screen. The video you are watching is a small part of a whole. You will recieve subtitles for the current video, use them as a master reference for context.

**SYSTEM INSTRUCTIONS FOR VISUAL-ONLY DESCRIPTIONS**

1. **Literal Visuals Only**: Describe only elements clearly visible on screen
2. **Zero Interpretation**: No speculation about emotions, motivations, intentions, backstory, or symbolic meaning
3. **Neutral Language**: Use objective descriptors. "A person's eyes water" not "the person appears sad"
4. **Action Without Motivation**: "A man walks toward the door" not "rushes anxiously toward the exit"
5. **Camera Work When Notable**: Mention obvious techniques (zooms, pans, tracking shots, split-diopter, dolly zooms, focus pulls) only when prominent
6. **No Contextual Assumptions**: Do not infer genre, story arc, historical setting, or character relationships
7. **Direct Language**: Simple, factual reporting. No poetic or dramatic phrasing
8. **Objective Stance**: Never use "we see," "it seems," "this suggests," "perhaps," "maybe," "likely," "possibly," "suggesting," "there may be," "resembles," etc
9. **Report Only What's Present**: Do not mention absences or omissions
10. **Never Reference These Rules**: Do not mention these instructions in your output
11. **Don't assume speaker**: Without concrete evidence do not correlate any given subtitle with any given character
12. **Don't reproduce subtitles**: Only use subtitles to give yourself context to better describe visuals
13. **Rely on subtitles for visual context**
14. **Size responses appropriately to amount of action**

Thank you so much Qwen!
"""

USER_PROMPT = """Briefly describe this video in under 200 words. Included are the subtitles for this video - rely on them for context in understanding the current video."""

# ============================================================================
# API KEY
# ============================================================================

def get_api_key():
    """Get OpenRouter API key from env or prompt."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    key_file = BASE_DIR / ".openrouter_key"
    if key_file.exists():
        return key_file.read_text().strip()

    print("No OpenRouter API key found.")
    print("Set OPENROUTER_API_KEY env var, or enter it now to save locally.")
    key = input("API key: ").strip()
    if not key:
        print("No key provided. Exiting.")
        sys.exit(1)

    key_file.write_text(key)
    print(f"  Saved to {key_file}")
    return key

# ============================================================================
# UTILITIES
# ============================================================================

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_step(step_num, total_steps, description):
    print()
    print(f"[STEP {step_num}/{total_steps}] {description}")
    print("-" * 60)

# ============================================================================
# STEP 1: VIDEO SELECTION
# ============================================================================

def generate_clean_prefix(filename):
    """
    Generate a clean prefix from video filename using guessit.
    Movies: "Synecdoche, New York (2008) ..." -> "SynecdocheNewYork"
    TV Shows: "Breaking Bad S02E12 ..." -> "BreakingBad-s02e12"
    """
    if not HAS_SUBTITLE_DEPS:
        # Fallback: strip extension, replace spaces/dots with nothing
        stem = Path(filename).stem
        clean = re.sub(r'[^a-zA-Z0-9]', '', stem)
        return clean if clean else "video"

    guess = guessit.guessit(filename)

    title = guess.get('title', 'video')
    clean_title = ''.join(word.capitalize() for word in title.replace('-', ' ').split())

    if guess.get('type') == 'episode':
        season = guess.get('season', 0)
        episode = guess.get('episode', 0)
        if isinstance(episode, list):
            episode = episode[0]
        return f"{clean_title}-s{season:02d}e{episode:02d}"
    else:
        return clean_title

def select_video():
    """Let user select a video from Source/ and generate clean prefix."""
    if not SOURCE_DIR.exists():
        SOURCE_DIR.mkdir(exist_ok=True)

    videos = sorted(
        f for f in SOURCE_DIR.iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    )

    if not videos:
        print(f"No videos found in {SOURCE_DIR}/")
        print("  Please add video files to the Source folder")
        return None, None

    print(f"Found {len(videos)} video(s) in Source/:\n")
    for i, video in enumerate(videos):
        size_mb = video.stat().st_size / (1024 * 1024)
        clean_prefix = generate_clean_prefix(video.name)
        print(f"  [{i}] {video.name}")
        print(f"       -> Prefix: {clean_prefix} ({size_mb:.1f} MB)")
        print()

    while True:
        try:
            choice = int(input(f"Select video (0-{len(videos)-1}): "))
            if 0 <= choice < len(videos):
                selected = videos[choice]
                prefix = generate_clean_prefix(selected.name)
                return selected, prefix
            print(f"Please enter a number between 0 and {len(videos)-1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None, None

# ============================================================================
# STEP 2: SUBTITLE EXTRACTION / DOWNLOAD
# ============================================================================

def download_subtitles(video_path, prefix, timeout_stage1=10, timeout_stage2=45):
    """
    Attempt to download subtitles using subliminal with timeout and intelligent fallback.
    Stage 1: Search by exact filename (10s)
    Stage 2: Parse movie name and search by title/year (45s)
    """
    if not HAS_SUBTITLE_DEPS:
        print("  subliminal/guessit/babelfish not installed. Skipping auto-download.")
        print("  Install with: pip install subliminal guessit babelfish")
        return False

    print("  Attempting to download subtitles automatically...")

    def try_exact_filename(queue):
        try:
            video = subliminal.Video.fromname(str(video_path))
            subtitles = subliminal.download_best_subtitles(
                {video},
                languages={Language('eng')},
                min_score=0
            )
            if video in subtitles and len(subtitles[video]) > 0:
                queue.put(('success', video, subtitles, 'exact filename'))
            else:
                queue.put(('no_results', None, None))
        except Exception as e:
            queue.put(('error', str(e), None))

    def try_title_search(queue, guess, unused):
        try:
            video = subliminal.Video.fromguess(str(video_path), guess)
            subtitle_list = subliminal.list_subtitles({video}, {Language('eng')})

            if video in subtitle_list and len(subtitle_list[video]) > 0:
                available = subtitle_list[video]
                movie_title = video.title.lower() if video.title else ""
                movie_year = video.year

                filtered = []
                for sub in available:
                    sub_title = getattr(sub, 'movie_name', '').lower()
                    sub_year = getattr(sub, 'movie_year', None)
                    title_match = movie_title in sub_title or sub_title in movie_title
                    year_match = (sub_year == movie_year) if sub_year else True
                    if title_match and year_match:
                        filtered.append(sub)

                if not filtered:
                    queue.put(('no_results', None, None))
                    return

                best_sub = sorted(filtered, key=lambda s: (s.hearing_impaired, s.provider_name))[0]
                subliminal.download_subtitles([best_sub])
                subliminal.save_subtitles(video, [best_sub])
                subtitles = {video: [best_sub]}
                queue.put(('success', video, subtitles, 'title search'))
            else:
                queue.put(('no_results', None, None))
        except Exception as e:
            queue.put(('error', str(e), None))

    def save_subtitle(video, subtitles, search_method):
        best_subtitle = subtitles[video][0]
        output_path = SUBTITLES_DIR / f"{prefix}.srt"

        subliminal.save_subtitles(video, subtitles[video])

        downloaded_path = video_path.with_suffix('.srt')
        if downloaded_path.exists():
            downloaded_path.rename(output_path)
            print(f"  Downloaded subtitles from {best_subtitle.provider_name} (via {search_method})")
            print(f"  Saved to: {output_path.name}")
            return True

        alt_path = video_path.with_suffix('.en.srt')
        if alt_path.exists():
            alt_path.rename(output_path)
            print(f"  Downloaded subtitles from {best_subtitle.provider_name} (via {search_method})")
            print(f"  Saved to: {output_path.name}")
            return True

        return False

    # STAGE 1: Try exact filename match
    print(f"  Stage 1: Searching by exact filename...")

    queue1 = Queue()
    thread1 = Thread(target=try_exact_filename, args=(queue1,))
    thread1.daemon = True
    thread1.start()

    try:
        status, video, subtitles, *extra = queue1.get(timeout=timeout_stage1)
        if status == 'success':
            search_method = extra[0] if extra else 'exact filename'
            if save_subtitle(video, subtitles, search_method):
                return True
    except Empty:
        print(f"  Stage 1 timed out after {timeout_stage1} seconds")

    # STAGE 2: Parse movie name and try title search
    print(f"  Stage 2: Parsing movie name for broader search...")

    try:
        guess = guessit.guessit(video_path.name)
        title = guess.get('title', '')
        year = guess.get('year', None)

        if not title:
            print("  Could not parse movie title from filename")
            return False

        print(f"  Detected: {title}" + (f" ({year})" if year else ""))

        queue2 = Queue()
        thread2 = Thread(target=try_title_search, args=(queue2, guess, None))
        thread2.daemon = True
        thread2.start()

        status, video, subtitles, *extra = queue2.get(timeout=timeout_stage2)
        if status == 'success':
            search_method = extra[0] if extra else 'title search'
            if save_subtitle(video, subtitles, search_method):
                return True

        print("  No matching subtitles found")
        return False

    except Empty:
        print(f"  Stage 2 timed out after {timeout_stage2} seconds")
        return False
    except Exception as e:
        print(f"  Stage 2 failed: {e}")
        return False

def get_subtitle_tracks(video_path):
    """Get subtitle tracks from video with detailed metadata for smart selection."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', 's',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)
        data = json.loads(result.stdout)

        TEXT_CODECS = ['subrip', 'ass', 'ssa', 'webvtt', 'mov_text', 'srt', 'text']

        tracks = []
        for stream in data.get('streams', []):
            codec = stream.get('codec_name', 'unknown')
            disposition = stream.get('disposition', {})
            tags = stream.get('tags', {})

            if codec in TEXT_CODECS:
                title = tags.get('title', '').lower()

                is_sdh = (
                    disposition.get('hearing_impaired', 0) == 1 or
                    'sdh' in title or
                    'hearing' in title
                )
                is_forced = (
                    disposition.get('forced', 0) == 1 or
                    'forced' in title
                )

                tracks.append({
                    'index': stream['index'],
                    'codec': codec,
                    'language': tags.get('language', 'unknown'),
                    'title': tags.get('title', ''),
                    'forced': is_forced,
                    'sdh': is_sdh,
                    'default': disposition.get('default', 0) == 1
                })

        return tracks
    except:
        return []

def parse_srt(srt_path):
    """Parse SRT file into subtitle entries."""
    try:
        with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        subtitles = []
        blocks = re.split(r'\n\n+', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                timestamp_line = lines[1]
                text_lines = lines[2:]

                match = re.match(r'([\d:,]+)\s*-->\s*([\d:,]+)', timestamp_line)
                if match:
                    start_time = parse_srt_timestamp(match.group(1))
                    text = ' '.join(text_lines)
                    subtitles.append({'start': start_time, 'text': text})

        return subtitles
    except:
        return []

def parse_srt_timestamp(timestamp):
    """Convert SRT timestamp to seconds."""
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', timestamp)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    return 0

def split_subtitles(subtitles, chunk_duration):
    """Split subtitles into time-based chunks."""
    if not subtitles:
        return []

    chunks = []
    current_chunk = []
    chunk_start = 0
    chunk_end = chunk_duration

    for sub in subtitles:
        while sub['start'] >= chunk_end:
            chunks.append({
                'start': chunk_start,
                'end': chunk_end,
                'subtitles': current_chunk
            })
            chunk_start = chunk_end
            chunk_end += chunk_duration
            current_chunk = []

        current_chunk.append(sub)

    if current_chunk:
        chunks.append({
            'start': chunk_start,
            'end': chunk_end,
            'subtitles': current_chunk
        })

    return chunks

def save_subtitle_chunks(subtitles, prefix):
    """Parse, split, and save subtitle chunks to SUBTITLES_DIR."""
    if not subtitles:
        print("  SRT file is empty or invalid")
        return False

    print(f"  Found {len(subtitles)} subtitle entries")

    chunks = split_subtitles(subtitles, CHUNK_DURATION)
    print(f"  Created {len(chunks)} chunks")

    SUBTITLES_DIR.mkdir(exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        start_ts = format_time(chunk['start'])
        end_ts = format_time(chunk['end'])

        filename = f"{prefix}.chunk{chunk_num:04d}.{start_ts.replace(':', '-')}_to_{end_ts.replace(':', '-')}.txt"
        filepath = SUBTITLES_DIR / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for sub in chunk['subtitles']:
                timestamp = format_time(sub['start'])
                f.write(f"[{timestamp}] {sub['text']}\n")

    print(f"  Saved {len(chunks)} subtitle chunk files")
    return True

def extract_subtitles(video_path, prefix):
    """
    Extract subtitles from video. Priority:
    1. Embedded text-based subtitle tracks
    2. External SRT file in subtitles/
    3. Auto-download via subliminal
    """
    print(f"Extracting subtitles from {video_path.name}...")

    # Try embedded tracks first
    tracks = get_subtitle_tracks(video_path)

    if tracks:
        # Show available tracks
        print(f"  Found {len(tracks)} subtitle track(s):")
        for t in tracks:
            flags = []
            if t['sdh']:
                flags.append('SDH')
            if t['forced']:
                flags.append('forced')
            if t['default']:
                flags.append('default')
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            title_str = f" - {t['title']}" if t['title'] else ""
            print(f"    #{t['index']}: {t['language']} ({t['codec']}){flag_str}{title_str}")

        # Smart track selection
        selected_track = None

        # Priority 1: English SDH
        for track in tracks:
            if track['language'] == 'eng' and track['sdh'] and not track['forced']:
                selected_track = track
                print(f"  -> Selected: English SDH track (most complete)")
                break

        # Priority 2: English non-forced
        if not selected_track:
            for track in tracks:
                if track['language'] == 'eng' and not track['forced']:
                    selected_track = track
                    print(f"  -> Selected: English non-forced track")
                    break

        # Priority 3: Any English
        if not selected_track:
            for track in tracks:
                if track['language'] == 'eng':
                    selected_track = track
                    print(f"  -> Selected: English track (forced - may be incomplete!)")
                    break

        # Priority 4: First non-forced
        if not selected_track:
            for track in tracks:
                if not track['forced']:
                    selected_track = track
                    print(f"  -> Selected: {track['language']} non-forced track [fallback]")
                    break

        # Priority 5: First track
        if not selected_track:
            selected_track = tracks[0]
            print(f"  -> Selected: first available track [last resort]")

        # Extract to temporary SRT
        temp_srt = BASE_DIR / "temp_subtitles.srt"
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-map', f"0:{selected_track['index']}",
            '-y',
            str(temp_srt)
        ]

        try:
            subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', check=True)
        except:
            print("  Failed to extract subtitles from embedded track")
            temp_srt.unlink(missing_ok=True)
            # Fall through to external SRT / download
            tracks = []

        if tracks:
            subtitles = parse_srt(temp_srt)
            temp_srt.unlink(missing_ok=True)

            if subtitles:
                return save_subtitle_chunks(subtitles, prefix)
            else:
                print("  No subtitles found in extracted track")
                # Fall through

    # No embedded tracks (or extraction failed) — try external SRT
    print("  No usable embedded subtitle tracks")

    external_srt = SUBTITLES_DIR / f"{prefix}.srt"
    if external_srt.exists():
        print(f"  Found external SRT file: {external_srt.name}")
        subtitles = parse_srt(external_srt)
        if subtitles:
            return save_subtitle_chunks(subtitles, prefix)
        else:
            print("  External SRT file is empty or invalid")

    # Try auto-download
    if download_subtitles(video_path, prefix):
        external_srt = SUBTITLES_DIR / f"{prefix}.srt"
        subtitles = parse_srt(external_srt)
        if subtitles:
            return save_subtitle_chunks(subtitles, prefix)
        else:
            print("  Downloaded SRT file is empty or invalid")

    print("  Continuing without subtitles")
    return True  # Not fatal — we can still process video

# ============================================================================
# STEP 3: SPLIT VIDEO
# ============================================================================

def get_video_info(video_path):
    """Get video metadata (duration, width, height, fps)."""
    try:
        cmd_format = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', str(video_path)
        ]
        result = subprocess.run(cmd_format, capture_output=True, text=True, check=True)
        format_data = json.loads(result.stdout)

        cmd_stream = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', str(video_path)
        ]
        result = subprocess.run(cmd_stream, capture_output=True, text=True, check=True)
        stream_data = json.loads(result.stdout)

        duration = float(format_data['format']['duration'])
        stream = stream_data['streams'][0]

        return {
            'duration': duration,
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'fps': eval(stream.get('r_frame_rate', '30/1'))
        }
    except:
        return None

def split_video(video_path, prefix):
    """Split video into 30s chunks at 720p, 2fps, NVENC p7."""
    print(f"Splitting video: {video_path.name}...")

    info = get_video_info(video_path)
    if not info:
        print("  Failed to analyze video")
        return False

    print(f"  Duration: {format_time(info['duration'])}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  Settings: {VIDEO_QUALITY}p max, {VIDEO_FPS}fps, NVENC {NVENC_PRESET}")

    total_chunks = int(info['duration'] // CHUNK_DURATION)
    if info['duration'] % CHUNK_DURATION > 0:
        total_chunks += 1

    print(f"  Will create {total_chunks} chunks")

    OUTPUTS_DIR.mkdir(exist_ok=True)

    successful = 0
    skipped = 0

    for chunk_num in range(1, total_chunks + 1):
        start_time = (chunk_num - 1) * CHUNK_DURATION

        if chunk_num == total_chunks:
            duration = info['duration'] - start_time
        else:
            duration = CHUNK_DURATION

        start_ts = format_time(start_time)
        end_ts = format_time(start_time + duration)

        output_file = OUTPUTS_DIR / f"{prefix}.chunk{chunk_num:04d}.{start_ts.replace(':', '-')}_to_{end_ts.replace(':', '-')}.mp4"

        # Resume: skip existing chunks
        if output_file.exists():
            skipped += 1
            successful += 1
            continue

        vf = f"scale='min({VIDEO_QUALITY},iw)':-2,fps={VIDEO_FPS}"

        cmd = [
            "ffmpeg", "-hwaccel", "cuda",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", vf,
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", "0",
            "-an", "-y",
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode == 0:
            successful += 1
            if chunk_num % 10 == 0 or chunk_num == total_chunks:
                print(f"  Progress: {chunk_num}/{total_chunks} chunks")
        else:
            print(f"  Failed to create chunk {chunk_num}")
            return False

    if skipped > 0:
        print(f"  Skipped {skipped} existing chunks")
    print(f"  Created {successful} video chunks total")
    return True

# ============================================================================
# STEP 4: OPENROUTER API / PROCESS CHUNKS
# ============================================================================

def send_to_qwen(api_key, video_path, user_prompt):
    """Send a video clip to Qwen3.5-397B-A17B via OpenRouter."""
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "reasoning": {"effort": "none"}
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vidtest",
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=300
    )
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise Exception(f"API error: {data['error']}")

    result = data["choices"][0]["message"]["content"] or ""

    # Strip thinking tags if model still thinks despite disable flag
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

    usage = data.get("usage", {})

    return result, usage

def get_chunk_subtitles(prefix, chunk_num):
    """Get subtitles for a specific chunk."""
    pattern = f"{prefix}.chunk{chunk_num:04d}.*.txt"
    files = list(SUBTITLES_DIR.glob(pattern))

    if not files:
        return None

    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return None

def process_chunks(prefix, api_key):
    """Process all video chunks through OpenRouter Qwen."""
    print(f"Processing video chunks...")

    video_files = sorted(OUTPUTS_DIR.glob(f"{prefix}.chunk*.mp4"))

    if not video_files:
        print("  No video chunks found")
        return False

    print(f"  Found {len(video_files)} chunks to process")

    DESCRIPTIONS_DIR.mkdir(exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for video_file in video_files:
        # Extract chunk number
        match = re.search(r'chunk(\d+)', video_file.stem)
        chunk_num = int(match.group(1)) if match else 0

        # Extract time range for description filename
        match = re.search(r'chunk(\d+)\.(\d+-\d+-\d+)_to_(\d+-\d+-\d+)', video_file.stem)
        if match:
            chunk_num_str = match.group(1)
            start_time = match.group(2)
            end_time = match.group(3)
        else:
            chunk_num_str = f"{chunk_num:04d}"
            start_time = "00-00-00"
            end_time = "00-00-00"

        desc_file = DESCRIPTIONS_DIR / f"{prefix}.chunk{chunk_num_str}.{start_time}_to_{end_time}.txt"

        # Resume: skip already processed
        if desc_file.exists():
            print(f"  [{chunk_num}/{len(video_files)}] already done, skipping")
            continue

        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"  [{chunk_num}/{len(video_files)}] {video_file.name} ({file_size_mb:.1f} MB)...", end=" ", flush=True)

        # Build prompt with context
        subtitles = get_chunk_subtitles(prefix, chunk_num)

        if subtitles:
            enhanced_prompt = USER_PROMPT
            enhanced_prompt += f"\n\n**Dialogue/Subtitles for this scene:**\n{subtitles}"
        else:
            enhanced_prompt = "Briefly describe this video in under 200 words."

        try:
            start = time.time()
            description, usage = send_to_qwen(api_key, video_file, enhanced_prompt)
            elapsed = time.time() - start

            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            cost = (in_tok * 0.55 / 1_000_000) + (out_tok * 3.50 / 1_000_000)

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_cost += cost

            # Save description
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write(f"Video: {video_file.name}\n")
                f.write(f"Model: {OPENROUTER_MODEL}\n")
                f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Tokens: {in_tok} in / {out_tok} out (${cost:.4f})\n")
                f.write("-" * 60 + "\n")
                f.write(description)

            print(f"OK ({elapsed:.1f}s, {in_tok}+{out_tok} tok, ${cost:.4f})")

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    print()
    print(f"  Tokens: {total_input_tokens:,} input + {total_output_tokens:,} output")
    print(f"  Cost:   ${total_cost:.4f}")
    return True

# ============================================================================
# STEP 5: COMBINE OUTPUTS
# ============================================================================

def combine_outputs(prefix):
    """Combine subtitles and descriptions into [DIALOGUE] + [VISUAL DESCRIPTION] per chunk."""
    print(f"Combining outputs...")

    subtitle_files = sorted(SUBTITLES_DIR.glob(f"{prefix}.chunk*.txt"))
    desc_files = sorted(DESCRIPTIONS_DIR.glob(f"{prefix}.chunk*.txt"))

    # Use whichever set exists — we need at least descriptions
    if not desc_files:
        print("  No description files found")
        return False

    RESULTS_DIR.mkdir(exist_ok=True)

    combined_path = RESULTS_DIR / f"{prefix}_analysis.txt"

    processed = 0
    with open(combined_path, 'w', encoding='utf-8') as out:
        out.write(f"Video Analysis: {prefix}\n")
        out.write(f"Model: {OPENROUTER_MODEL}\n")
        out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("=" * 60 + "\n\n")

        for desc_file in desc_files:
            # Parse chunk info from filename
            match = re.search(r'chunk(\d+)\.(\d+-\d+-\d+)_to_(\d+-\d+-\d+)', desc_file.name)
            if not match:
                continue

            chunk_num = match.group(1)
            start_time = match.group(2).replace('-', ':')
            end_time = match.group(3).replace('-', ':')

            # Read description (skip header lines)
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    separator_idx = next(i for i, line in enumerate(lines) if '----' in line)
                    description = ''.join(lines[separator_idx + 1:]).strip()
            except:
                description = desc_file.read_text(encoding='utf-8').strip()

            # Read matching subtitle chunk
            sub_file = SUBTITLES_DIR / desc_file.name
            dialogue = ""
            if sub_file.exists():
                try:
                    dialogue = sub_file.read_text(encoding='utf-8').strip()
                except:
                    pass

            # Build chunk content
            chunk_content = f"=== CHUNK {chunk_num}: {start_time} to {end_time} ===\n\n"
            if dialogue:
                chunk_content += "[DIALOGUE]\n"
                chunk_content += f"{dialogue}\n\n"
            chunk_content += "[VISUAL DESCRIPTION]\n"
            chunk_content += f"{description}\n"

            # Write to combined file
            out.write(chunk_content + "\n")

            # Write individual chunk file
            chunk_path = RESULTS_DIR / f"{prefix}.chunk{chunk_num}.{start_time.replace(':', '-')}_to_{end_time.replace(':', '-')}.txt"
            with open(chunk_path, 'w', encoding='utf-8') as cf:
                cf.write(chunk_content)

            processed += 1

    print(f"  Combined {processed} chunks -> {combined_path.name}")
    print(f"  Individual chunks -> {RESULTS_DIR}/")
    return True

# ============================================================================
# STEP 6: POST-PROCESS WITH CLAUDE
# ============================================================================

def postprocess_with_claude(prefix):
    """Run Claude Code headless to deduplicate and compress visual descriptions."""
    print(f"Post-processing with Claude Code...")

    # Check that claude is available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("  'claude' CLI not found in PATH. Skipping post-processing.")
        print("  Install Claude Code to enable this step.")
        return False
    except subprocess.CalledProcessError:
        pass  # --version might return non-zero but CLI exists

    # Check that CLAUDE.md exists in the postprocess dir
    claude_md = POSTPROCESS_DIR / "CLAUDE.md"
    if not claude_md.exists():
        print(f"  CLAUDE.md not found in {POSTPROCESS_DIR}/")
        print("  Skipping post-processing.")
        return False

    # Check for chunk files to process
    chunk_files = sorted(RESULTS_DIR.glob(f"{prefix}.chunk*.txt"))
    if not chunk_files:
        print("  No chunk files found to post-process.")
        return False

    # Check if already processed
    existing_output = sorted(POSTPROCESS_DIR.glob(f"{prefix}.chunk*.txt"))
    if len(existing_output) >= len(chunk_files):
        print(f"  Post-processed files already exist ({len(existing_output)} files)")
        print("  Skipping post-processing...")
        return True

    print(f"  Processing {len(chunk_files)} chunks...")

    prompt = f"Please process all chunks for {prefix}, thank you!"

    cmd = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "--verbose",
        "--allowedTools", "Read,Write,Edit,Glob",
        "--model", "sonnet[1m]",
        "--effort", "low",
        "-p", prompt,
    ]

    timeout = 1800  # 30 minutes

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            cwd=str(POSTPROCESS_DIR),
        )

        # Read stream-json events line by line and print tool calls as they happen
        stderr_lines = []
        current_tool_name = None
        current_tool_input = {}
        current_tool_input_str = ""

        def _read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)

        stderr_thread = Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        start_time = time.time()

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            # Tool call start — grab the tool name
            if etype == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool_name = block.get("name", "")
                    current_tool_input_str = ""

            # Tool call input arrives in deltas
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "input_json_delta":
                    current_tool_input_str += delta.get("partial_json", "")

            # Tool call complete — parse input and print
            elif etype == "content_block_stop" and current_tool_name:
                elapsed = int(time.time() - start_time)
                try:
                    tool_input = json.loads(current_tool_input_str) if current_tool_input_str else {}
                except json.JSONDecodeError:
                    tool_input = {}

                # Format the log line based on tool type
                if current_tool_name == "Write":
                    file_path = tool_input.get("file_path", "")
                    filename = Path(file_path).name if file_path else "?"
                    print(f"  [{elapsed}s] Write({filename})", flush=True)
                elif current_tool_name == "Edit":
                    file_path = tool_input.get("file_path", "")
                    filename = Path(file_path).name if file_path else "?"
                    print(f"  [{elapsed}s] Edit({filename})", flush=True)
                elif current_tool_name == "Read":
                    file_path = tool_input.get("file_path", "")
                    filename = Path(file_path).name if file_path else "?"
                    print(f"  [{elapsed}s] Read({filename})", flush=True)
                elif current_tool_name not in ("Think",):
                    # Log other tool calls by name only (skip Think to reduce noise)
                    print(f"  [{elapsed}s] {current_tool_name}()", flush=True)

                current_tool_name = None
                current_tool_input_str = ""

        process.wait(timeout=timeout)

        if process.returncode == 0:
            output_files = sorted(POSTPROCESS_DIR.glob(f"{prefix}.chunk*.txt"))
            print(f"  Post-processing complete: {len(output_files)} files written")
            return True
        else:
            print(f"  Claude exited with code {process.returncode}")
            stderr_output = "".join(stderr_lines)
            if stderr_output:
                print(f"  stderr: {stderr_output[:500]}")
            return False

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"  Claude timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  Post-processing failed: {e}")
        return False

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print()
    print("=" * 60)
    print("    VIDEO ANALYZER - All-in-One Pipeline")
    print("    OpenRouter Qwen3.5-397B-A17B")
    print("=" * 60)
    print()

    print("Pipeline:")
    print("  1. Select video from Source/")
    print("  2. Extract/download subtitles -> 30s chunks")
    print("  3. Split video -> 30s chunks (720p, 2fps, NVENC p7)")
    print("  4. Process each chunk through OpenRouter Qwen")
    print("  5. Combine [DIALOGUE] + [VISUAL DESCRIPTION]")
    print("  6. Post-process with Claude (deduplicate descriptions)")
    print("  7. Generate analysis file")
    print()
    print(f"Settings: {CHUNK_DURATION}s chunks, {VIDEO_QUALITY}p, {VIDEO_FPS}fps, {NVENC_PRESET}")
    print()

    if not HAS_SUBTITLE_DEPS:
        print("NOTE: subliminal/guessit/babelfish not installed.")
        print("  Auto subtitle download and smart prefix generation disabled.")
        print("  Install with: pip install subliminal guessit babelfish")
        print()

    api_key = get_api_key()

    # STEP 1: Select video
    print_step(1, 7, "SELECT VIDEO")
    video_path, prefix = select_video()

    if not video_path or not prefix:
        print("\nCancelled.")
        return

    print(f"\n  Selected: {video_path.name}")
    print(f"  Prefix: {prefix}")

    # Movie context (optional)
    print()
    print("Enter movie context (cast, year, etc.) or leave blank:")
    context = input("> ").strip()
    if context:
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip("\n") + f"\n\n**Current movie context**\n{context}\n"

    # STEP 2: Extract/download subtitles
    print_step(2, 7, "EXTRACT SUBTITLES")
    SUBTITLES_DIR.mkdir(exist_ok=True)

    existing_sub_chunks = list(SUBTITLES_DIR.glob(f"{prefix}.chunk*.txt"))
    if existing_sub_chunks:
        print(f"  Subtitle chunks already exist ({len(existing_sub_chunks)} files)")
        print("  Skipping subtitle extraction...")
    else:
        if not extract_subtitles(video_path, prefix):
            print("  Subtitle extraction failed. Continuing anyway...")

    # STEP 3: Split video
    print_step(3, 7, "SPLIT VIDEO")
    existing_video_chunks = list(OUTPUTS_DIR.glob(f"{prefix}.chunk*.mp4"))
    if existing_video_chunks:
        info = get_video_info(video_path)
        if info:
            expected = int(info['duration'] // CHUNK_DURATION)
            if info['duration'] % CHUNK_DURATION > 0:
                expected += 1
            if len(existing_video_chunks) >= expected:
                print(f"  Video chunks already exist ({len(existing_video_chunks)} files)")
                print("  Skipping video splitting...")
            else:
                print(f"  Found partial video chunks ({len(existing_video_chunks)}/{expected})")
                if not split_video(video_path, prefix):
                    print("  Video splitting failed. Exiting.")
                    return
        else:
            if not split_video(video_path, prefix):
                print("  Video splitting failed. Exiting.")
                return
    else:
        if not split_video(video_path, prefix):
            print("  Video splitting failed. Exiting.")
            return

    # STEP 4: Process chunks through OpenRouter
    print_step(4, 7, "PROCESS WITH QWEN")
    if not process_chunks(prefix, api_key):
        print("  Processing failed.")
        return

    # STEP 5: Combine outputs
    print_step(5, 7, "COMBINE OUTPUTS")
    if not combine_outputs(prefix):
        print("  Combining outputs failed.")

    # STEP 6: Post-process with Claude
    print_step(6, 7, "POST-PROCESS WITH CLAUDE")
    POSTPROCESS_DIR.mkdir(exist_ok=True)
    if not postprocess_with_claude(prefix):
        print("  Post-processing failed or skipped. Raw results still available.")

    # STEP 7: Summary
    print_step(7, 7, "COMPLETE")
    print(f"  Results:       {RESULTS_DIR}/")
    print(f"  Post-processed: {POSTPROCESS_DIR}/")
    print(f"  Descriptions:  {DESCRIPTIONS_DIR}/")
    print(f"  Subtitles:     {SUBTITLES_DIR}/")
    print(f"  Video chunks:  {OUTPUTS_DIR}/")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
