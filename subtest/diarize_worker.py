"""
Standalone diarization worker — runs inside the venv.
Called by analyze_speak.py as a subprocess.

Usage: python diarize_worker.py <audio_path> <output_dir> <prefix> <chunk_duration>
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from pathlib import Path


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    if len(sys.argv) != 5:
        print("Usage: diarize_worker.py <audio_path> <output_dir> <prefix> <chunk_duration>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    prefix = sys.argv[3]
    chunk_duration = int(sys.argv[4])

    device = "cuda"
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set")
        sys.exit(1)

    # Transcribe
    print("Loading Whisper model (large-v3)...")
    model = whisperx.load_model("large-v3", device, compute_type="float16")

    print("Transcribing...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16, language="en")
    print(f"Transcribed: {len(result['segments'])} segments")

    del model

    # Align
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print(f"Aligned: {len(result['segments'])} segments")

    del model_a

    # Diarize
    print("Running speaker diarization...")
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-community-1",
        token=hf_token,
        device=device
    )
    diarize_segments = diarize_model(audio)
    result = assign_word_speakers(diarize_segments, result)
    del diarize_model

    # Collect lines
    all_lines = []
    for seg in result["segments"]:
        start = seg["start"]
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        if text:
            all_lines.append((start, speaker, text))

    if not all_lines:
        print("No transcript lines produced")
        sys.exit(1)

    # Split into chunks and write
    output_dir.mkdir(exist_ok=True)

    max_time = max(t for t, _, _ in all_lines)
    num_chunks = int(max_time // chunk_duration) + 1

    chunks = [[] for _ in range(num_chunks)]
    for start, speaker, text in all_lines:
        idx = min(int(start // chunk_duration), num_chunks - 1)
        chunks[idx].append((start, speaker, text))

    written = 0
    for i in range(num_chunks):
        chunk_num = i + 1
        chunk_start = i * chunk_duration
        chunk_end = (i + 1) * chunk_duration

        start_ts = format_time(chunk_start)
        end_ts = format_time(chunk_end)

        filename = f"{prefix}.chunk{chunk_num:04d}.{start_ts.replace(':', '-')}_to_{end_ts.replace(':', '-')}.txt"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for start, speaker, text in chunks[i]:
                timestamp = format_time(start)
                f.write(f"[{timestamp}] {speaker}: {text}\n")

        if chunks[i]:
            written += 1

    # Write full transcript
    full_path = output_dir / f"{prefix}_diarized.txt"
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(f"Source: {prefix}\n")
        f.write(f"Speakers: {sorted(set(s for _, s, _ in all_lines))}\n")
        f.write(f"{'='*60}\n\n")
        for start, speaker, text in all_lines:
            mins = int(start // 60)
            secs = start % 60
            f.write(f"[{mins:02d}:{secs:05.2f}] {speaker}: {text}\n")

    print(f"Saved {num_chunks} subtitle chunks ({written} with dialogue)")
    print(f"Full transcript: {full_path.name}")


if __name__ == "__main__":
    main()
