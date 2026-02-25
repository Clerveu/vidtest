"""Single clip tester for OpenRouter Qwen3.5-397B-A17B."""

import os
import sys
import base64
import json
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "Outputs"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3.5-397b-a17b"

SYSTEM_PROMPT = """You are narrating a portion of a film. Your goal is to very briefly describe the visuals happening on screen.

**SYSTEM INSTRUCTIONS FOR VISUAL-ONLY DESCRIPTIONS**

1. **Literal Visuals Only**: Describe only elements clearly visible on screen
2. **Zero Interpretation**: No speculation about emotions, motivations, intentions, backstory, or symbolic meaning
3. **Neutral Language**: Use objective descriptors. "A person's eyes water" not "the person appears sad"
4. **Action Without Motivation**: "A man walks toward the door" not "rushes anxiously toward the exit"
5. **Camera Work When Notable**: Mention obvious techniques (zooms, pans, tracking shots, split-diopter, dolly zooms, focus pulls) only when prominent
6. **No Contextual Assumptions**: Do not infer genre, story arc, historical setting, or character relationships
7. **Direct Language**: Simple, factual reporting. No poetic or dramatic phrasing
8. **Objective Stance**: Never use "we see," "it seems," "this suggests," "perhaps," "maybe," "likely," "possibly," "suggesting," "there may be," "resembles"
9. **Report Only What's Present**: Do not mention absences or omissions
10. **Never Reference These Rules**: Do not mention these instructions in your output
11. **Don't assume speaker**: Without concrete evidence do not correlate any given subtitle with any given character
12. **Don't reproduce subtitles**: Only use subtitles to give yourself context to better describe visuals
13. **Rely heavily on subtitles for visual context**
14. **Be Concise**: Maximum 200 words

Thank you so much Qwen!
"""

def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    key_file = BASE_DIR / ".openrouter_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = input("OpenRouter API key: ").strip()
    if not key:
        sys.exit(1)
    key_file.write_text(key)
    return key


def main():
    api_key = get_api_key()

    clips = sorted(f for f in OUTPUTS_DIR.iterdir() if f.suffix.lower() == ".mp4")
    if not clips:
        print("No clips in Outputs/")
        sys.exit(1)

    print(f"\n{len(clips)} clips available:\n")
    for i, c in enumerate(clips, 1):
        size_kb = c.stat().st_size / 1024
        print(f"  {i}. {c.name} ({size_kb:.0f} KB)")

    while True:
        try:
            idx = int(input(f"\nPick clip (1-{len(clips)}): ")) - 1
            if 0 <= idx < len(clips):
                break
        except (ValueError, EOFError):
            pass

    clip = clips[idx]

    # Optional context
    print("\nMovie context (or blank):")
    context = input("> ").strip()
    sys_prompt = SYSTEM_PROMPT
    if context:
        sys_prompt += f"\n**Current movie context**\n{context}\n"

    # Optional custom prompt
    print("\nCustom user prompt (or blank for default):")
    user_prompt = input("> ").strip()
    if not user_prompt:
        user_prompt = "Briefly describe this video in under 200 words."

    # Send
    file_size_mb = clip.stat().st_size / (1024 * 1024)
    print(f"\nSending {clip.name} ({file_size_mb:.1f} MB)...", flush=True)

    with open(clip, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    start = time.time()
    resp = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
                ]}
            ],
            "max_tokens": 1024,
            "reasoning": {"effort": "none"}
        },
        timeout=300
    )
    elapsed = time.time() - start

    data = resp.json()

    # Dump raw response
    log_path = BASE_DIR / "last_response.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nRaw response saved to: {log_path}")

    if "error" in data:
        print(f"\nERROR: {data['error']}")
        sys.exit(1)

    message = data["choices"][0]["message"]
    raw_content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""

    # Show what we got
    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    cost = (in_tok * 0.55 / 1_000_000) + (out_tok * 3.50 / 1_000_000)

    print(f"\n{'=' * 60}")
    print(f"  {elapsed:.1f}s | {in_tok} in + {out_tok} out tokens | ${cost:.4f}")
    print(f"{'=' * 60}")

    # Raw content analysis
    print(f"\n--- RAW CONTENT ({len(raw_content)} chars) ---")
    has_think_tags = "<think>" in raw_content
    print(f"  Contains <think> tags: {has_think_tags}")
    if has_think_tags:
        import re as _re
        think_match = _re.search(r'<think>(.*?)</think>', raw_content, flags=_re.DOTALL)
        if think_match:
            think_len = len(think_match.group(1))
            print(f"  Thinking content length: {think_len} chars")
        cleaned = _re.sub(r'<think>.*?</think>', '', raw_content, flags=_re.DOTALL).strip()
        print(f"  Content after stripping: {len(cleaned)} chars")
    else:
        cleaned = raw_content.strip()

    if reasoning:
        print(f"\n--- REASONING FIELD ({len(reasoning)} chars) ---")
        print(reasoning[:500] + ("..." if len(reasoning) > 500 else ""))

    # Show any extra fields on the message
    extra_keys = [k for k in message.keys() if k not in ("role", "content")]
    if extra_keys:
        print(f"\n--- EXTRA MESSAGE FIELDS: {extra_keys} ---")
        for k in extra_keys:
            val = str(message[k])
            print(f"  {k}: {val[:300]}{'...' if len(val) > 300 else ''}")

    print(f"\n--- FINAL OUTPUT ---\n")
    print(cleaned if cleaned else "(empty)")
    print()


if __name__ == "__main__":
    main()
