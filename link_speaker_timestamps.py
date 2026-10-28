import os
import re
import json
import pandas as pd
from difflib import SequenceMatcher

# ---------------------------
# CONFIG
# ---------------------------
SRT_ROOT = "data/raw_srt"
SPEAKER_LOG_PATH = "data/speaker_mapping_log.json"
OUTPUT_PATH = "processed/speaker_segments_with_time.csv"

MATCH_THRESHOLD = 0.85   # adjust if needed

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # remove speaker labels
    text = re.sub(r"Speaker \d+:", "", text)

    # remove punctuation (Arabic + English)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------------------
# SIMILARITY FUNCTION
# ---------------------------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------
# PARSE SRT FILE
# ---------------------------
def parse_srt(file_path, episode_name):
    segments = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content)

    for block in blocks:
        lines = block.strip().split("\n")

        if len(lines) >= 3:
            try:
                time_line = lines[1]
                start, end = time_line.split(" --> ")

                text = " ".join(lines[2:])
                cleaned = clean_text(text)

                segments.append({
                    "episode": episode_name,
                    "start": start.strip(),
                    "end": end.strip(),
                    "text": text.strip(),
                    "clean_text": cleaned
                })
            except:
                continue

    return segments

# ---------------------------
# LOAD ALL SRT DATA
# ---------------------------
all_srt_segments = []

print("📂 Loading SRT files...")

for file in os.listdir(SRT_ROOT):
    if file.endswith(".srt"):
        episode_name = file.replace(".srt", "")
        file_path = os.path.join(SRT_ROOT, file)

        segments = parse_srt(file_path, episode_name)
        all_srt_segments.extend(segments)

srt_df = pd.DataFrame(all_srt_segments)

print(f"✅ Loaded {len(srt_df)} SRT segments")

# ---------------------------
# LOAD SPEAKER LOGS
# ---------------------------
with open(SPEAKER_LOG_PATH, "r", encoding="utf-8") as f:
    speaker_logs = json.load(f)

print(f"✅ Loaded {len(speaker_logs)} speaker log entries")

# Pre-clean SRT text (speed optimization)
srt_df["clean_text"] = srt_df["clean_text"].fillna("")

# ---------------------------
# MATCHING
# ---------------------------
rows = []
used_indices = set()

print("🔍 Matching speaker logs with SRT...")

for i, log in enumerate(speaker_logs):
    log_text_raw = log.get("text", "")
    log_text = clean_text(log_text_raw)

    if not log_text:
        continue

    best_match_idx = None
    best_score = 0

    # Loop over SRT rows
    for idx, srt_row in srt_df.iterrows():
        if idx in used_indices:
            continue  # prevent duplicate matching

        score = similarity(log_text, srt_row["clean_text"])

        if score > best_score:
            best_score = score
            best_match_idx = idx

    # Accept match if above threshold
    if best_match_idx is not None and best_score >= MATCH_THRESHOLD:
        used_indices.add(best_match_idx)

        match = srt_df.loc[best_match_idx]

        rows.append({
            "segment_id": i + 1,
            "episode": match["episode"],
            "start": match["start"],
            "end": match["end"],
            "speaker": log.get("global_speaker"),
            "text": log_text_raw,
            "match_score": round(best_score, 3)
        })

    # Progress log (every 100 rows)
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(speaker_logs)} logs")

# ---------------------------
# SAVE OUTPUT
# ---------------------------
final_df = pd.DataFrame(rows)

final_df.to_csv(OUTPUT_PATH, index=False)

print("\n DONE!")
print(f" Matched segments: {len(final_df)} / {len(speaker_logs)}")
print(f" Output saved at: {OUTPUT_PATH}")