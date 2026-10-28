import pandas as pd
import os
import json

# ---------------------------
# CONFIG
# ---------------------------
ALIGNMENT_ROOT = "data/alignment_csv"
SPEAKER_LOG_PATH = "data/speaker_mapping_log.json"
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_rows = []

# ---------------------------
# STEP 1: LOAD ALL EPISODES
# ---------------------------
for episode_folder in os.listdir(ALIGNMENT_ROOT):
    episode_path = os.path.join(ALIGNMENT_ROOT, episode_folder)

    if not os.path.isdir(episode_path):
        continue

    print(f"Processing {episode_folder}...")

    for file in os.listdir(episode_path):
        if file.endswith(".csv"):
            file_path = os.path.join(episode_path, file)

            try:
                df = pd.read_csv(file_path)

                # Rename columns (adjust if needed)
                df = df.rename(columns={
                    "orig_start": "start",
                    "orig_end": "end",
                    "orig_text": "source_text",
                    "trans_text": "translation"
                })

                # Keep required columns
                df = df[["start", "end", "source_text", "translation"]]

                # Add episode column
                df["episode"] = episode_folder

                all_rows.append(df)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Combine all
full_df = pd.concat(all_rows, ignore_index=True)

# ---------------------------
# STEP 2: CLEAN DATA
# ---------------------------
full_df = full_df.dropna(subset=["source_text", "translation"])
full_df = full_df.drop_duplicates()

# Create sentence ID
full_df = full_df.reset_index(drop=True)
full_df["sentence_id"] = full_df.index + 1

# Save cleaned alignment
alignment_cleaned_path = os.path.join(OUTPUT_DIR, "alignment_cleaned.csv")
full_df.to_csv(alignment_cleaned_path, index=False)

print(f"alignment_cleaned.csv saved")

# ---------------------------
# STEP 3: EVALUATION DATASET
# ---------------------------
eval_df = full_df[[
    "sentence_id",
    "episode",
    "start",
    "end",
    "source_text",
    "translation"
]]

eval_path = os.path.join(OUTPUT_DIR, "evaluation_dataset.csv")
eval_df.to_csv(eval_path, index=False)

print(f"✅ evaluation_dataset.csv saved")

# ---------------------------
# STEP 4: SPEAKER SEGMENTS
# ---------------------------
with open(SPEAKER_LOG_PATH, "r", encoding="utf-8") as f:
    speaker_logs = json.load(f)

speaker_rows = []

for i, entry in enumerate(speaker_logs):
    speaker_rows.append({
        "segment_id": i + 1,
        "speaker": entry.get("global_speaker"),
        "text": entry.get("text")
    })

speaker_df = pd.DataFrame(speaker_rows)
speaker_df = speaker_df.dropna(subset=["text", "speaker"])

speaker_path = os.path.join(OUTPUT_DIR, "speaker_segments.csv")
speaker_df.to_csv(speaker_path, index=False)

print(f"speaker_segments.csv saved")

print("\nALL EPISODES PROCESSED SUCCESSFULLY!")