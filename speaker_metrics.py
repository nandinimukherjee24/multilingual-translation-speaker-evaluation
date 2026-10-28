import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from collections import defaultdict

# ---------------------------
# CONFIG
# ---------------------------
SEGMENTS_PATH = "processed/speaker_segments_with_time.csv"
EMBEDDINGS_PATH = "data/speaker_vectors.json"
OUTPUT_PATH = "outputs/speaker_metrics.csv"

# ---------------------------
# LOAD DATA
# ---------------------------
segments_df = pd.read_csv(SEGMENTS_PATH)

with open(EMBEDDINGS_PATH, "r") as f:
    embeddings_data = json.load(f)

# Convert embeddings to dict
speaker_embeddings = {}
for item in embeddings_data:
    speaker_embeddings[item["speaker_id"]] = np.array(item["embedding"])

# ---------------------------
# MAP EMBEDDINGS TO SEGMENTS
# ---------------------------
embeddings = []
valid_rows = []

for _, row in segments_df.iterrows():
    speaker = row["speaker"]

    if speaker in speaker_embeddings:
        embeddings.append(speaker_embeddings[speaker])
        valid_rows.append(row)

# Convert to arrays
X = np.array(embeddings)
valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)

print(f"Loaded {len(X)} valid speaker segments")

# ---------------------------
# INTRA / INTER SIMILARITY
# ---------------------------
intra_sims = []
inter_sims = []

for i in range(len(X)):
    for j in range(i + 1, len(X)):
        sim = cosine_similarity([X[i]], [X[j]])[0][0]

        if valid_df.loc[i, "speaker"] == valid_df.loc[j, "speaker"]:
            intra_sims.append(sim)
        else:
            inter_sims.append(sim)

mean_intra = np.mean(intra_sims) if intra_sims else 0
mean_inter = np.mean(inter_sims) if inter_sims else 0

print(f" Mean Intra-Speaker Similarity: {mean_intra:.4f}")
print(f" Mean Inter-Speaker Similarity: {mean_inter:.4f}")

# ---------------------------
# CLUSTERING (SILHOUETTE SCORE)
# ---------------------------
labels = valid_df["speaker"].astype("category").cat.codes

if len(set(labels)) > 1:
    sil_score = silhouette_score(X, labels)
else:
    sil_score = -1  # not enough speakers

print(f" Silhouette Score: {sil_score:.4f}")

# ---------------------------
# SPEAKER-WISE STATS
# ---------------------------
speaker_groups = defaultdict(list)

for i, speaker in enumerate(valid_df["speaker"]):
    speaker_groups[speaker].append(X[i])

speaker_stats = []

for speaker, vecs in speaker_groups.items():
    vecs = np.array(vecs)

    if len(vecs) > 1:
        sims = cosine_similarity(vecs)
        avg_sim = np.mean(sims)
    else:
        avg_sim = 1.0

    speaker_stats.append({
        "speaker": speaker,
        "num_segments": len(vecs),
        "avg_self_similarity": avg_sim
    })

speaker_stats_df = pd.DataFrame(speaker_stats)

# ---------------------------
# SAVE RESULTS
# ---------------------------
summary = pd.DataFrame([{
    "mean_intra_similarity": mean_intra,
    "mean_inter_similarity": mean_inter,
    "silhouette_score": sil_score
}])

# Save both
summary.to_csv("outputs/speaker_summary.csv", index=False)
speaker_stats_df.to_csv(OUTPUT_PATH, index=False)

print("\n Speaker evaluation completed!")
print(" Saved:")
print("- outputs/speaker_summary.csv")
print("- outputs/speaker_metrics.csv")