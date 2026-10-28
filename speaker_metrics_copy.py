import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# =========================
# INPUT FORMAT EXPECTED
# =========================
# dataframe with:
# speaker | segment_id | embedding (list/np.array)

# Example:
df = pd.read_csv("speaker_embeddings.csv")

# Ensure embeddings are numpy arrays
def to_array(x):
    return np.array(eval(x)) if isinstance(x, str) else np.array(x)

df["embedding"] = df["embedding"].apply(to_array)

# =========================
# GROUP BY SPEAKER
# =========================
speaker_groups = df.groupby("speaker")

intra_similarities = []
inter_similarities = []
temporal_similarities = []

speaker_summary = []

all_embeddings = np.stack(df["embedding"].values)
all_speakers = df["speaker"].values

# =========================
# INTRA-SPEAKER SIMILARITY
# =========================
for speaker, group in speaker_groups:
    embeddings = np.stack(group["embedding"].values)
    n = len(embeddings)

    if n < 2:
        avg_intra = np.nan  # FIX: no fake 1.0
    else:
        sims = []
        for i in range(n):
            for j in range(i + 1, n):  # FIX: no self comparison
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                sims.append(sim)

        avg_intra = np.mean(sims)
        intra_similarities.extend(sims)

    speaker_summary.append({
        "speaker": speaker,
        "num_segments": n,
        "avg_intra_similarity": avg_intra
    })

# =========================
# INTER-SPEAKER SIMILARITY
# =========================
n_total = len(all_embeddings)

for i in range(n_total):
    for j in range(i + 1, n_total):
        if all_speakers[i] != all_speakers[j]:
            sim = cosine_similarity(
                all_embeddings[i].reshape(1, -1),
                all_embeddings[j].reshape(1, -1)
            )[0][0]
            inter_similarities.append(sim)

# =========================
# TEMPORAL CONSISTENCY
# =========================
df_sorted = df.sort_values("segment_id")

for i in range(len(df_sorted) - 1):
    e1 = df_sorted.iloc[i]["embedding"]
    e2 = df_sorted.iloc[i + 1]["embedding"]

    sim = cosine_similarity(
        e1.reshape(1, -1),
        e2.reshape(1, -1)
    )[0][0]

    temporal_similarities.append(sim)

# =========================
# SILHOUETTE SCORE
# =========================
try:
    silhouette = silhouette_score(all_embeddings, all_speakers)
except:
    silhouette = np.nan

# =========================
# SUMMARY METRICS
# =========================
summary = {
    "mean_intra_similarity": np.nanmean(intra_similarities),
    "mean_inter_similarity": np.mean(inter_similarities),
    "std_inter_similarity": np.std(inter_similarities),
    "min_inter_similarity": np.min(inter_similarities),
    "max_inter_similarity": np.max(inter_similarities),
    "silhouette_score": silhouette,
    "mean_temporal_similarity": np.mean(temporal_similarities)
}

# =========================
# SAVE OUTPUTS
# =========================
pd.DataFrame(speaker_summary).to_csv("outputs/speaker_metrics.csv", index=False)
pd.DataFrame([summary]).to_csv("outputs/speaker_summary.csv", index=False)

print(" Speaker evaluation complete")