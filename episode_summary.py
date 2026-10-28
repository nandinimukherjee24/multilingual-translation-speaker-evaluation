import pandas as pd

df = pd.read_csv("outputs/final_evaluation.csv")

summary = df.groupby("episode").agg({
    "similarity_score": "mean",
    "fluency_score": "mean",
    "hallucination_flag": "mean",
    "final_score": "mean"
}).reset_index()

# Rename for clarity
summary.columns = [
    "episode",
    "avg_similarity",
    "avg_fluency",
    "hallucination_rate",
    "avg_final_score"
]

# Save
summary.to_csv("outputs/episode_summary.csv", index=False)

print(" Episode summary created.")