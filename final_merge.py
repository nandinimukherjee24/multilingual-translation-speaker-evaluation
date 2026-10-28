import pandas as pd

# Load datasets
df_base = pd.read_csv("processed/evaluation_dataset.csv")
df_metrics = pd.read_csv("outputs/translation_metrics.csv")

#  FIX: rename column
df_metrics.rename(columns={"segment_id": "sentence_id"}, inplace=True)

print("df_base columns:", df_base.columns)
print("df_metrics columns:", df_metrics.columns)

# Merge
df_final = df_base.merge(
    df_metrics[
        ["sentence_id", "similarity_score", "length_ratio", "hallucination_flag", "fluency_score"]
    ],
    on="sentence_id",
    how="left"
)

# Final score (improved version)
df_final["final_score"] = (
    0.6 * df_final["similarity_score"]
    + 0.3 * df_final["fluency_score"]
)

df_final.loc[df_final["hallucination_flag"] == 1, "final_score"] *= 0.5

# Save
df_final.to_csv("outputs/final_evaluation.csv", index=False)

print("Final merged dataset created.")