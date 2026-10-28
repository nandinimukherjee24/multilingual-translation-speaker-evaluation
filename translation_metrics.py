import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re


# --- A2: Terminology Helpers ---

def extract_numbers(text):
    return re.findall(r'\d+', str(text))

def extract_capital_words(text):
    return re.findall(r'\b[A-Z][a-z]+\b', str(text))


def terminology_score(source, translation):
    src_nums = set(extract_numbers(source))
    tgt_nums = set(extract_numbers(translation))

    num_score = 1 if len(src_nums) == 0 else len(src_nums & tgt_nums) / len(src_nums)

    src_entities = set(extract_capital_words(source))
    tgt_entities = set(extract_capital_words(translation))

    ent_score = 1 if len(src_entities) == 0 else len(src_entities & tgt_entities) / len(src_entities)

    return 0.7 * num_score + 0.3 * ent_score


# --- A3: Rule Checks ---
NEG_WORDS = ["لا", "لم", "لن", "ما"]

def has_negation(text):
    return any(word in str(text) for word in NEG_WORDS)

def numeric_violation(source, translation):
    return 0 if set(extract_numbers(source)) == set(extract_numbers(translation)) else 1

def negation_violation(source, translation):
    return 0 if has_negation(source) == has_negation(translation) else 1

def length_violation(length_ratio):
    return 1 if (length_ratio < 0.5 or length_ratio > 2.0) else 0
# -----------------------------
# Load model
# -----------------------------
print("Loading model...")
model = SentenceTransformer('sentence-transformers/LaBSE')

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("processed/evaluation_dataset.csv")

# Ensure text columns are strings
df["source_text"] = df["source_text"].astype(str)
df["translation"] = df["translation"].astype(str)

# -----------------------------
# 1. BATCH EMBEDDINGS (FAST)
# -----------------------------
print("Encoding source texts...")
src_embeddings = model.encode(
    df["source_text"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

print("Encoding translated texts...")
tgt_embeddings = model.encode(
    df["translation"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

# -----------------------------
# 2. SIMILARITY COMPUTATION
# -----------------------------
print("Computing similarity scores...")
similarities = [
    cosine_similarity([src_embeddings[i]], [tgt_embeddings[i]])[0][0]
    for i in range(len(df))
]

# -----------------------------
# 3. LENGTH RATIO
# -----------------------------
def length_ratio(src, tgt):
    return len(tgt) / (len(src) + 1e-5)

df["length_ratio"] = [
    length_ratio(src, tgt)
    for src, tgt in zip(df["source_text"], df["translation"])
]

# -----------------------------
# 4. HALLUCINATION DETECTION
# -----------------------------
def detect_hallucination(src, tgt):
    if len(tgt) > 2 * len(src):
        return 1
    if len(tgt.strip()) == 0:
        return 1
    return 0

df["hallucination_flag"] = [
    detect_hallucination(src, tgt)
    for src, tgt in zip(df["source_text"], df["translation"])
]

# -----------------------------
# 5. FLUENCY SCORE
# -----------------------------
def fluency_score(text):
    if len(text.strip()) == 0:
        return 0

    sentences = re.split(r'[.!?।]', text)
    word_counts = [len(s.split()) for s in sentences if len(s.split()) > 0]

    if not word_counts:
        return 0

    avg_len = np.mean(word_counts)

    return min(avg_len / 20, 1.0)

df["fluency_score"] = df["translation"].apply(fluency_score)

# -----------------------------
# ADD SIMILARITY COLUMN
# -----------------------------
df["similarity_score"] = similarities

df["terminology_score"] = df.apply(
    lambda x: terminology_score(x["source_text"], x["translation"]),
    axis=1
)

df["numeric_violation"] = df.apply(
    lambda x: numeric_violation(x["source_text"], x["translation"]),
    axis=1
)

df["negation_violation"] = df.apply(
    lambda x: negation_violation(x["source_text"], x["translation"]),
    axis=1
)

df["length_violation"] = df["length_ratio"].apply(length_violation)

df["violation_flag"] = (
    df["numeric_violation"] +
    df["negation_violation"] +
    df["length_violation"]
)

# -----------------------------
# FINAL SCORE (IMPORTANT)
# -----------------------------
df["final_score"] = (
    0.5 * df["similarity_score"]
    + 0.2 * df["fluency_score"]
    + 0.2 * df["terminology_score"]
)

# Apply penalties
df.loc[df["hallucination_flag"] == 1, "final_score"] *= 0.5
df.loc[df["violation_flag"] > 0, "final_score"] *= 0.7

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_path = "outputs/translation_metrics.csv"
df.to_csv(output_path, index=False)

print(f" Done! File saved at: {output_path}")
