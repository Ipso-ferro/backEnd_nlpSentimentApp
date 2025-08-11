import re, json, random, pandas as pd
from pathlib import Path
from typing import Optional

# ---- Small English stopword list used to drop uninformative words ----
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "of",
    "at",
    "by",
    "for",
    "to",
    "in",
    "on",
    "with",
    "as",
    "is",
    "it",
    "this",
    "that",
    "these",
    "those",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "them",
    "me",
    "my",
    "mine",
    "your",
    "yours",
    "his",
    "her",
    "its",
    "our",
    "ours",
    "their",
    "theirs",
}


def clean_text(text: str, max_tokens=128) -> Optional[str]:
    """
    Normalize + lightly tokenize a review string.
    Returns a space-joined string of tokens, or None if too short.
    """
    # 1) lowercase everything
    t = text.lower()

    # 2) remove digits
    t = re.sub(r"[0-9]", " ", t)

    # 3) keep only letters, spaces, and apostrophes
    t = re.sub(r"[^a-z\s']", " ", t)

    # 4) remove apostrophes that are NOT between letters (strip stray quotes)
    #    (?<![a-z])  = previous char is not a letter
    #    '(?![a-z])  = next char is not a letter
    t = re.sub(r"(?<![a-z])'(?![a-z])", " ", t)

    # 5) tokenize: words with an optional internal apostrophe (e.g., don't)
    tokens = re.findall(r"[a-z]+'?[a-z]+|[a-z]+", t)

    # 6) drop stopwords
    tokens = [tok for tok in tokens if tok not in STOPWORDS]

    # 7) truncate to a fixed max length (your own “token” length, not model tokens)
    tokens = tokens[:max_tokens]

    # 8) filter out very short items (less than 5 tokens after cleaning)
    if len(tokens) < 5:
        return None

    # 9) return as a single space-joined string
    return " ".join(tokens)


# System instruction that will be inserted in each training example
SYSTEM = "You are a sentiment classifier. Reply with exactly 'positive' or 'negative'."


def row_to_chat(rec):
    """
    Convert one cleaned row into a chat-format training record for fine-tuning.
    Structure matches OpenAI's supervised fine-tuning format: messages[...]
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Classify the sentiment: {rec['text']}"},
            {"role": "assistant", "content": rec["label"]},
        ]
    }


def main(csv_path="reviews.csv", out_dir="artifacts"):
    # Ensure the output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load the raw labeled dataset (expects columns: text,label)
    df = pd.read_csv(csv_path)

    # Drop rows missing text or label
    df = df.dropna(subset=["text", "label"])

    # Keep only the two target labels
    df = df[df["label"].isin(["positive", "negative"])].copy()

    # Clean the text column
    df["text"] = df["text"].astype(str).map(clean_text)

    # Remove rows that became None after cleaning, and deduplicate by text
    df = (
        df.dropna(subset=["text"])
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    # ---- Train/Val/Test split: 80/10/10 ----
    idx = list(df.index)
    random.shuffle(idx)  # (Tip: set random.seed(...) before this for reproducibility)
    n = len(idx)
    train_idx = idx[: int(0.8 * n)]
    val_idx = idx[int(0.8 * n) : int(0.9 * n)]
    test_idx = idx[int(0.9 * n) :]

    # Slice the dataframe for each split
    splits = {
        "train.jsonl": df.loc[train_idx],
        "val.jsonl": df.loc[val_idx],
        "test.jsonl": df.loc[test_idx],
    }
    print(splits)  # debug print: shows the three DataFrames

    # ---- Write each split to JSONL in chat fine-tune format ----
    for name, part in splits.items():
        with open(Path(out_dir) / name, "w", encoding="utf-8") as f:
            for _, r in part.iterrows():
                # Build one training example with system+user+assistant
                record = row_to_chat({"text": r["text"], "label": r["label"]})
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Final summary
    print(
        f"Done. Wrote {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test to {out_dir}/"
    )


if __name__ == "__main__":
    main()
