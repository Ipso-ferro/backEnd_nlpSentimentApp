# scripts/prepare_for_finetune.py
import os, re, json, random, pandas as pd
from pathlib import Path
from typing import Optional

STOPWORDS = {
    "a","an","the","and","or","but","if","then","so","of","at","by","for","to",
    "in","on","with","as","is","it","this","that","these","those","am","are","was",
    "were","be","been","being","i","you","he","she","we","they","them","me","my",
    "mine","your","yours","his","her","its","our","ours","their","theirs"
}

def clean_text(text: str, max_tokens=128) -> Optional[str]:
    t = text.lower()
    t = re.sub(r"[0-9]", " ", t)
    t = re.sub(r"[^a-z\s']", " ", t)
    t = re.sub(r"(?<![a-z])'(?![a-z])", " ", t)
    tokens = re.findall(r"[a-z]+'?[a-z]+|[a-z]+", t)
    tokens = [tok for tok in tokens if tok not in STOPWORDS]
    tokens = tokens[:max_tokens]
    if len(tokens) < 5:
        return None
    return " ".join(tokens)

SYSTEM = "You are a sentiment classifier. Reply with exactly 'positive' or 'negative'."

def row_to_chat(rec):
    return {
        "messages":[
            {"role":"system","content": SYSTEM},
            {"role":"user","content": f"Classify the sentiment: {rec['text']}"},
            {"role":"assistant","content": rec["label"]}
        ]
    }

def main(csv_path="reviews.csv", out_dir="artifacts"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text","label"])
    df = df[df["label"].isin(["positive","negative"])].copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # split 80/10/10
    idx = list(df.index)
    random.shuffle(idx)
    n = len(idx)
    train_idx = idx[: int(0.8*n)]
    val_idx   = idx[int(0.8*n): int(0.9*n)]
    test_idx  = idx[int(0.9*n):]

    splits = {
        "train.jsonl": df.loc[train_idx],
        "val.jsonl":   df.loc[val_idx],
        "test.jsonl":  df.loc[test_idx],
    }
    print(splits)

    for name, part in splits.items():
        with open(Path(out_dir)/name, "w", encoding="utf-8") as f:
            for _, r in part.iterrows():
                f.write(json.dumps(row_to_chat({"text": r["text"], "label": r["label"]}), ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test to {out_dir}/")

if __name__ == "__main__":
    main()
