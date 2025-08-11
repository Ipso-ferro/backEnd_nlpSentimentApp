from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re

# ---------- helpers ----------

def _normalize_spaces(s: str) -> str:
    return " ".join(s.split())

def _extract_visible_text(html_fragment: str) -> str:
    """Return visible text from an HTML/XML-ish fragment."""
    soup = BeautifulSoup(html_fragment, "lxml")
    return _normalize_spaces(soup.get_text(separator=" ", strip=True))

def _comment_from_unique_id(text: str) -> str:
    """
    Many lines look like: 0312355645:horrible_book,_horrible.:mark_gospri
    Keep only the middle piece (the comment/title).
    """
    parts = text.split(":")
    if len(parts) >= 3:
        comment = parts[1]
    else:
        comment = text
    comment = comment.replace("_", " ")
    comment = comment.replace(",", " ")
    comment = _normalize_spaces(comment)
    return comment

def _extract_reviews_from_file(path: Path) -> List[str]:
    """
    Try to extract one review per <review> tag. If none, fall back to the whole file text.
    Also handle <unique_id> inside <review> specially to keep only the 'comment' part.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    reviews: List[str] = []

    tags = soup.find_all("review")
    if tags:
        for tag in tags:
            # Prefer unique_id content if present (common in your samples)
            uid = tag.find("unique_id")
            if uid and uid.get_text(strip=True):
                text = _comment_from_unique_id(_normalize_spaces(uid.get_text(" ", strip=True)))
            else:
                text = _normalize_spaces(tag.get_text(" ", strip=True))
            # light cleanup of stray markup remnants
            text = re.sub(r"<[^>]+>", " ", text)
            text = _normalize_spaces(text)
            if len(text) >= 2:
                reviews.append(text)
    else:
        # Fallback: no <review> tags -> take the whole visible text
        text = _extract_visible_text(raw)
        if text:
            # Try to squeeze a comment if it looks like id:comment:author
            text = _comment_from_unique_id(text)
            reviews.append(text)

    # Deduplicate within this file
    seen = set()
    uniq = []
    for r in reviews:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq

# ---------- main builders ----------

def build_reviews_from_data(
    base_dir: str = "data",
    out_csv_labeled: str = "reviews.csv",
    out_csv_unlabeled: Optional[str] = "unlabeled_reviews.csv",
    min_words: int = 2
) -> None:
    """
    Walk each category folder in `base_dir` and read:
      - positive.review  -> label='positive'
      - negative.review  -> label='negative'
      - (unlabaled|unlabeled).review -> collected separately
    Writes:
      - reviews.csv with columns: category, text, label
      - unlabeled_reviews.csv with columns: category, text   (if any + path exists)
    """
    base = Path(base_dir)
    if not base.exists():
        raise SystemExit(f"Base folder not found: {base_dir}")

    labeled_rows: List[Dict[str, str]] = []
    unlabeled_rows: List[Dict[str, str]] = []

    category_dirs = [p for p in base.iterdir() if p.is_dir()]

    for cat_dir in tqdm(category_dirs, desc="Categories"):
        category_name = cat_dir.name

        # Labeled files
        for fname, label in [("negative.review", "negative"), ("positive.review", "positive")]:
            fpath = cat_dir / fname
            if fpath.exists():
                reviews = _extract_reviews_from_file(fpath)
                for r in reviews:
                    if len(r.split()) >= min_words:
                        labeled_rows.append({
                            "category": category_name,
                            "text": r,
                            "label": label,
                        })

        # Unlabeled (handle both spellings)
        for unlabeled_name in ("unlabaled.review", "unlabeled.review"):
            f_unl = cat_dir / unlabeled_name
            if f_unl.exists():
                reviews = _extract_reviews_from_file(f_unl)
                for r in reviews:
                    if len(r.split()) >= min_words:
                        unlabeled_rows.append({
                            "category": category_name,
                            "text": r,
                        })

    # Build DataFrames
    if labeled_rows:
        df_l = pd.DataFrame(labeled_rows)
        # Drop duplicates by text (keep first occurrence)
        df_l = df_l.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df_l.to_csv(out_csv_labeled, index=False)
        print(f"Wrote {len(df_l)} labeled rows -> {out_csv_labeled}")
    else:
        print("No labeled rows found (positive/negative).")

    if out_csv_unlabeled and unlabeled_rows:
        df_u = pd.DataFrame(unlabeled_rows)
        df_u = df_u.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df_u.to_csv(out_csv_unlabeled, index=False)
        print(f"Wrote {len(df_u)} unlabeled rows -> {out_csv_unlabeled}")

if __name__ == "__main__":
    # Change outputs if you want them under artifacts/
    build_reviews_from_data(
        base_dir="data",
        out_csv_labeled="reviews.csv",
        out_csv_unlabeled="unlabeled_reviews.csv",
        min_words=2
    )
