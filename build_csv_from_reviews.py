from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re

# ---------- helpers ----------


def _normalize_spaces(s: str) -> str:
    """Collapse any run of whitespace into single spaces."""
    return " ".join(s.split())


def _extract_visible_text(html_fragment: str) -> str:
    """
    Parse an HTML/XML-ish string and return only the visible text.
    Using lxml parser (fast & forgiving). Then normalize spaces.
    """
    soup = BeautifulSoup(html_fragment, "lxml")
    return _normalize_spaces(soup.get_text(separator=" ", strip=True))


def _comment_from_unique_id(text: str) -> str:
    """
    If a line looks like: 0312355645:horrible_book,_horrible.:mark_gospri
    keep only the middle piece (assumed to be the actual review/comment).
    Also replace underscores/commas and normalize spaces.
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
    Extract a list of review strings from a single .review file.

    Strategy:
    - If <review> tags exist: one review per tag.
      - If a <unique_id> exists inside, extract it and keep only the comment part.
      - Else, use the text content of the <review>.
    - If there are no <review> tags:
      - Extract all visible text from the file and try to pull a comment
        from id:comment:author structure if present.

    Finally, deduplicate reviews within this file.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    reviews: List[str] = []

    # Find explicit <review> tags
    tags = soup.find_all("review")
    if tags:
        for tag in tags:
            # Prefer <unique_id> content when present (common in your data)
            uid = tag.find("unique_id")
            if uid and uid.get_text(strip=True):
                text = _comment_from_unique_id(
                    _normalize_spaces(uid.get_text(" ", strip=True))
                )
            else:
                # Fallback: just the text of the <review> tag
                text = _normalize_spaces(tag.get_text(" ", strip=True))

            # Clean any stray markup remnants and normalize
            text = re.sub(r"<[^>]+>", " ", text)
            text = _normalize_spaces(text)

            # Keep non-trivial reviews
            if len(text) >= 2:
                reviews.append(text)
    else:
        # No <review> tags: treat the entire file as text, then try to pull comment middle
        text = _extract_visible_text(raw)
        if text:
            text = _comment_from_unique_id(text)
            reviews.append(text)

    # Deduplicate reviews from this single file (preserve first seen)
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
    min_words: int = 2,
) -> None:
    """
    Walk each category folder under `base_dir` and collect reviews.

    We expect a layout like:
      data/
        books/
          positive.review
          negative.review
          unlabeled.review (or 'unlabaled.review')
        electronics/
          ...

    Behavior:
      - Read positive.review  -> label='positive'
      - Read negative.review  -> label='negative'
      - Read unlabaled/unlabeled.review -> collected separately (no label)
      - Each file may contain multiple <review> tags (we extract them all)
      - Filter out too-short texts (by word count)
      - Drop duplicates by review text
      - Write two CSVs:
          * Labeled:   columns: category, text, label
          * Unlabeled: columns: category, text
    """
    base = Path(base_dir)
    if not base.exists():
        raise SystemExit(f"Base folder not found: {base_dir}")

    labeled_rows: List[Dict[str, str]] = []
    unlabeled_rows: List[Dict[str, str]] = []

    # Each subfolder (e.g., books, dvd, electronics, etc.)
    category_dirs = [p for p in base.iterdir() if p.is_dir()]

    for cat_dir in tqdm(category_dirs, desc="Categories"):
        category_name = cat_dir.name

        # Collect from labeled files first
        for fname, label in [
            ("negative.review", "negative"),
            ("positive.review", "positive"),
        ]:
            fpath = cat_dir / fname
            if fpath.exists():
                reviews = _extract_reviews_from_file(fpath)
                for r in reviews:
                    if len(r.split()) >= min_words:
                        labeled_rows.append(
                            {
                                "category": category_name,
                                "text": r,
                                "label": label,
                            }
                        )

        # Collect unlabeled (supporting both 'unlabaled' misspelling and 'unlabeled')
        for unlabeled_name in ("unlabaled.review", "unlabeled.review"):
            f_unl = cat_dir / unlabeled_name
            if f_unl.exists():
                reviews = _extract_reviews_from_file(f_unl)
                for r in reviews:
                    if len(r.split()) >= min_words:
                        unlabeled_rows.append(
                            {
                                "category": category_name,
                                "text": r,
                            }
                        )

    # ---- Build and write the labeled CSV -----------------------------------
    if labeled_rows:
        df_l = pd.DataFrame(labeled_rows)
        # Drop duplicates by exact 'text' content to avoid train duplication
        df_l = df_l.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df_l.to_csv(out_csv_labeled, index=False)
        print(f"Wrote {len(df_l)} labeled rows -> {out_csv_labeled}")
    else:
        print("No labeled rows found (positive/negative).")

    # ---- Build and write the unlabeled CSV (optional) ----------------------
    if out_csv_unlabeled and unlabeled_rows:
        df_u = pd.DataFrame(unlabeled_rows)
        df_u = df_u.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df_u.to_csv(out_csv_unlabeled, index=False)
        print(f"Wrote {len(df_u)} unlabeled rows -> {out_csv_unlabeled}")


if __name__ == "__main__":
    # Entry point: you can change paths or filenames as needed
    build_reviews_from_data(
        base_dir="data",
        out_csv_labeled="reviews.csv",
        out_csv_unlabeled="unlabeled_reviews.csv",
        min_words=2,
    )
