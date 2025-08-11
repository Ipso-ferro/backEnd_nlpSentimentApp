# pip install pandas symspellpy tqdm
import re, json, os
import pandas as pd
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity

# --- Basic English stopwords (trim/extend for your domain) -----------------
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

# --- Optional: build a SymSpell spell-corrector ----------------------------
# Uses a frequency dictionary to suggest top corrections for tokens.
# max_dictionary_edit_distance=2 allows up to 2 edits; prefix_length=7 speeds lookups.
def build_symspell():
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # Make sure this file is present; it's bundled in symspellpy or can be downloaded.
    # term_index=0 (word column), count_index=1 (frequency column)
    sym.load_dictionary(
        "frequency_dictionary_en_82_765.txt", term_index=0, count_index=1
    )
    return sym


# Instantiate once and reuse for speed
symspell = build_symspell()


def clean_text(text, max_tokens=128, do_spell=True):
    """
    End-to-end cleaner:
    1) lowercase
    2) strip digits/punct except apostrophes inside words
    3) remove stopwords
    4) (optional) spell-correct each token with SymSpell
    5) pad/truncate to a fixed length
    6) filter items that are too short after cleaning
    Returns a space-joined string of tokens (including '<pad>' if padded), or None if too short.
    """

    # 1) normalize case
    t = text.lower()

    # 2) keep letters/spaces/apostrophes; remove digits & other punctuation
    t = re.sub(r"[0-9]", " ", t)  # remove digits
    t = re.sub(r"[^a-z\s']", " ", t)  # strip non-letters except apostrophe and space
    # remove apostrophes not between letters (e.g., "' hello '" -> " hello ")
    t = re.sub(r"(?<![a-z])'(?![a-z])", " ", t)

    # Tokenize: words with optional internal apostrophe (e.g., don't)
    tokens = re.findall(r"[a-z]+'?[a-z]+|[a-z]+", t)

    # 3) drop stopwords
    tokens = [tok for tok in tokens if tok not in STOPWORDS]

    # 4) optional SymSpell correction (fast, single best suggestion per token)
    if do_spell:
        corrected = []
        for tok in tokens:
            sug = symspell.lookup(tok, Verbosity.TOP, max_edit_distance=1)
            corrected.append(sug[0].term if sug else tok)
        tokens = corrected

    # 5) enforce a fixed-length token sequence (useful for classical ML / RNNs)
    #    For modern LLM fine-tuning, padding in the text itself is usually NOT needed.
    tokens = tokens[:max_tokens]  # truncate
    if len(tokens) < max_tokens:  # pad if short
        tokens = tokens + (["<pad>"] * (max_tokens - len(tokens)))

    # 6) reject examples that are too small (count only non-pad tokens)
    non_pad = [t for t in tokens if t != "<pad>"]
    if len(non_pad) < 5:
        return None

    # Return the cleaned space-joined tokens (downstream can join or split)
    return " ".join(tokens)


# --- Bulk cleaner for a CSV dataset ----------------------------------------
def load_and_clean(csv_path):
    """
    Read a CSV with columns: text,label (label ∈ {positive,negative}),
    apply clean_text() to each row, keep only valid/cleaned rows,
    and return a list of dicts: {"text": cleaned_text, "label": label}.
    """
    df = pd.read_csv(csv_path)  # expects columns: text, label
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        cleaned = clean_text(str(r["text"]))
        if cleaned and r["label"] in ("positive", "negative"):
            rows.append({"text": cleaned, "label": r["label"]})
    return rows


# Run the cleaner and report how many items are usable after filtering
cleaned_rows = load_and_clean("reviews.csv")
print(len(cleaned_rows), "usable examples")
