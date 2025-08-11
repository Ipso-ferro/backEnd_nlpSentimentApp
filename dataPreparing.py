# pip install pandas symspellpy tqdm
import re, json, os
import pandas as pd
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity

# --- basic English stopwords (trim/add as needed) ---
STOPWORDS = {
    "a","an","the","and","or","but","if","then","so","of","at","by","for","to",
    "in","on","with","as","is","it","this","that","these","those","am","are","was",
    "were","be","been","being","i","you","he","she","we","they","them","me","my",
    "mine","your","yours","his","her","its","our","ours","their","theirs"
}

# --- optional spell corrector (small dictionary) ---
def build_symspell():
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # Download once: https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt
    sym.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    return sym
symspell = build_symspell()

def clean_text(
    text,
    max_tokens=128,
    do_spell=True
):
    # 1) lowercase
    t = text.lower()

    # 2) remove digits & punctuation except apostrophes **inside** words
    # keep letters, spaces and apostrophes, then strip apostrophes not between letters
    t = re.sub(r"[0-9]", " ", t)
    t = re.sub(r"[^a-z\s']", " ", t)
    t = re.sub(r"(?<![a-z])'(?![a-z])", " ", t)  # remove leading/trailing apostrophes

    # 3) remove stopwords
    tokens = re.findall(r"[a-z]+'?[a-z]+|[a-z]+", t)
    tokens = [tok for tok in tokens if tok not in STOPWORDS]

    # 4) (optional) plain-English spell correction for each token (fast & simple)
    if do_spell:
        corrected = []
        for tok in tokens:
            sug = symspell.lookup(tok, Verbosity.TOP, max_edit_distance=1)
            corrected.append(sug[0].term if sug else tok)
        tokens = corrected

    # 5) padding / truncation to fixed length
    tokens = tokens[:max_tokens]
    if len(tokens) < max_tokens:
        tokens = tokens + (["<pad>"] * (max_tokens - len(tokens)))

    # 6) filter short/irrelevant (example rule: must have ≥ 5 non-pad tokens)
    non_pad = [t for t in tokens if t != "<pad>"]
    if len(non_pad) < 5:
        return None

    # return both token list and rejoined text (you can choose which to feed)
    return " ".join(tokens)

# Example: turn a CSV (columns: text,label) into a list of cleaned items
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)  # expects columns: text, label ("positive"/"negative")
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        cleaned = clean_text(str(r["text"]))
        if cleaned and r["label"] in ("positive","negative"):
            rows.append({"text": cleaned, "label": r["label"]})
    return rows

cleaned_rows = load_and_clean("reviews.csv")
print(len(cleaned_rows), "usable examples")
