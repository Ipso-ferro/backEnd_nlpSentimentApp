# pip install openai
import os, io, json
from openai import OpenAI
from dataPreparing import cleaned_rows  # list of {"text": "...", "label": "..."} dicts
import random

client = OpenAI(api_key="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")


from dataPreparing import cleaned_rows

def sample_examples(rows, per_class: int = 3) -> str:
    """Pick a few positive and negative examples to use as in-prompt guidance."""
    positives = [r for r in rows if r.get("label") == "positive"]
    negatives = [r for r in rows if r.get("label") == "negative"]
    random.shuffle(positives)
    random.shuffle(negatives)
    shots = positives[:per_class] + negatives[:per_class]
    random.shuffle(shots)
    return "\n\n".join(f"text: {r['text']}\nlabel: {r['label']}" for r in shots)

def classify(comment: str) -> dict:
    examples_block = sample_examples(cleaned_rows, per_class=3)

    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        max_tokens=20,  # small—only returning 2 fields
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier for short English reviews. "
                    "Return ONLY valid JSON: {\"sentiment\":\"positive|negative\"}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are labeled examples:\n"
                    f"{examples_block}\n\n"
                    "Now classify this text with the same labels.\n"
                    f'text: "{comment}"'
                ),
            },
        ],
        # Most SDK versions support this JSON mode with Chat Completions
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)

if __name__ == "__main__":
    print(classify("awful support, totally broken and late"))
    print(classify("i loved it, super fast delivery and amazing quality"))