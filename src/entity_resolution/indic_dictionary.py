"""
Indic script token dictionary learning and application.

The nine major Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
Tamil, Telugu, Kannada, Malayalam) share the ISCII-derived layout, so a single
offset->Latin table transliterates all of them. However, the same Indic word can be
transliterated in multiple ways (e.g., "प्राइवेट" -> "pvt" / "private" / "pivate").
We learn a token-level dictionary from the training data to map common Indic tokens
to their most frequent Latin form.
"""
import re
import json
from pathlib import Path
from collections import Counter

import polars as pl

from .config import cache_path, SEED

DICT_PATH = cache_path("indic_dict.json")
_INDIC_RE = re.compile("[ऀ-ൿ]")


def _indic_tokens(text: str):
    if not _INDIC_RE.search(text):
        return []
    return _INDIC_RE.findall(text)


def learn():
    """Learn token mapping from training data (both sources)."""
    print("Learning Indic token dictionary...")
    mapping = Counter()
    for src in ("source2", "source3"):
        df = pl.read_parquet(cache_path(f"train_{src}.parquet"))
        for name in df["business_name"].to_list():
            for tok in _indic_tokens(name):
                mapping[tok] += 1
    # Keep tokens seen at least 5 times
    filtered = {k: v for k, v in mapping.items() if v >= 5}
    print(f"  learned {len(filtered)} Indic tokens (>=5 occurrences)")
    json.dump(filtered, open(DICT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  saved to {DICT_PATH}")


def load():
    if not DICT_PATH.exists():
        learn()
    return json.load(open(DICT_PATH, encoding="utf-8"))


def apply(split: str, mapping: dict):
    """Apply dictionary to add transliterated variants to name_norm/name_core."""
    for src in ("source1", "source2", "source3"):
        p = cache_path(f"{split}_{src}_norm.parquet")
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        def replace_indic(text):
            if not text or not _INDIC_RE.search(text):
                return text
            for k, v in mapping.items():
                text = text.replace(k, v)
            return text
        df = df.with_columns(
            pl.col("name_norm").map_elements(replace_indic, return_dtype=pl.String).alias("name_norm"),
            pl.col("name_core").map_elements(replace_indic, return_dtype=pl.String).alias("name_core"),
        )
        df.write_parquet(p)
        print(f"[{split}/{src}] applied Indic dictionary")


if __name__ == "__main__":
    learn()