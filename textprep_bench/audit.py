"""Data-quality audit and cleaning (duplicates, label conflicts, noise stats)."""
from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF]")
_ELONG = re.compile(r"(\w)\1{2,}")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_key(text) -> str:
    """Key under which texts are considered (near-)duplicates: lower-cased,
    URLs and mentions removed, only alphanumerics and single spaces kept."""
    t = str(text if text is not None else "").lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    return _NON_ALNUM.sub(" ", t).strip()


def audit_dataset(df: pd.DataFrame) -> Dict:
    """Return a dictionary of data-quality indicators for a (text, label) frame."""
    n = len(df)
    keys = df["text"].map(normalize_key)
    out: Dict = {"n_rows": int(n)}
    out["n_empty_text"] = int((keys == "").sum())

    counts = df["label"].value_counts().sort_index()
    out["label_counts"] = {int(k): int(v) for k, v in counts.items()}
    out["positive_rate"] = float(df["label"].mean()) if n else float("nan")
    out["imbalance_ratio"] = (
        float(counts.max() / counts.min()) if len(counts) > 1 else float("inf")
    )

    nonempty = keys != ""
    out["exact_duplicate_rows"] = int(df.duplicated(["text", "label"]).sum())
    grp = pd.DataFrame({"key": keys[nonempty], "label": df.loc[nonempty, "label"]})
    sizes = grp.groupby("key")["label"].transform("size")
    nuniq = grp.groupby("key")["label"].transform("nunique")
    out["redundant_rows_after_normalisation"] = int(
        (grp.duplicated("key")).sum()
    )
    out["rows_in_label_conflicts"] = int((nuniq > 1).sum())
    out["label_conflict_groups"] = int(grp[nuniq > 1]["key"].nunique())
    out["rows_in_duplicate_groups"] = int((sizes > 1).sum())

    tokens = df["text"].astype(str).str.split().map(len)
    out["tokens_mean"] = float(tokens.mean()) if n else float("nan")
    out["tokens_median"] = float(tokens.median()) if n else float("nan")
    out["tokens_p95"] = float(np.percentile(tokens, 95)) if n else float("nan")

    text = df["text"].astype(str)
    for name, rx in (("url", _URL), ("mention", _MENTION), ("hashtag", _HASHTAG),
                     ("emoji", _EMOJI), ("elongation", _ELONG)):
        out[f"pct_with_{name}"] = float(text.map(lambda s: bool(rx.search(s))).mean() * 100) if n else 0.0
    out["pct_non_ascii"] = float(text.map(lambda s: not s.isascii()).mean() * 100) if n else 0.0
    return out


def clean_dataset(df: pd.DataFrame, drop_empty: bool = True, dedupe: bool = True,
                  drop_conflicts: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Remove empty texts, label-conflicting duplicates and (near-)duplicates.

    Returns the cleaned frame and a log of how many rows each step removed.
    """
    log = {"rows_in": int(len(df))}
    d = df.reset_index(drop=True).copy()
    d["_key"] = d["text"].map(normalize_key)

    if drop_empty:
        mask = d["_key"] == ""
        log["removed_empty"] = int(mask.sum())
        d = d[~mask]
    if drop_conflicts:
        nuniq = d.groupby("_key")["label"].transform("nunique")
        log["removed_label_conflicts"] = int((nuniq > 1).sum())
        d = d[nuniq == 1]
    if dedupe:
        mask = d.duplicated("_key", keep="first")
        log["removed_duplicates"] = int(mask.sum())
        d = d[~mask]

    d = d.drop(columns="_key").reset_index(drop=True)
    log["rows_out"] = int(len(d))
    return d, log


def format_audit(a: Dict) -> str:
    """Human-readable markdown bullet list."""
    lines = [
        f"- rows: {a['n_rows']} (empty after normalisation: {a['n_empty_text']})",
        f"- labels: {a['label_counts']} (positive rate {a['positive_rate']:.3f}, "
        f"imbalance ratio {a['imbalance_ratio']:.2f})",
        f"- exact duplicate rows: {a['exact_duplicate_rows']}; redundant rows after "
        f"normalisation: {a['redundant_rows_after_normalisation']}",
        f"- rows in label-conflicting duplicate groups: {a['rows_in_label_conflicts']} "
        f"({a['label_conflict_groups']} groups)",
        f"- tokens per text: mean {a['tokens_mean']:.1f}, median {a['tokens_median']:.0f}, "
        f"p95 {a['tokens_p95']:.0f}",
        f"- noise: URL {a['pct_with_url']:.1f}%, mention {a['pct_with_mention']:.1f}%, "
        f"hashtag {a['pct_with_hashtag']:.1f}%, emoji {a['pct_with_emoji']:.1f}%, "
        f"elongation {a['pct_with_elongation']:.1f}%, non-ASCII {a['pct_non_ascii']:.1f}%",
    ]
    return "\n".join(lines)
