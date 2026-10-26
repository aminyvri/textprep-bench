"""Dataset loading, fingerprinting and a SYNTHETIC demo-data generator.

The synthetic generator exists only so that the code runs end to end without
downloading anything (tests, CI, first-run demo). Results obtained on it say
nothing about real-world behaviour and must never be reported as findings.
"""
from __future__ import annotations

import hashlib
import random
import string
from pathlib import Path

import pandas as pd


def load_csv(path, text_col="text", label_col="label", positive_label=None, sep=","):
    """Load a CSV into a DataFrame with columns ``text`` (str) and ``label`` (0/1)."""
    df = pd.read_csv(path, sep=sep)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found; available: {list(df.columns)}")
    text = df[text_col].fillna("").astype(str)
    raw = df[label_col]
    if positive_label is None:
        values = set(raw.dropna().unique().tolist())
        if not values <= {0, 1}:
            raise ValueError(
                f"labels {sorted(map(str, values))} are not binary 0/1; "
                "pass positive_label to say which class is positive"
            )
        label = raw.astype(int)
    else:
        label = (raw.astype(str) == str(positive_label)).astype(int)
    return pd.DataFrame({"text": text, "label": label}).reset_index(drop=True)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for text, label in zip(df["text"], df["label"]):
        h.update(f"{int(label)}\t{text}\n".encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------
# Synthetic data (demo / tests only)
# --------------------------------------------------------------------------
_NEG = ["sad", "tired", "empty", "lonely", "drained", "numb", "anxious",
        "exhausted", "miserable", "down"]
_POS = ["happy", "excited", "grateful", "relaxed", "proud", "energetic",
        "cheerful", "calm", "hopeful", "fine"]
_TOPICS = ["work", "working", "works", "school", "friends", "friend", "family",
           "weekend", "coffee", "movie", "movies", "dinner", "music", "project",
           "projects", "game", "games", "weather", "morning", "night"]
_EMO_NEG = ["\U0001F61E", "\U0001F614", "\U0001F622"]
_EMO_POS = ["\U0001F60A", "\U0001F604", "\U0001F642"]


def _make_text(rng: random.Random, y: int) -> str:
    parts = []
    for _ in range(rng.choice([1, 1, 2])):
        negated = rng.random() < 0.25
        pool = (_POS if y else _NEG) if negated else (_NEG if y else _POS)
        w = rng.choice(pool)
        if rng.random() < 0.2:
            w = w + w[-1] * rng.randint(2, 4)  # elongation: "sad" -> "saddd"
        neg = "not " if negated else ""
        topic = rng.choice(_TOPICS)
        parts.append(rng.choice([
            f"i am {neg}{w} about {topic}",
            f"{neg}{w} again after {topic}",
            f"honestly i am {neg}{w} because of {topic}",
        ]))
    text = ". ".join(parts)
    r = rng.random()
    if r < 0.15:
        text = text.upper()
    elif r < 0.5:
        text = text.capitalize()
    if rng.random() < 0.2:
        text += " https://t.co/" + "".join(rng.choices(string.ascii_letters + string.digits, k=6))
    if rng.random() < 0.25:
        text += f" @user{rng.randint(1, 99)}"
    if rng.random() < 0.25:
        text += " #" + rng.choice(_TOPICS)
    if rng.random() < 0.4:
        same = rng.random() < 0.85
        text += " " + rng.choice(_EMO_NEG if (y == 1) == same else _EMO_POS)
    return text + rng.choice(["", "!", "...", "!!!", "."])


def make_synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic, deliberately noisy corpus: URLs, mentions, hashtags, emoji,
    elongation, negation, label noise, exact/near duplicates, label conflicts
    and empty rows. NOT real data."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        y = int(rng.random() < 0.45)
        label = 1 - y if rng.random() < 0.04 else y  # 4% label noise
        rows.append((_make_text(rng, y), label))

    extra = []
    for text, label in rng.sample(rows, int(0.08 * n)):          # exact duplicates
        extra.append((text, label))
    for text, label in rng.sample(rows, int(0.03 * n)):          # near duplicates
        extra.append(("  " + text.lower().replace(".", ""), label))
    for text, label in rng.sample(rows, int(0.015 * n)):         # label conflicts
        extra.append((text, 1 - label))
    for _ in range(int(0.01 * n)):                               # empty rows
        extra.append(("", rng.randint(0, 1)))

    rows.extend(extra)
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])
