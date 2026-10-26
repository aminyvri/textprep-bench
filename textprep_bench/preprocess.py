"""Configurable, stateless text preprocessing for noisy social-media text.

Every step is an explicit, documented option so that a pipeline can be
described completely by one ``PreprocessConfig`` (and therefore reported,
shared and reproduced).

Fixed order of operations
-------------------------
1. HTML unescape, apostrophe and whitespace normalisation
2. lower-casing
3. URLs, @mentions, #hashtags, emoji (keep / remove / replace by a token)
4. reduction of character elongation ("sooooo" -> "soo")
5. tokenisation (words, mentions, hashtags, punctuation and emoji are tokens)
6. stop-word removal (optionally protecting negations)
7. negation marking ("not happy" -> "not not_happy")
8. punctuation stripping
9. stemming
10. minimum token length

The functions are stateless (nothing is learned from data), so applying them
to the full corpus before cross-validation cannot leak information.
"""
from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable, Iterable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+"
)
_ELONG_RE = re.compile(r"(\w)\1{2,}")
_TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|\w+(?:'\w+)?|@\w+|#\w+|[^\w\s]")
_WORD_RE = re.compile(r"\w+(?:'\w+)?")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

NEGATIONS = frozenset(
    {"no", "not", "never", "none", "nobody", "nothing", "nowhere",
     "neither", "nor", "cannot", "without"}
)
_STOP_KEEP_NEG = frozenset(ENGLISH_STOP_WORDS - NEGATIONS)
NEGATION_SCOPE = 3  # number of following word tokens that get marked

_CHOICES = {
    "urls": {"keep", "remove", "token"},
    "mentions": {"keep", "remove", "token"},
    "hashtags": {"keep", "strip", "remove"},
    "emoji": {"keep", "remove", "token"},
    "stemming": {"none", "light", "porter"},
}


@dataclass(frozen=True)
class PreprocessConfig:
    """A complete, hashable description of a preprocessing pipeline."""

    lowercase: bool = False
    urls: str = "keep"
    mentions: str = "keep"
    hashtags: str = "keep"
    emoji: str = "keep"
    elongation: bool = False
    strip_punctuation: bool = False
    stopwords: bool = False
    keep_negations: bool = True
    negation_marking: bool = False
    stemming: str = "none"
    min_token_len: int = 1

    def __post_init__(self) -> None:
        for field, allowed in _CHOICES.items():
            value = getattr(self, field)
            if value not in allowed:
                raise ValueError(f"{field}={value!r} not in {sorted(allowed)}")
        if self.min_token_len < 1:
            raise ValueError("min_token_len must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


def light_stem(word: str) -> str:
    """A deliberately crude suffix stripper (no external data needed).

    Use ``stemming='porter'`` (requires NLTK) for reportable experiments.
    """
    if len(word) <= 3:
        return word
    for suffix, repl in (("ingly", ""), ("edly", ""), ("ies", "y"), ("ied", "y"),
                         ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) + len(repl) >= 3:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                continue
            return word[: -len(suffix)] + repl
    return word


@lru_cache(maxsize=None)
def _get_stemmer(name: str) -> Callable[[str], str]:
    if name == "light":
        return light_stem
    if name == "porter":
        try:
            from nltk.stem import PorterStemmer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("stemming='porter' requires NLTK: pip install nltk") from exc
        return PorterStemmer().stem
    raise ValueError(name)


def _apply(pattern: re.Pattern, text: str, mode: str, token: str) -> str:
    if mode == "keep":
        return text
    return pattern.sub(" " if mode == "remove" else token, text)


def _is_negation(low: str) -> bool:
    return low in NEGATIONS or low.endswith("n't")


def _is_punct(tok: str) -> bool:
    return _PUNCT_RE.fullmatch(tok) is not None and _EMOJI_RE.fullmatch(tok) is None


def _stem_token(tok: str, stem: Callable[[str], str]) -> str:
    prefix, core = ("not_", tok[4:]) if tok.startswith("not_") else ("", tok)
    if not core.isalpha() or core.startswith("xx"):
        return tok
    return prefix + stem(core)


def preprocess(text, cfg: PreprocessConfig) -> str:
    """Return the preprocessed text as a space-separated token string."""
    if text is None or (isinstance(text, float) and text != text):
        text = ""
    t = html.unescape(str(text)).replace("\u2019", "'")
    t = _WS_RE.sub(" ", t).strip()
    if cfg.lowercase:
        t = t.lower()

    t = _apply(_URL_RE, t, cfg.urls, " xxurl ")
    t = _apply(_MENTION_RE, t, cfg.mentions, " xxuser ")
    if cfg.hashtags == "strip":
        t = _HASHTAG_RE.sub(r" \1 ", t)
    elif cfg.hashtags == "remove":
        t = _HASHTAG_RE.sub(" ", t)
    t = _apply(_EMOJI_RE, t, cfg.emoji, " xxemoji ")
    if cfg.elongation:
        t = _ELONG_RE.sub(r"\1\1", t)

    tokens: List[str] = _TOKEN_RE.findall(t)

    if cfg.stopwords:
        stop = _STOP_KEEP_NEG if cfg.keep_negations else ENGLISH_STOP_WORDS
        tokens = [tok for tok in tokens if tok.lower() not in stop]

    if cfg.negation_marking:
        marked, remaining = [], 0
        for tok in tokens:
            if _is_negation(tok.lower()):
                marked.append(tok)
                remaining = NEGATION_SCOPE
            elif _WORD_RE.fullmatch(tok):
                if remaining > 0:
                    marked.append("not_" + tok)
                    remaining -= 1
                else:
                    marked.append(tok)
            else:  # punctuation, emoji, mention or hashtag ends the scope
                remaining = 0
                marked.append(tok)
        tokens = marked

    if cfg.strip_punctuation:
        tokens = [tok for tok in tokens if not _is_punct(tok)]

    if cfg.stemming != "none":
        stem = _get_stemmer(cfg.stemming)
        tokens = [_stem_token(tok, stem) for tok in tokens]

    if cfg.min_token_len > 1:
        tokens = [
            tok for tok in tokens
            if not _WORD_RE.fullmatch(tok)
            or len(tok[4:] if tok.startswith("not_") else tok) >= cfg.min_token_len
        ]
    return " ".join(tokens)


def preprocess_many(texts: Iterable, cfg: PreprocessConfig) -> List[str]:
    return [preprocess(t, cfg) for t in texts]
