"""Ablation experiments over preprocessing choices with paired, repeated CV."""
from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .audit import audit_dataset, clean_dataset, normalize_key
from .data import dataframe_sha256
from .preprocess import PreprocessConfig, preprocess_many
from .stats import corrected_resampled_ttest, holm_adjust

METRICS = ("accuracy", "macro_f1", "f1_pos", "roc_auc")

# Minimal processing (only whitespace/HTML normalisation) and a "full" pipeline.
RAW = PreprocessConfig()
FULL = PreprocessConfig(
    lowercase=True, urls="token", mentions="token", hashtags="strip", emoji="token",
    elongation=True, strip_punctuation=True, stopwords=True, keep_negations=True,
    negation_marking=True, stemming="light", min_token_len=2,
)

# Each ablation "step" is a group of config fields switched together.
STEP_FIELDS: Dict[str, Tuple[str, ...]] = {
    "lowercase": ("lowercase",),
    "urls": ("urls",),
    "mentions": ("mentions",),
    "hashtags": ("hashtags",),
    "emoji": ("emoji",),
    "elongation": ("elongation",),
    "strip_punctuation": ("strip_punctuation",),
    "stopwords": ("stopwords", "keep_negations"),
    "negation_marking": ("negation_marking",),
    "stemming": ("stemming",),
    "min_token_len": ("min_token_len",),
}

# Extra add-one variants that are alternatives to the FULL setting of a step.
EXTRA_ADD_ONE: Dict[str, Dict] = {
    "urls_remove": {"urls": "remove"},
    "emoji_remove": {"emoji": "remove"},
    "stopwords_incl_negations": {"stopwords": True, "keep_negations": False},
}


def _switch(base: PreprocessConfig, source: PreprocessConfig, fields) -> PreprocessConfig:
    return replace(base, **{f: getattr(source, f) for f in fields})


def design_configs(design: str = "both") -> Tuple[Dict[str, PreprocessConfig], Dict[str, str]]:
    """Return (configs, baselines). ``baselines[name]`` is the config a given
    variant is compared against. Baselines themselves are not in ``baselines``."""
    if design not in {"add-one", "drop-one", "both"}:
        raise ValueError("design must be add-one, drop-one or both")
    configs: Dict[str, PreprocessConfig] = {"RAW": RAW}
    baselines: Dict[str, str] = {}
    if design in {"add-one", "both"}:
        for step, fields in STEP_FIELDS.items():
            name = f"+{step}"
            configs[name] = _switch(RAW, FULL, fields)
            baselines[name] = "RAW"
        for name, changes in EXTRA_ADD_ONE.items():
            configs[f"+{name}"] = replace(RAW, **changes)
            baselines[f"+{name}"] = "RAW"
    configs["FULL"] = FULL
    baselines["FULL"] = "RAW"
    if design in {"drop-one", "both"}:
        for step, fields in STEP_FIELDS.items():
            name = f"-{step}"
            configs[name] = _switch(FULL, RAW, fields)
            baselines[name] = "FULL"
    return configs, baselines


def _make_pipeline(model: str, seed: int) -> Pipeline:
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if model == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif model == "svm":
        clf = LinearSVC(random_state=seed)
    elif model == "nb":
        clf = MultinomialNB()
    else:
        raise ValueError(f"unknown model {model!r} (choose logreg, svm, nb)")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def make_splits(y, n_splits: int, n_repeats: int, seed: int):
    """Splits are created once and reused for every pipeline => paired scores."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), y))


def _evaluate(name: str, texts: np.ndarray, y: np.ndarray, model: str, splits, seed: int,
              n_splits: int) -> List[dict]:
    rows = []
    for i, (tr, te) in enumerate(splits):
        pipe = _make_pipeline(model, seed)
        pipe.fit(texts[tr], y[tr])
        pred = pipe.predict(texts[te])
        score = (pipe.decision_function(texts[te]) if hasattr(pipe, "decision_function")
                 else pipe.predict_proba(texts[te])[:, 1])
        rows.append({
            "config": name, "model": model, "split": i,
            "repeat": i // n_splits, "fold": i % n_splits,
            "accuracy": accuracy_score(y[te], pred),
            "macro_f1": f1_score(y[te], pred, average="macro"),
            "f1_pos": f1_score(y[te], pred, pos_label=1),
            "roc_auc": roc_auc_score(y[te], score),
            "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
        })
    return rows


def compare_to_baselines(folds: pd.DataFrame, baselines: Dict[str, str], metric: str,
                         n_splits: int) -> pd.DataFrame:
    ratio = 1.0 / (n_splits - 1)
    out = []
    for model, g in folds.groupby("model", sort=False):
        piv = g.pivot(index="split", columns="config", values=metric)
        rows = []
        for cfg, base in baselines.items():
            d = (piv[cfg] - piv[base]).to_numpy()
            r = corrected_resampled_ttest(d, ratio)
            rows.append({"model": model, "config": cfg, "baseline": base, "metric": metric, **r})
        for row, p_adj in zip(rows, holm_adjust([r["p"] for r in rows])):
            row["p_holm"] = p_adj
        out.extend(rows)
    return pd.DataFrame(out)


def _git_commit() -> Optional[str]:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=5)
        return res.stdout.strip() or None if res.returncode == 0 else None
    except Exception:
        return None


def collect_metadata(extra: Optional[dict] = None) -> dict:
    meta = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        "git_commit": _git_commit(),
    }
    meta.update(extra or {})
    return meta


def run_experiment(df: pd.DataFrame, design: str = "both", models=("logreg",),
                   n_splits: int = 5, n_repeats: int = 3, seed: int = 42,
                   metric: str = "macro_f1", clean: bool = True, n_jobs: int = 1,
                   data_source: str = "") -> dict:
    """Run the ablation and return folds / summary / comparisons / metadata / audit."""
    audit_before = audit_dataset(df)
    clean_log = None
    if clean:
        df, clean_log = clean_dataset(df)
    df = df.reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)

    configs, baselines = design_configs(design)
    splits = make_splits(y, n_splits, n_repeats, seed)
    processed = {name: np.array(preprocess_many(df["text"], cfg), dtype=object)
                 for name, cfg in configs.items()}

    tasks = [(name, model) for name in configs for model in models]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate)(name, processed[name], y, model, splits, seed, n_splits)
        for name, model in tasks
    )
    folds = pd.DataFrame([row for chunk in results for row in chunk])
    order = {name: i for i, name in enumerate(configs)}
    folds = folds.sort_values(["model", "config", "split"], key=lambda s: s.map(order) if s.name == "config" else s).reset_index(drop=True)

    agg = {m: ["mean", "std"] for m in (*METRICS, "n_features")}
    summary = folds.groupby(["model", "config"], sort=False).agg(agg)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    comparisons = compare_to_baselines(folds, baselines, metric, n_splits)
    metadata = collect_metadata({
        "data_source": data_source, "data_sha256": dataframe_sha256(df),
        "n_rows_used": int(len(df)), "design": design, "models": list(models),
        "n_splits": n_splits, "n_repeats": n_repeats, "seed": seed, "primary_metric": metric,
        "cleaning": clean_log,
        "configs": {n: c.to_dict() for n, c in configs.items()},
    })
    return {"folds": folds, "summary": summary, "comparisons": comparisons,
            "metadata": metadata, "audit": audit_before, "baselines": baselines}


def run_leakage_experiment(df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3,
                           seed: int = 42, model: str = "logreg") -> dict:
    """Quantify how much duplicates inflate cross-validation scores.

    The same FULL pipeline is evaluated on the raw data (duplicates left in) and
    on the cleaned data. ``leak_fraction`` is the share of test rows whose
    normalised text also appears in the training part of the split.
    """
    out = {}
    cleaned, _ = clean_dataset(df)
    for name, d in (("raw_with_duplicates", df.reset_index(drop=True)),
                    ("cleaned", cleaned)):
        y = d["label"].to_numpy().astype(int)
        texts = np.array(preprocess_many(d["text"], FULL), dtype=object)
        keys = d["text"].map(normalize_key).tolist()
        leak, f1s = [], []
        for tr, te in make_splits(y, n_splits, n_repeats, seed):
            train_keys = {keys[i] for i in tr}
            leak.append(float(np.mean([keys[i] in train_keys for i in te])))
            pipe = _make_pipeline(model, seed).fit(texts[tr], y[tr])
            f1s.append(f1_score(y[te], pipe.predict(texts[te]), average="macro"))
        out[name] = {"n_rows": int(len(d)), "leak_fraction": float(np.mean(leak)),
                     "macro_f1": float(np.mean(f1s))}
    out["macro_f1_inflation"] = (out["raw_with_duplicates"]["macro_f1"]
                                 - out["cleaned"]["macro_f1"])
    return out
