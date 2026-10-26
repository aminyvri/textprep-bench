"""Statistics for comparing preprocessing pipelines on repeated cross-validation.

Fold scores from repeated k-fold CV are *not* independent, so a plain paired
t-test or Wilcoxon test is over-confident. We use the corrected resampled
t-test of Nadeau & Bengio (2003) and Holm's step-down adjustment for the
many pipeline comparisons.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from scipy import stats


def corrected_resampled_ttest(diffs: Sequence[float], test_train_ratio: float) -> Dict[str, float]:
    """Nadeau-Bengio corrected t-test on paired per-split score differences.

    ``test_train_ratio`` is n_test / n_train of a single split
    (= 1 / (k - 1) for k-fold cross-validation).
    """
    d = np.asarray(diffs, dtype=float)
    k = d.size
    if k < 2:
        raise ValueError("need at least two paired differences")
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    se = math.sqrt((1.0 / k + test_train_ratio) * var)
    if se == 0.0:
        p = 1.0 if mean == 0.0 else 0.0
        return {"mean_diff": mean, "se": 0.0, "t": 0.0 if mean == 0.0 else math.inf,
                "p": p, "ci_low": mean, "ci_high": mean}
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), k - 1))
    half = float(stats.t.ppf(0.975, k - 1) * se)
    return {"mean_diff": mean, "se": se, "t": t, "p": p,
            "ci_low": mean - half, "ci_high": mean + half}


def holm_adjust(pvalues: Sequence[float]) -> list:
    """Holm-Bonferroni step-down adjusted p-values (original order)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()
