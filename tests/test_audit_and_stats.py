import math

import pandas as pd

from textprep_bench.audit import audit_dataset, clean_dataset, normalize_key
from textprep_bench.data import load_csv, make_synthetic
from textprep_bench.stats import corrected_resampled_ttest, holm_adjust


def _frame():
    return pd.DataFrame({
        "text": ["I am sad", "i am SAD!", "so happy", "so happy", "  ", "same text", "same text"],
        "label": [1, 1, 0, 0, 1, 0, 1],
    })


def test_normalize_key():
    assert normalize_key("I am SAD! https://t.co/x @bob") == "i am sad"


def test_audit_counts():
    a = audit_dataset(_frame())
    assert a["n_rows"] == 7
    assert a["n_empty_text"] == 1
    assert a["exact_duplicate_rows"] == 1
    assert a["label_conflict_groups"] == 1
    assert a["rows_in_label_conflicts"] == 2


def test_clean_removes_empty_conflicts_and_duplicates():
    cleaned, log = clean_dataset(_frame())
    assert log["removed_empty"] == 1
    assert log["removed_label_conflicts"] == 2
    assert log["removed_duplicates"] == 2
    assert sorted(cleaned["text"]) == ["I am sad", "so happy"]
    assert log["rows_out"] == 2


def test_load_csv_binary_and_positive_label(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"t": ["a", "b", "c"], "y": ["dep", "ctl", "dep"]}).to_csv(p, index=False)
    df = load_csv(p, "t", "y", positive_label="dep")
    assert df["label"].tolist() == [1, 0, 1]
    try:
        load_csv(p, "t", "y")
    except ValueError:
        pass
    else:
        raise AssertionError("non-binary labels need positive_label")


def test_synthetic_is_seeded_and_dirty():
    a, b = make_synthetic(300, seed=1), make_synthetic(300, seed=1)
    assert a.equals(b)
    audit = audit_dataset(a)
    assert audit["exact_duplicate_rows"] > 0 and audit["n_empty_text"] > 0


def test_corrected_ttest_is_more_conservative_than_plain():
    diffs = [0.02, 0.03, 0.01, 0.025, 0.015, 0.03, 0.02, 0.01, 0.02, 0.03]
    r = corrected_resampled_ttest(diffs, 0.25)
    from scipy import stats
    plain_p = stats.ttest_1samp(diffs, 0).pvalue
    assert r["p"] > plain_p
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]


def test_ttest_zero_variance():
    assert corrected_resampled_ttest([0.0] * 5, 0.25)["p"] == 1.0
    assert corrected_resampled_ttest([0.1] * 5, 0.25)["p"] == 0.0


def test_holm():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))
    assert math.isclose(adj[0], 0.03) and math.isclose(adj[2], 0.06) and math.isclose(adj[1], 0.06)
    assert max(adj) <= 1.0
