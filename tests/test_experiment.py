import numpy as np

from textprep_bench.data import make_synthetic
from textprep_bench.experiment import (FULL, RAW, design_configs, make_splits,
                                       run_experiment, run_leakage_experiment)


def test_design_configs_and_baselines():
    configs, baselines = design_configs("both")
    assert configs["RAW"] == RAW and configs["FULL"] == FULL
    assert baselines["+lowercase"] == "RAW" and baselines["-lowercase"] == "FULL"
    assert configs["-negation_marking"].negation_marking is False
    assert configs["+negation_marking"].negation_marking is True
    assert "RAW" not in baselines
    add, _ = design_configs("add-one")
    assert not any(n.startswith("-") for n in add)


def test_splits_are_reproducible_and_paired():
    y = np.array([0, 1] * 50)
    a, b = make_splits(y, 5, 2, 7), make_splits(y, 5, 2, 7)
    assert all((x[1] == z[1]).all() for x, z in zip(a, b))
    assert len(a) == 10


def test_small_run_is_reproducible():
    df = make_synthetic(400, seed=3)
    kw = dict(design="add-one", models=("logreg",), n_splits=3, n_repeats=2, seed=1)
    r1, r2 = run_experiment(df, **kw), run_experiment(df, **kw)
    assert r1["folds"].equals(r2["folds"])
    assert set(r1["comparisons"]["baseline"]) == {"RAW"}
    n_cfg = len(design_configs("add-one")[0])
    assert len(r1["folds"]) == n_cfg * 6
    for col in ("p", "p_holm", "ci_low", "ci_high"):
        assert r1["comparisons"][col].between(-1, 1).all()
    assert r1["metadata"]["data_sha256"] and r1["metadata"]["seed"] == 1


def test_leakage_experiment_detects_duplicates():
    out = run_leakage_experiment(make_synthetic(500, seed=2), n_splits=3, n_repeats=2)
    assert out["raw_with_duplicates"]["leak_fraction"] > out["cleaned"]["leak_fraction"]
    assert out["cleaned"]["leak_fraction"] == 0.0
