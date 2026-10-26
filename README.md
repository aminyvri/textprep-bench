# textprep-bench

**A reproducible ablation benchmark of text-preprocessing choices for social-media text classification (e.g. depression-related language detection).**

Papers on mental-health NLP often differ in *how* they clean text (lower-casing, stop-word removal, emoji handling, negation, duplicates...) and rarely report it completely. This project measures, under one controlled and fully documented protocol, **how much each preprocessing decision actually changes classifier performance**, and whether the change is statistically distinguishable from noise.

> ⚠️ **Research / educational tool.** It evaluates preprocessing choices on labelled public datasets. It is **not** a diagnostic tool and must not be used to assess real individuals.

---

## What it does

| Component | Purpose |
|-----------|---------|
| `preprocess.py` | 11 independent, documented preprocessing steps described by one hashable `PreprocessConfig` (so every pipeline can be reported exactly) |
| `audit.py` | Data-quality audit: empty texts, exact/near duplicates, **label conflicts**, class imbalance, noise statistics (URLs, mentions, emoji, elongation...) and a cleaning routine |
| `experiment.py` | **add-one** (RAW + one step) and **drop-one** (FULL minus one step) ablations with TF-IDF + Logistic Regression / Linear SVM / Naive Bayes |
| `stats.py` | Nadeau–Bengio **corrected resampled t-test** + **Holm** correction for the many comparisons |
| `report.py` | `folds.csv`, `summary.csv`, `comparisons.csv`, `metadata.json`, `audit.json`, `report.md`, forest plot |
| leakage experiment | Measures how much duplicates between train and test folds change the CV score |

### Preprocessing steps

lower-casing · URLs (keep/remove/token) · @mentions (keep/remove/token) · #hashtags (keep/strip/remove) · emoji (keep/remove/token) · elongation reduction (`sooooo → soo`) · punctuation stripping · stop-word removal (with or without protecting negations) · negation marking (`not happy → not not_happy`) · stemming (light built-in, or Porter via NLTK) · minimum token length.

---

## Quick start

```bash
git clone https://github.com/<username>/textprep-bench.git
cd textprep-bench
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                                            # unit tests
python -m textprep_bench demo --out results/demo     # ~15 s, SYNTHETIC data
```

### On a real dataset

```bash
# 1) look at your data first
python -m textprep_bench audit --data data/my_dataset.csv --text-col text --label-col label

# 2) run the ablation (5-fold x 3 repeats, 3 models)
python -m textprep_bench run --data data/my_dataset.csv --text-col text --label-col label \
    --positive-label depression --models logreg svm nb --folds 5 --repeats 3 \
    --seed 42 --n-jobs 4 --out results/my_dataset

# 3) how much do duplicates distort cross-validation on this dataset?
python -m textprep_bench leakage --data data/my_dataset.csv --text-col text --label-col label
```

See [`data/README.md`](data/README.md) for the expected CSV format and notes on obtaining datasets responsibly.

---

## Design decisions (why you can trust the numbers)

1. **No leakage from preprocessing.** All steps are stateless, so applying them to the whole corpus before cross-validation learns nothing from test data. TF-IDF is fitted *inside* each training fold.
2. **Paired comparisons.** The CV splits are generated once per seed and reused by every pipeline, so differences are measured on identical folds.
3. **Correct statistics for repeated CV.** Fold scores are not independent; a plain t-test is over-confident. We use the corrected resampled t-test (Nadeau & Bengio, 2003) and Holm adjustment across all comparisons.
4. **Data quality is part of the experiment.** Exact and near-duplicates are removed *before* splitting, and rows with conflicting labels are dropped (switch off with `--no-clean`). The `leakage` command quantifies the effect of not doing so.
5. **Full provenance.** `metadata.json` stores the dataset SHA-256, seeds, library versions, git commit and every pipeline configuration.
6. **Deterministic.** The same data and seed give identical `folds.csv` (covered by a test).

---

## Outputs

```
results/<run>/
├── report.md          # human-readable report with tables and plot
├── effects.png        # forest plot: paired difference vs baseline with 95% CI
├── summary.csv        # mean/std of accuracy, macro-F1, F1(positive), ROC-AUC, #features
├── comparisons.csv    # mean diff, CI, p, Holm-adjusted p per pipeline
├── folds.csv          # every fold score (for your own analysis)
├── metadata.json      # provenance
└── audit.json         # data-quality audit
```

A **synthetic** example report is in [`examples/synthetic_demo/`](examples/synthetic_demo/report.md). It only demonstrates the output format; synthetic numbers carry no scientific meaning.

---

## Results

_Results on real datasets will be added here after running the benchmark (see **Quick start**). No performance claims are made for the synthetic demo._

| Dataset | Models | Main finding | Report |
|---------|--------|--------------|--------|
| _to be added_ | | | |

---

## Project structure

```
textprep-bench/
├── textprep_bench/
│   ├── preprocess.py   # PreprocessConfig + preprocess()
│   ├── audit.py        # audit_dataset(), clean_dataset()
│   ├── data.py         # load_csv(), fingerprints, synthetic demo generator
│   ├── experiment.py   # ablation designs, paired CV, leakage experiment
│   ├── stats.py        # corrected resampled t-test, Holm
│   ├── report.py       # CSV / JSON / Markdown / plot
│   └── cli.py          # audit | run | leakage | demo
├── tests/              # preprocessing, audit, statistics, reproducibility
├── examples/synthetic_demo/
├── data/README.md
└── .github/workflows/ci.yml
```

### Adding your own preprocessing step

1. Add a field to `PreprocessConfig` and implement it in `preprocess()`.
2. Register it in `STEP_FIELDS` in `experiment.py` (the add-one and drop-one designs pick it up automatically).
3. Add a unit test.

---

## Limitations

- Bag-of-words (TF-IDF, uni+bigrams) models only; transformer models may react differently to preprocessing.
- The built-in `light` stemmer is a crude approximation; use `stemming="porter"` (NLTK) for reportable results.
- Single-dataset results are exploratory and may not transfer across platforms or languages (English only).
- Binary classification only. Step effects can interact; the add-one/drop-one designs show main effects, not full factorial interactions.
- Public mental-health datasets have known label noise and sampling biases (e.g. subreddit membership as a label proxy).

## Responsible use

Mental-health text is sensitive. Use only data you are permitted to use, never commit raw posts, never attempt to identify authors, and do not deploy models trained this way to make judgements about individuals.

## References

- Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization Error.* Machine Learning, 52, 239–281.
- Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics, 6, 65–70.

## Author

Your Name · [GitHub](https://github.com/<username>) · [ORCID](https://orcid.org/)

Motivated by the author's ongoing research on data preprocessing and reproducibility in depression-detection studies.

## License

MIT, see [LICENSE](LICENSE).
