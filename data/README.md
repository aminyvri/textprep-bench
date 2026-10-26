# Data

No data is stored in this repository, and real datasets must never be committed
(`.gitignore` blocks `data/*` and `*.csv`).

## Expected format

A CSV with one text column and one binary label column:

| text | label |
|------|-------|
| "i can't sleep again..." | 1 |
| "great hike today!" | 0 |

```bash
python -m textprep_bench audit --data data/my_dataset.csv --text-col text --label-col label
python -m textprep_bench run   --data data/my_dataset.csv --text-col text --label-col label \
       --positive-label depression --out results/my_dataset
```

If your labels are not already 0/1, pass `--positive-label VALUE`.

## Where to get data

Use only datasets whose licence / data-use terms you have read and accepted.
Datasets about mental health are sensitive, so:

- Prefer openly available corpora (for example Reddit-based stress/depression
  corpora such as Dreaddit; check the current licence and terms before use).
- Some shared-task corpora (for example CLPsych, eRisk) and clinical interview
  corpora require a signed data-use agreement. Never redistribute them.
- Do not try to identify or contact authors of posts. Do not publish raw posts.
