"""Write machine-readable results and a human-readable Markdown report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .audit import format_audit  # noqa: E402


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _forest_plot(comp: pd.DataFrame, path: Path, metric: str) -> None:
    c = comp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(c) + 1.5))
    y = range(len(c))
    ax.errorbar(c["mean_diff"], list(y),
                xerr=[c["mean_diff"] - c["ci_low"], c["ci_high"] - c["mean_diff"]],
                fmt="o", color="black", ecolor="gray", capsize=2)
    ax.axvline(0, color="red", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{n} (vs {b})" for n, b in zip(c["config"], c["baseline"])])
    ax.set_xlabel(f"paired difference in {metric} (95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(bundle: dict, out_dir, title: str = "Preprocessing ablation report",
                  banner: str = "") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle["folds"].to_csv(out / "folds.csv", index=False)
    bundle["summary"].to_csv(out / "summary.csv", index=False)
    bundle["comparisons"].to_csv(out / "comparisons.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(bundle["metadata"], indent=2), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(bundle["audit"], indent=2), encoding="utf-8")

    meta, comp = bundle["metadata"], bundle["comparisons"]
    metric = meta["primary_metric"]
    first_model = comp["model"].iloc[0]
    _forest_plot(comp[comp["model"] == first_model], out / "effects.png", metric)

    summ = bundle["summary"][["model", "config", "macro_f1_mean", "macro_f1_std",
                              "roc_auc_mean", "n_features_mean"]]
    comp_tbl = comp[["model", "config", "baseline", "mean_diff", "ci_low", "ci_high",
                     "p", "p_holm"]]
    md = [f"# {title}", ""]
    if banner:
        md += [f"> **{banner}**", ""]
    md += [
        "## Setup",
        f"- data: `{meta['data_source']}` (sha256 `{meta['data_sha256'][:16]}...`), "
        f"{meta['n_rows_used']} rows after cleaning",
        f"- design: {meta['design']}; models: {', '.join(meta['models'])}; "
        f"{meta['n_repeats']}x{meta['n_splits']}-fold stratified CV; seed {meta['seed']}",
        f"- primary metric: {metric}; comparisons use the Nadeau-Bengio corrected "
        "resampled t-test with Holm adjustment",
        f"- scikit-learn {meta['scikit_learn']}, python {meta['python']}, "
        f"commit {meta['git_commit']}",
        "", "## Data audit (before cleaning)", format_audit(bundle["audit"]), "",
        "## Scores per pipeline", df_to_markdown(summ), "",
        f"## Effect of each step on {metric} (variant minus baseline)",
        df_to_markdown(comp_tbl), "",
        "![effects](effects.png)", "",
        "## How to read this",
        "- `+step`: RAW plus that single step. `-step`: FULL pipeline without that step.",
        "- Differences whose CI includes 0 (or p_holm >= 0.05) are not evidence of an effect.",
        "- One dataset and a handful of models: treat findings as exploratory.",
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return out / "report.md"
