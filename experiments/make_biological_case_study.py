"""Generate a cell-line-level biological case study from ORCA predictions.

Example:
  python experiments/make_biological_case_study.py \
    --predictions_glob "runs/gdsc_morgan_final_orca_family/orca_ensemble_split_*/predictions_test_orca_ensemble.csv" \
    --prediction_col y_pred_orca_ensemble \
    --out_dir runs/gdsc_morgan_final_orca_family/case_study_all_splits
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    if np.std(y_true[mask]) < 1e-12 or np.std(y_pred[mask]) < 1e-12:
        return float("nan")
    return float(fn(y_true[mask], y_pred[mask])[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "n_drugs": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": safe_corr(pearsonr, y_true, y_pred),
        "spearman": safe_corr(spearmanr, y_true, y_pred),
    }


def load_drug_mapping(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    meta = pd.read_csv(path)
    if "SMILES" not in meta.columns or "CompoundName" not in meta.columns:
        raise ValueError("Drug metadata must contain SMILES and CompoundName columns.")
    cols = [c for c in ["SMILES", "CompoundName", "CompoundID", "Synonyms", "TargetOrMechanism"] if c in meta.columns]
    return meta[cols].dropna(subset=["SMILES"]).drop_duplicates("SMILES").rename(columns={"SMILES": "drug_id"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_csv", default=None)
    parser.add_argument("--predictions_glob", default=None)
    parser.add_argument("--prediction_col", default="y_pred_orca_ensemble")
    parser.add_argument("--cell_col", default="cell_id")
    parser.add_argument("--drug_col", default="drug_id")
    parser.add_argument("--y_true_col", default="y_true")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_drugs", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--select_cell", default=None)
    parser.add_argument("--drug_metadata_csv", default=None)
    args = parser.parse_args()

    if not args.predictions_csv and not args.predictions_glob:
        raise ValueError("Provide either --predictions_csv or --predictions_glob.")

    paths = sorted(Path(p) for p in glob.glob(args.predictions_glob)) if args.predictions_glob else [Path(args.predictions_csv)]
    if not paths:
        raise ValueError("No prediction files matched the requested input.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx, path in enumerate(paths, start=1):
        part = pd.read_csv(path)
        part["source_file"] = str(path)
        part["source_index"] = idx
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    required = {args.cell_col, args.drug_col, args.y_true_col, args.prediction_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")

    work = df[[args.cell_col, args.drug_col, args.y_true_col, args.prediction_col, "source_file", "source_index"]].copy()
    work = work.rename(
        columns={
            args.cell_col: "cell_id",
            args.drug_col: "drug_id",
            args.y_true_col: "y_true",
            args.prediction_col: "y_pred",
        }
    ).dropna(subset=["cell_id", "drug_id", "y_true", "y_pred"])

    rows: List[Dict[str, object]] = []
    for cell_id, group in work.groupby("cell_id"):
        row = regression_metrics(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float))
        row["cell_id"] = cell_id
        rows.append(row)
    per_cell = pd.DataFrame(rows).sort_values(["spearman", "n_drugs"], ascending=[False, False])
    per_cell.to_csv(out_dir / "per_cell_line_metrics.csv", index=False)

    candidates = per_cell[per_cell["n_drugs"] >= args.min_drugs]
    if candidates.empty:
        candidates = per_cell

    if args.select_cell:
        selected_cell = str(args.select_cell)
        selected_metrics = per_cell[per_cell["cell_id"].astype(str) == selected_cell].iloc[0].to_dict()
    else:
        selected_metrics = candidates.iloc[0].to_dict()
        selected_cell = str(selected_metrics["cell_id"])

    selected = work[work["cell_id"].astype(str) == selected_cell].copy()
    selected["abs_error"] = (selected["y_true"] - selected["y_pred"]).abs()
    selected = selected.sort_values("y_pred", ascending=True)

    mapping = load_drug_mapping(args.drug_metadata_csv)
    if mapping is not None:
        selected = selected.merge(mapping, on="drug_id", how="left")

    top_sensitive = selected.head(args.top_k).copy()
    top_resistant = selected.tail(args.top_k).sort_values("y_pred", ascending=False).copy()
    top_sensitive.to_csv(out_dir / "selected_cell_line_top_sensitive.csv", index=False)
    top_resistant.to_csv(out_dir / "selected_cell_line_top_resistant.csv", index=False)

    summary = {
        "prediction_files": [str(p) for p in paths],
        "prediction_col": args.prediction_col,
        "selected_cell_id": selected_cell,
        "selection_rule": "highest per-cell-line Spearman among cell lines meeting min_drugs",
        "min_drugs": args.min_drugs,
        "top_k": args.top_k,
        "selected_cell_metrics": selected_metrics,
        "n_cell_lines": int(per_cell.shape[0]),
        "median_per_cell_spearman": float(np.nanmedian(per_cell["spearman"].to_numpy(float))),
        "median_per_cell_mae": float(np.nanmedian(per_cell["mae"].to_numpy(float))),
    }
    (out_dir / "case_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7.0, 4.2))
    vals = per_cell["spearman"].dropna().to_numpy(float)
    plt.hist(vals, bins=25, color="#4c78a8", edgecolor="white")
    plt.axvline(np.nanmedian(vals), color="black", linestyle="--", linewidth=1.0, label="median")
    plt.xlabel("Per-cell-line Spearman correlation")
    plt.ylabel("Number of held-out cell lines")
    plt.title("ORCA-Ensemble drug-ranking performance by held-out cell line")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "per_cell_line_spearman.pdf", dpi=300)
    plt.savefig(out_dir / "per_cell_line_spearman.png", dpi=300)
    plt.close()

    plot_df = pd.concat([top_sensitive.assign(group="Predicted sensitive"), top_resistant.assign(group="Predicted resistant")])
    label_col = "CompoundName" if "CompoundName" in plot_df.columns else "drug_id"
    labels = plot_df[label_col].fillna(plot_df["drug_id"]).astype(str).str.slice(0, 28)
    x = np.arange(plot_df.shape[0])

    plt.figure(figsize=(9.0, 4.8))
    plt.scatter(x, plot_df["y_true"], label="Observed AUC", color="#1f77b4", s=35)
    plt.scatter(x, plot_df["y_pred"], label="Predicted AUC", color="#d62728", marker="x", s=45)
    plt.axvline(len(top_sensitive) - 0.5, color="grey", linestyle=":", linewidth=1.0)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("AUC response lower indicates higher sensitivity")
    plt.title(f"Selected held-out cell line: {selected_cell}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_cell_line_drug_ranking.pdf", dpi=300)
    plt.savefig(out_dir / "selected_cell_line_drug_ranking.png", dpi=300)
    plt.close()

    print("Saved case-study outputs to:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
