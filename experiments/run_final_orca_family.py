"""Run the final DepMap/GDSC2 ORCA-family benchmark.

This runner reproduces the Bioinformatics submission benchmark:
  - basic ORCA plus shared-input baselines across split seeds 1, 2, 3
  - late-fusion two-tower MLP baseline across split seeds 1, 2, 3
  - two-tower ORCA across split seeds 1, 2, 3
  - ORCA-Ensemble with five independently initialized two-tower models per split

Example:
  python experiments/run_final_orca_family.py \
    --omics_csv data/prepared_gdsc_morgan/omics.csv \
    --response_csv data/prepared_gdsc_morgan/response.csv \
    --drug_descriptor_csv data/prepared_gdsc_morgan/drug_descriptors.csv \
    --out_dir runs/gdsc_morgan_final_orca_family
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "experiments" / "omics_drug_response.py"
METRICS = ["rmse", "mae", "pearson", "spearman"]


def run_cmd(cmd: List[str], metrics_path: Path) -> None:
    if metrics_path.exists():
        print(f"[skip] {metrics_path.parent}")
        return
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensemble_predictions(root: Path, split_seed: int) -> Dict[str, float]:
    model_dirs = sorted((root / f"orca_ensemble_split_{split_seed}").glob("model_seed_*"))
    frames = []
    for model_dir in model_dirs:
        path = model_dir / "predictions_test.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
        pred_col = "y_pred_orca" if "y_pred_orca" in df.columns else pred_cols[0]
        frames.append(df[["cell_id", "drug_id", "y_true", pred_col]].rename(columns={pred_col: "pred"}))

    if not frames:
        raise RuntimeError(f"No ensemble predictions found for split {split_seed}.")

    base = frames[0][["cell_id", "drug_id", "y_true"]].reset_index(drop=True)
    for i, frame in enumerate(frames[1:], start=2):
        check = frame[["cell_id", "drug_id", "y_true"]].reset_index(drop=True)
        if not base.equals(check):
            raise RuntimeError(f"Prediction rows do not match for ensemble member {i}.")

    pred = np.vstack([frame["pred"].to_numpy(float) for frame in frames]).mean(axis=0)
    y = base["y_true"].to_numpy(float)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "pearson": float(pearsonr(y, pred)[0]),
        "spearman": float(spearmanr(y, pred)[0]),
        "n_models": int(len(frames)),
        "split_seed": int(split_seed),
    }
    out_dir = root / f"orca_ensemble_split_{split_seed}"
    base["y_pred_orca_ensemble"] = pred
    base.to_csv(out_dir / "predictions_test_orca_ensemble.csv", index=False)
    (out_dir / "metrics_orca_ensemble.json").write_text(
        json.dumps({"orca_ensemble": metrics}, indent=2),
        encoding="utf-8",
    )
    return metrics


def summarize(root: Path) -> pd.DataFrame:
    methods: Dict[str, List[Dict[str, float]]] = {}

    def add(name: str, row: Dict[str, float]) -> None:
        methods.setdefault(name, []).append(row)

    for seed_dir in sorted((root / "orca").glob("seed_*")):
        path = seed_dir / "metrics.json"
        if path.exists():
            for name, row in json.loads(path.read_text()).items():
                add(name, row)

    for seed_dir in sorted((root / "two_tower_orca").glob("seed_*")):
        path = seed_dir / "metrics.json"
        if path.exists():
            data = json.loads(path.read_text())
            if "orca" in data:
                add("two_tower_orca", data["orca"])

    for seed_dir in sorted((root / "two_tower_mlp").glob("seed_*")):
        path = seed_dir / "metrics.json"
        if path.exists():
            data = json.loads(path.read_text())
            if "two_tower_mlp" in data:
                add("two_tower_mlp", data["two_tower_mlp"])

    for ens_dir in sorted(root.glob("orca_ensemble_split_*")):
        path = ens_dir / "metrics_orca_ensemble.json"
        if path.exists():
            data = json.loads(path.read_text())
            if "orca_ensemble" in data:
                add("orca_ensemble", data["orca_ensemble"])

    order = [
        "elastic_net",
        "ridge",
        "hist_gradient_boosting",
        "lightgbm",
        "mlp",
        "two_tower_mlp",
        "orca",
        "two_tower_orca",
        "orca_ensemble",
    ]
    rows = []
    for method in order:
        if method not in methods:
            continue
        out = {"method": method}
        for metric in METRICS:
            vals = [float(row[metric]) for row in methods[method]]
            out[f"{metric}_mean"] = st.mean(vals)
            out[f"{metric}_sd"] = st.stdev(vals) if len(vals) > 1 else 0.0
        rows.append(out)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "final_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omics_csv", required=True)
    parser.add_argument("--response_csv", required=True)
    parser.add_argument("--drug_descriptor_csv", required=True)
    parser.add_argument("--drug_descriptor_col", default="drug")
    parser.add_argument("--cell_col", default="cell_id")
    parser.add_argument("--response_cell_col", default="cell_id")
    parser.add_argument("--drug_col", default="drug")
    parser.add_argument("--y_col", default="response")
    parser.add_argument("--out_dir", default="runs/gdsc_morgan_final_orca_family")
    parser.add_argument("--split_seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--model_seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--omics_csv", args.omics_csv,
        "--response_csv", args.response_csv,
        "--drug_descriptor_csv", args.drug_descriptor_csv,
        "--drug_descriptor_col", args.drug_descriptor_col,
        "--cell_col", args.cell_col,
        "--response_cell_col", args.response_cell_col,
        "--drug_col", args.drug_col,
        "--y_col", args.y_col,
        "--test_size", "0.2",
        "--val_size", "0.15",
        "--max_omics_features", "5000",
        "--hidden_dim", "256",
        "--depth", "2",
        "--dropout", "0.1",
        "--lr", "0.001",
        "--weight_decay", "0.0001",
        "--batch_size", "256",
        "--epochs", "300",
        "--patience", "30",
        "--device", args.device,
        "--e_alpha", "10",
        "--residual_t_scale", "0.5",
    ]

    for split_seed in args.split_seeds:
        run_cmd(
            [sys.executable, str(BENCHMARK_SCRIPT), *common, "--seed", str(split_seed), "--split_seed", str(split_seed), "--orca_arch", "plain", "--no_two_tower_mlp_baseline", "--out_dir", str(out_root / "orca" / f"seed_{split_seed}")],
            out_root / "orca" / f"seed_{split_seed}" / "metrics.json",
        )
        run_cmd(
            [sys.executable, str(BENCHMARK_SCRIPT), *common, "--seed", str(split_seed), "--split_seed", str(split_seed), "--only_two_tower_mlp", "--no_sklearn_baselines", "--out_dir", str(out_root / "two_tower_mlp" / f"seed_{split_seed}")],
            out_root / "two_tower_mlp" / f"seed_{split_seed}" / "metrics.json",
        )
        run_cmd(
            [sys.executable, str(BENCHMARK_SCRIPT), *common, "--seed", str(split_seed), "--split_seed", str(split_seed), "--orca_arch", "twotower", "--no_sklearn_baselines", "--out_dir", str(out_root / "two_tower_orca" / f"seed_{split_seed}")],
            out_root / "two_tower_orca" / f"seed_{split_seed}" / "metrics.json",
        )
        for model_seed in args.model_seeds:
            run_cmd(
                [sys.executable, str(BENCHMARK_SCRIPT), *common, "--seed", str(model_seed), "--split_seed", str(split_seed), "--orca_arch", "twotower", "--no_sklearn_baselines", "--out_dir", str(out_root / f"orca_ensemble_split_{split_seed}" / f"model_seed_{model_seed}")],
                out_root / f"orca_ensemble_split_{split_seed}" / f"model_seed_{model_seed}" / "metrics.json",
            )
        metrics = ensemble_predictions(out_root, split_seed)
        print(f"ORCA-Ensemble split {split_seed}: {metrics}")

    summary = summarize(out_root)
    print("\nFinal summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
