"""
Run the ORCA omics drug-response benchmark across multiple random seeds.

Example:
  python experiments/run_repeated_omics_benchmark.py \
    --omics_csv data/prepared_gdsc/omics.csv \
    --response_csv data/prepared_gdsc/response.csv \
    --seeds 1 2 3 4 5 \
    --out_dir runs/gdsc_orca_repeated
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "experiments" / "omics_drug_response.py"


def load_metrics(path: Path, seed: int) -> List[Dict[str, object]]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for method, vals in metrics.items():
        row: Dict[str, object] = {"seed": seed, "method": method}
        row.update(vals)
        rows.append(row)
    return rows


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omics_csv", required=True)
    parser.add_argument("--response_csv", required=True)
    parser.add_argument("--cell_col", default="cell_id")
    parser.add_argument("--response_cell_col", default="cell_id")
    parser.add_argument("--drug_col", default="drug")
    parser.add_argument("--y_col", default="response")
    parser.add_argument("--out_dir", default="runs/gdsc_orca_repeated")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--drug_hash_dim", type=int, default=256)
    parser.add_argument("--max_omics_features", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--no_sklearn_baselines", action="store_true")
    parser.add_argument("--disable_residual_t", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []

    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--omics_csv",
            args.omics_csv,
            "--response_csv",
            args.response_csv,
            "--cell_col",
            args.cell_col,
            "--response_cell_col",
            args.response_cell_col,
            "--drug_col",
            args.drug_col,
            "--y_col",
            args.y_col,
            "--out_dir",
            str(seed_dir),
            "--seed",
            str(seed),
            "--test_size",
            str(args.test_size),
            "--val_size",
            str(args.val_size),
            "--drug_hash_dim",
            str(args.drug_hash_dim),
            "--max_omics_features",
            str(args.max_omics_features),
            "--hidden_dim",
            str(args.hidden_dim),
            "--depth",
            str(args.depth),
            "--dropout",
            str(args.dropout),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
        ]
        if args.max_rows:
            cmd.extend(["--max_rows", str(args.max_rows)])
        if args.no_sklearn_baselines:
            cmd.append("--no_sklearn_baselines")
        if args.disable_residual_t:
            cmd.append("--disable_residual_t")

        print(f"\nRunning seed {seed}:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        all_rows.extend(load_metrics(seed_dir / "metrics.json", seed))

    raw = pd.DataFrame(all_rows)
    raw_path = out_root / "metrics_by_seed.csv"
    raw.to_csv(raw_path, index=False)

    metric_cols = [c for c in ["rmse", "mae", "pearson", "spearman"] if c in raw.columns]
    grouped = raw.groupby("method")[metric_cols].agg(["mean", "std"])
    grouped.to_csv(out_root / "summary_mean_std_numeric.csv")

    order = grouped[("rmse", "mean")].sort_values().index if "rmse" in metric_cols else grouped.index
    pretty = pd.DataFrame(index=order)
    for metric in metric_cols:
        pretty[metric] = [
            format_mean_std(grouped.loc[method, (metric, "mean")], grouped.loc[method, (metric, "std")])
            for method in order
        ]
    pretty.to_csv(out_root / "summary_mean_std.csv")

    print("\nSummary")
    print(pretty.to_string())
    print(f"\nSaved repeated-run outputs to: {out_root}")


if __name__ == "__main__":
    main()
