"""
Prepare omics and drug-response CSV files for the ORCA benchmark.

This script standardizes two files:

1. omics_out.csv
   cell_id,gene_1,gene_2,...

2. response_out.csv
   cell_id,drug_id,response
   or
   cell_id,smiles,response

It accepts drug response data in either long format:
   cell_id,drug_id,response

or wide matrix format:
   cell_id,drug_A,drug_B,drug_C,...

Example:
  python experiments/prepare_omics_drug_csv.py \
    --omics_csv raw/DepMap_expression.csv \
    --omics_cell_col DepMap_ID \
    --response_csv raw/GDSC_response.csv \
    --response_format long \
    --response_cell_col DepMap_ID \
    --drug_col drug_name \
    --y_col LN_IC50 \
    --drug_annotation_csv raw/drug_annotations.csv \
    --annotation_drug_col drug_name \
    --smiles_col smiles \
    --out_dir data/prepared_gdsc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def numeric_columns(df: pd.DataFrame, exclude: set[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def standardize_omics(path: str, cell_col: str, max_features: Optional[int]) -> pd.DataFrame:
    omics = pd.read_csv(path)
    if cell_col not in omics.columns:
        raise ValueError(f"Missing omics cell id column: {cell_col}")
    feature_cols = numeric_columns(omics, exclude={cell_col})
    if not feature_cols:
        raise ValueError("No numeric omics feature columns found.")
    omics = omics[[cell_col] + feature_cols].drop_duplicates(cell_col)
    omics = omics.rename(columns={cell_col: "cell_id"})
    omics = omics.dropna(axis=0, how="any")

    if max_features and len(feature_cols) > max_features:
        variances = omics[feature_cols].var(axis=0).sort_values(ascending=False)
        selected = variances.index[:max_features].tolist()
        omics = omics[["cell_id"] + selected]
    return omics


def response_long(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.response_csv)
    for col in [args.response_cell_col, args.drug_col, args.y_col]:
        if col not in df.columns:
            raise ValueError(f"Missing response column: {col}")
    response = df[[args.response_cell_col, args.drug_col, args.y_col]].copy()
    response = response.rename(
        columns={
            args.response_cell_col: "cell_id",
            args.drug_col: "drug_id",
            args.y_col: "response",
        }
    )
    return response


def response_wide(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.response_csv)
    cell_col = args.response_cell_col
    if cell_col not in df.columns:
        raise ValueError(f"Missing response cell id column: {cell_col}")
    value_cols = [c for c in df.columns if c != cell_col]
    if not value_cols:
        raise ValueError("No drug columns found in wide response table.")
    response = df.melt(
        id_vars=[cell_col],
        value_vars=value_cols,
        var_name="drug_id",
        value_name="response",
    )
    response = response.rename(columns={cell_col: "cell_id"})
    return response


def add_smiles_if_available(response: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.drug_annotation_csv:
        response["drug_key"] = response["drug_id"].astype(str)
        return response

    annot = pd.read_csv(args.drug_annotation_csv)
    for col in [args.annotation_drug_col, args.smiles_col]:
        if col not in annot.columns:
            raise ValueError(f"Missing drug annotation column: {col}")

    annot = annot[[args.annotation_drug_col, args.smiles_col]].dropna().drop_duplicates(args.annotation_drug_col)
    annot = annot.rename(columns={args.annotation_drug_col: "drug_id", args.smiles_col: "smiles"})
    response = response.merge(annot, on="drug_id", how="inner")
    response["drug_key"] = response["smiles"].astype(str)
    return response


def filter_response(response: pd.DataFrame, min_cells_per_drug: int, min_drugs_per_cell: int) -> pd.DataFrame:
    response = response.replace([np.inf, -np.inf], np.nan).dropna(subset=["cell_id", "drug_key", "response"])
    response["cell_id"] = response["cell_id"].astype(str)
    response["drug_key"] = response["drug_key"].astype(str)
    response["response"] = pd.to_numeric(response["response"], errors="coerce")
    response = response.dropna(subset=["response"])

    if min_cells_per_drug > 1:
        counts = response.groupby("drug_key")["cell_id"].nunique()
        keep_drugs = counts[counts >= min_cells_per_drug].index
        response = response[response["drug_key"].isin(keep_drugs)]

    if min_drugs_per_cell > 1:
        counts = response.groupby("cell_id")["drug_key"].nunique()
        keep_cells = counts[counts >= min_drugs_per_cell].index
        response = response[response["cell_id"].isin(keep_cells)]

    response = response[["cell_id", "drug_key", "response"]].rename(columns={"drug_key": "drug"})
    return response.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omics_csv", required=True)
    parser.add_argument("--omics_cell_col", required=True)
    parser.add_argument("--response_csv", required=True)
    parser.add_argument("--response_format", choices=["long", "wide"], required=True)
    parser.add_argument("--response_cell_col", required=True)
    parser.add_argument("--drug_col", default=None, help="Required for long response format.")
    parser.add_argument("--y_col", default=None, help="Required for long response format.")
    parser.add_argument("--drug_annotation_csv", default=None)
    parser.add_argument("--annotation_drug_col", default=None)
    parser.add_argument("--smiles_col", default=None)
    parser.add_argument("--out_dir", default="data/prepared_gdsc")
    parser.add_argument("--max_omics_features", type=int, default=None)
    parser.add_argument("--min_cells_per_drug", type=int, default=10)
    parser.add_argument("--min_drugs_per_cell", type=int, default=5)
    args = parser.parse_args()

    if args.response_format == "long" and (not args.drug_col or not args.y_col):
        raise ValueError("--drug_col and --y_col are required for long response format.")
    if args.drug_annotation_csv and (not args.annotation_drug_col or not args.smiles_col):
        raise ValueError("--annotation_drug_col and --smiles_col are required with --drug_annotation_csv.")

    out = ensure_dir(args.out_dir)
    omics = standardize_omics(args.omics_csv, args.omics_cell_col, args.max_omics_features)
    response = response_long(args) if args.response_format == "long" else response_wide(args)
    response = add_smiles_if_available(response, args)
    response = filter_response(response, args.min_cells_per_drug, args.min_drugs_per_cell)

    shared_cells = sorted(set(omics["cell_id"].astype(str)) & set(response["cell_id"].astype(str)))
    omics = omics[omics["cell_id"].astype(str).isin(shared_cells)].reset_index(drop=True)
    response = response[response["cell_id"].astype(str).isin(shared_cells)].reset_index(drop=True)

    omics_path = out / "omics.csv"
    response_path = out / "response.csv"
    omics.to_csv(omics_path, index=False)
    response.to_csv(response_path, index=False)

    summary = {
        "n_cells": int(len(shared_cells)),
        "n_omics_features": int(omics.shape[1] - 1),
        "n_response_rows": int(len(response)),
        "n_drugs": int(response["drug"].nunique()),
        "omics_csv": str(omics_path),
        "response_csv": str(response_path),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved prepared data to: {out}")


if __name__ == "__main__":
    main()
