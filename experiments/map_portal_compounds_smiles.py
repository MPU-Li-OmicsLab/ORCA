"""
Map DepMap/GDSC compound IDs to SMILES using PortalCompounds.csv.

Input:
  data/prepared_gdsc/omics.csv
  data/prepared_gdsc/response.csv with columns: cell_id,drug,response
  raw/PortalCompounds.csv

Output:
  data/prepared_gdsc_smiles/omics.csv
  data/prepared_gdsc_smiles/response.csv with columns: cell_id,drug,response

Example:
  python experiments/map_portal_compounds_smiles.py \
    --compound_csv raw/PortalCompounds.csv \
    --omics_csv data/prepared_gdsc/omics.csv \
    --response_csv data/prepared_gdsc/response.csv \
    --out_dir data/prepared_gdsc_smiles
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def find_smiles_col(compounds: pd.DataFrame) -> str:
    for col in compounds.columns:
        if col.lower() == "smiles":
            return col
    for col in compounds.columns:
        if "smiles" in col.lower():
            return col
    raise ValueError("Could not find a SMILES column in PortalCompounds.csv.")


def find_best_compound_id_col(compounds: pd.DataFrame, response_drugs: set[str]) -> tuple[str, int]:
    candidates = []
    for col in compounds.columns:
        vals = compounds[col].dropna().astype(str)
        if vals.str.startswith("DPC-").any():
            overlap = len(response_drugs & set(vals))
            candidates.append((col, overlap))

    if not candidates:
        raise ValueError("Could not find any PortalCompounds column containing DPC- IDs.")

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_overlap = candidates[0]
    if best_overlap <= 0:
        details = ", ".join(f"{col}: {overlap}" for col, overlap in candidates[:10])
        raise ValueError(f"No overlap between response drug IDs and DPC columns. Candidates: {details}")
    return best_col, best_overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compound_csv", default="raw/PortalCompounds.csv")
    parser.add_argument("--omics_csv", default="data/prepared_gdsc/omics.csv")
    parser.add_argument("--response_csv", default="data/prepared_gdsc/response.csv")
    parser.add_argument("--out_dir", default="data/prepared_gdsc_smiles")
    args = parser.parse_args()

    compound_path = Path(args.compound_csv)
    omics_path = Path(args.omics_csv)
    response_path = Path(args.response_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compounds = pd.read_csv(compound_path)
    response = pd.read_csv(response_path)
    if not {"cell_id", "drug", "response"}.issubset(response.columns):
        raise ValueError("response_csv must contain columns: cell_id, drug, response")

    smiles_col = find_smiles_col(compounds)
    id_col, overlap = find_best_compound_id_col(compounds, set(response["drug"].astype(str)))

    mapping = compounds[[id_col, smiles_col]].dropna().drop_duplicates(id_col)
    mapping = mapping.rename(columns={id_col: "drug", smiles_col: "smiles"})

    merged = response.merge(mapping, on="drug", how="inner")
    smiles_response = merged[["cell_id", "smiles", "response"]].rename(columns={"smiles": "drug"})

    out_omics = out_dir / "omics.csv"
    out_response = out_dir / "response.csv"
    shutil.copyfile(omics_path, out_omics)
    smiles_response.to_csv(out_response, index=False)

    summary = {
        "compound_csv": str(compound_path),
        "compound_id_column": id_col,
        "smiles_column": smiles_col,
        "overlap_drugs": int(overlap),
        "original_response_rows": int(len(response)),
        "mapped_response_rows": int(len(smiles_response)),
        "original_drugs": int(response["drug"].nunique()),
        "mapped_drugs": int(merged["drug"].nunique()),
        "unique_smiles": int(smiles_response["drug"].nunique()),
        "omics_csv": str(out_omics),
        "response_csv": str(out_response),
    }
    (out_dir / "smiles_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved SMILES-mapped benchmark to: {out_dir}")


if __name__ == "__main__":
    main()
