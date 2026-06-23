"""
Build a Morgan-fingerprint drug-response dataset for the ORCA benchmark.

Inputs are the prepared SMILES-based ORCA files:
  data/prepared_gdsc_smiles/omics.csv
  data/prepared_gdsc_smiles/response.csv

The response table must contain a drug column whose values are canonical
SMILES strings. The script writes:
  out_dir/omics.csv
  out_dir/response.csv
  out_dir/drug_descriptors.csv

Example:
  python experiments/build_morgan_fingerprint_dataset.py \
    --omics_csv data/prepared_gdsc_smiles/omics.csv \
    --response_csv data/prepared_gdsc_smiles/response.csv \
    --out_dir data/prepared_gdsc_morgan
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omics_csv", default="data/prepared_gdsc_smiles/omics.csv")
    parser.add_argument("--response_csv", default="data/prepared_gdsc_smiles/response.csv")
    parser.add_argument("--out_dir", default="data/prepared_gdsc_morgan")
    parser.add_argument("--drug_col", default="drug")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omics = pd.read_csv(args.omics_csv)
    response = pd.read_csv(args.response_csv)
    if args.drug_col not in response.columns:
        raise ValueError(f"response_csv must contain drug column: {args.drug_col}")

    smiles_values = sorted(response[args.drug_col].dropna().astype(str).unique())
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.n_bits)

    records = []
    bad_smiles = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bad_smiles.append(smiles)
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((args.n_bits,), dtype=np.int8)
        arr[list(fp.GetOnBits())] = 1
        row = {args.drug_col: smiles}
        row.update({f"morgan_{i}": int(arr[i]) for i in range(args.n_bits)})
        records.append(row)

    descriptors = pd.DataFrame(records)
    keep = set(descriptors[args.drug_col])
    response = response[response[args.drug_col].astype(str).isin(keep)].copy()

    omics.to_csv(out_dir / "omics.csv", index=False)
    response.to_csv(out_dir / "response.csv", index=False)
    descriptors.to_csv(out_dir / "drug_descriptors.csv", index=False)

    print("omics:", omics.shape)
    print("response:", response.shape)
    print("drug_descriptors:", descriptors.shape)
    print("bad_smiles:", len(bad_smiles))
    if bad_smiles:
        print("bad_smiles_examples:", bad_smiles[:5])


if __name__ == "__main__":
    main()
