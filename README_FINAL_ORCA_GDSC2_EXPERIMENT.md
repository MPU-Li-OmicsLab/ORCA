# ORCA DepMap/GDSC2 Pharmacogenomic Experiment

This document describes the final Bioinformatics-oriented DepMap/GDSC2
experiment for ORCA. The benchmark uses DepMap expression features, GDSC2 AUC
drug-response values, and Morgan-fingerprint drug descriptors under held-out
cell-line splits.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-omics-drug-response.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Raw Input Files

Place the raw files in `raw/`:

```text
raw/DepMap_expression.csv
raw/GDSC2_AUC.csv
raw/PortalCompounds.csv
```

The manuscript experiment used DepMap 26Q1 expression profiles, GDSC2 AUC
drug-sensitivity measurements, and `PortalCompounds.csv` compound annotations.

## 1. Prepare DPC-ID Benchmark

```bash
python experiments/prepare_omics_drug_csv.py \
  --omics_csv raw/DepMap_expression.csv \
  --omics_cell_col ModelID \
  --response_csv raw/GDSC2_AUC.csv \
  --response_format wide \
  --response_cell_col "Unnamed: 0" \
  --out_dir data/prepared_gdsc
```

Expected scale:

```text
701 cancer cell lines
286 drugs
172,745 cell-line--drug response observations
```

## 2. Map Compounds to SMILES

```bash
python experiments/map_portal_compounds_smiles.py \
  --compound_csv raw/PortalCompounds.csv \
  --omics_csv data/prepared_gdsc/omics.csv \
  --response_csv data/prepared_gdsc/response.csv \
  --out_dir data/prepared_gdsc_smiles
```

## 3. Build Morgan-Fingerprint Dataset

```bash
python experiments/build_morgan_fingerprint_dataset.py \
  --omics_csv data/prepared_gdsc_smiles/omics.csv \
  --response_csv data/prepared_gdsc_smiles/response.csv \
  --out_dir data/prepared_gdsc_morgan \
  --drug_col drug \
  --radius 2 \
  --n_bits 2048
```

## 4. Run Final ORCA-Family Benchmark

This runner reports ridge, elastic net, histogram gradient boosting, LightGBM,
direct MLP, late-fusion two-tower MLP, basic ORCA, two-tower ORCA, and
ORCA-Ensemble. The two-tower MLP baseline controls for the benefit of separate
omics and drug encoders without ORCA nuisance residualization or additive
reconstruction.

If earlier ORCA-family outputs already exist in the same `out_dir`, the runner
skips completed runs and adds missing outputs such as the late-fusion two-tower
MLP baseline before regenerating `final_summary.csv`.

```bash
python experiments/run_final_orca_family.py \
  --omics_csv data/prepared_gdsc_morgan/omics.csv \
  --response_csv data/prepared_gdsc_morgan/response.csv \
  --drug_descriptor_csv data/prepared_gdsc_morgan/drug_descriptors.csv \
  --out_dir runs/gdsc_morgan_final_orca_family
```

The final summary is written to:

```text
runs/gdsc_morgan_final_orca_family/final_summary.csv
```

The manuscript reports:

```text
ORCA-Ensemble
RMSE     0.0996 +/- 0.0019
MAE      0.0683 +/- 0.0006
Pearson  0.8283 +/- 0.0068
Spearman 0.7448 +/- 0.0087
```

## 5. Generate Biological Case Study

```bash
python experiments/make_biological_case_study.py \
  --predictions_glob "runs/gdsc_morgan_final_orca_family/orca_ensemble_split_*/predictions_test_orca_ensemble.csv" \
  --prediction_col y_pred_orca_ensemble \
  --drug_metadata_csv raw/PortalCompounds.csv \
  --out_dir runs/gdsc_morgan_final_orca_family/case_study_all_splits \
  --min_drugs 30 \
  --top_k 10
```

Expected outputs include:

```text
case_study_summary.json
per_cell_line_metrics.csv
selected_cell_line_top_sensitive.csv
selected_cell_line_top_resistant.csv
per_cell_line_spearman.pdf
selected_cell_line_drug_ranking.pdf
```
