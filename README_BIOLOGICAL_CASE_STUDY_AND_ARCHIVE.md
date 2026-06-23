# Biological Case Study and Code Archive Checklist

This checklist supports the Bioinformatics submission package for ORCA.

## Generate the biological case-study outputs

Run this from the repository root after completing the final ORCA-family
benchmark:

```bash
python experiments/make_biological_case_study.py \
  --predictions_glob "runs/gdsc_morgan_final_orca_family/orca_ensemble_split_*/predictions_test_orca_ensemble.csv" \
  --prediction_col y_pred_orca_ensemble \
  --drug_metadata_csv raw/PortalCompounds.csv \
  --out_dir runs/gdsc_morgan_final_orca_family/case_study_all_splits \
  --min_drugs 30 \
  --top_k 10
```

Expected outputs:

```text
runs/gdsc_morgan_final_orca_family/case_study_all_splits/
  case_study_summary.json
  per_cell_line_metrics.csv
  selected_cell_line_top_sensitive.csv
  selected_cell_line_top_resistant.csv
  per_cell_line_spearman.pdf
  per_cell_line_spearman.png
  selected_cell_line_drug_ranking.pdf
  selected_cell_line_drug_ranking.png
```

## Archive the exact code version

Before journal submission:

1. Ensure the GitHub repository contains the final scripts and README files.
2. Create a GitHub release, for example `v1.0.0-bioinformatics-submission`.
3. Archive the release on Zenodo, Figshare, or Software Heritage.
4. Add the DOI or permanent archive identifier to the manuscript Code Availability section.

Suggested final wording:

```text
The exact code version used for this submission is archived at Zenodo:
https://doi.org/10.5281/zenodo.XXXXXXX.
```

Replace `XXXXXXX` with the real DOI before submission.
