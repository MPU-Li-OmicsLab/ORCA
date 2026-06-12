# ORCA: Orthogonal Residual Counterfactual Architecture

ORCA is a neural counterfactual learning framework for robust individualized
treatment-response estimation from observational data. It supports binary and
continuous treatments, binary and continuous outcomes, and several nuisance
model choices.

The core idea is to estimate nuisance functions for baseline outcome and
treatment assignment, construct residualized outcome and treatment signals, and
train a nonlinear residual prediction network. The final response surface is
obtained through additive reconstruction.

## Features

- Unified support for four treatment-outcome settings:
  - `contT_contY`: continuous treatment, continuous outcome
  - `contT_binY`: continuous treatment, binary outcome
  - `binT_contY`: binary treatment, continuous outcome
  - `binT_binY`: binary treatment, binary outcome
- Nuisance models: ridge, random forest, gradient boosting, and MLP
- Treatment representations: direct scalar input, Fourier features, and MLP embeddings
- Experimental K-fold cross-fitting support
- Command-line runner for NPZ or CSV datasets
- Research scripts for reproducing paper experiments

## Repository Layout

```text
ORCA-Neurocomputing/
|-- orca/                  # Clean reusable ORCA package
|   |-- config.py
|   |-- data.py
|   |-- metrics.py
|   |-- model.py
|   |-- nuisance.py
|   |-- trainer.py
|   `-- utils.py
|-- scripts/
|   `-- run_orca.py         # Simple CLI entry point
|-- experiments/            # Paper reproduction / research scripts
|-- docs/
|   `-- USAGE.md
|-- pyproject.toml
`-- requirements.txt
```

## Installation

```bash
git clone https://github.com/MPU-Li-OmicsLab/ORCA-Neurocomputing.git
cd ORCA-Neurocomputing
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Quick Start

Prepare an NPZ dataset with the required keys:

- `X_train`: training covariates, shape `(n_train, d)`
- `T_train`: treatment, shape `(n_train,)`
- `Y_train`: outcome, shape `(n_train,)`
- `X_test`: test covariates, shape `(n_test, d)`

Optional evaluation keys:

- For binary treatment tasks: `mu0_test`, `mu1_test`
- For continuous treatment tasks: `mu_grid_test`

Run ORCA:

```bash
python scripts/run_orca.py \
  --task contT_contY \
  --data_npz path/to/data.npz \
  --out_dir runs/demo \
  --nuisance rf \
  --tfeat fourier
```

Outputs are saved to `out_dir`, including `metrics.json` and prediction arrays.

## Paper Experiments

The `experiments/` folder contains the larger research scripts used for
benchmark and ablation studies:

- `ihdp_orca_ablation_all4.py`
- `news_orca_ablation_all4.py`
- `news_baselines_all4.py`
- `orca_chloride.py`
- `run_drnet_vcnet_csv_compare.py`

These scripts are intended for reproducibility and may contain dataset-specific
paths or assumptions. The `orca/` package is the cleaner reusable implementation
for public use.

## Documentation

See [docs/USAGE.md](docs/USAGE.md) for a more detailed usage guide.

## Citation

If you use this repository in academic work, please cite the associated ORCA
paper once bibliographic information is available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
