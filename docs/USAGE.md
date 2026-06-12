============================================================
ORCA (Orthogonal Residual Counterfactual Architecture) - USAGE GUIDE
============================================================

0) Quick Start (TL;DR)
----------------------
(1) Install:
    pip install -e .

(2) Prepare dataset in NPZ format (recommended):
    Must include:
      - X_train: (n_train, d) float32
      - T_train: (n_train,)   float32 (continuous) or {0,1} float32 (binary)
      - Y_train: (n_train,)   float32 (continuous) or {0,1} float32 (binary)
      - X_test : (n_test, d)  float32

    Optional (if you want evaluation with ground truth):
      - For binT_* tasks:
          mu0_test: (n_test,) float32
          mu1_test: (n_test,) float32
      - For contT_* tasks:
          mu_grid_test: (n_test, G) float32, where G = t_grid_size

(3) Run:
    python scripts/run_orca.py --task binT_contY --data_npz /path/to/data.npz --out_dir ./runs/demo

Outputs will be saved in out_dir:
    - metrics.json
    - (contT_*) pred_grid.npy + t_grid.npy
    - (binT_*) mu0.npy + mu1.npy


1) Installation
---------------
Requirements:
  - Python >= 3.9
  - PyTorch >= 2.0
  - numpy/pandas/scikit-learn/tqdm

Install in editable mode (recommended for research):
  pip install -e .

If you want a clean environment:
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -e .


2) Supported Tasks (4 settings)
-------------------------------
We support four canonical settings:
  (A) contT_contY : continuous treatment, continuous outcome
  (B) contT_binY  : continuous treatment, binary outcome
  (C) binT_contY  : binary treatment, continuous outcome
  (D) binT_binY   : binary treatment, binary outcome

CLI argument:
  --task {contT_contY, contT_binY, binT_contY, binT_binY}

Data constraints:
  - Binary treatment should be {0,1} in T_train.
  - Binary outcome should be {0,1} in Y_train.
  - Continuous treatment typically recommended to be normalized/standardized and clipped to a reasonable range.


3) Dataset Format (NPZ recommended)
-----------------------------------
Create a .npz file with keys:

Required:
  X_train  : float32, shape (n_train, d)
  T_train  : float32, shape (n_train,)
  Y_train  : float32, shape (n_train,)
  X_test   : float32, shape (n_test, d)

Optional ground-truth (for evaluation only):
  For binT_*:
    mu0_test : float32, shape (n_test,)
    mu1_test : float32, shape (n_test,)
    (for binT_binY, these should be probabilities p0/p1 in [0,1])

  For contT_*:
    mu_grid_test : float32, shape (n_test, G)
    where G equals t_grid_size (default 50).

Example: save NPZ in python:
  import numpy as np
  np.savez("mydata.npz",
           X_train=Xtr.astype("float32"),
           T_train=Ttr.astype("float32"),
           Y_train=Ytr.astype("float32"),
           X_test=Xte.astype("float32"),
           mu0_test=mu0.astype("float32"),
           mu1_test=mu1.astype("float32"))

NOTE:
  - For contT tasks, the code constructs t_grid internally:
      t_grid = linspace(t_min, t_max, t_grid_size)
    You should ensure mu_grid_test matches this grid.


4) Alternative: CSV Dataset (for quick demos)
---------------------------------------------
You can also provide a CSV:
  - Feature columns: any columns except t/y/split
  - t column: default name "t"
  - y column: default name "y"
  - optional split column: "split" with values {"train","test"}

Example CSV headers:
  x1,x2,x3,...,t,y,split

Run:
  python scripts/run_orca.py --task binT_contY --data_csv ./data.csv --out_dir ./runs/demo


5) Running ORCA: CLI Examples
-----------------------------
Basic run:
  python scripts/run_orca.py \
    --task binT_contY \
    --data_npz /path/to/data.npz \
    --out_dir ./runs/binT_contY \
    --device cpu \
    --seed 0 \
    --nuisance rf \
    --tfeat fourier

Change nuisance model:
  --nuisance {ridge,rf,gbdt,mlp}

Change treatment feature type:
  --tfeat {direct,fourier,mlp}

Enable experimental cross-fitting:
  --crossfit_folds 2


6) Outputs Explained
--------------------
In out_dir you will see:

(1) metrics.json
    Contains:
      - task name
      - metrics (if ground truth exists)
      - config used

(2) For contT_* tasks:
    - pred_grid.npy : shape (n_test, G)
      predicted dose-response curve on t_grid
    - t_grid.npy    : shape (G,)
      evaluation grid used

(3) For binT_* tasks:
    - mu0.npy : shape (n_test,)
    - mu1.npy : shape (n_test,)
      predicted potential outcomes under t=0 and t=1
      For binY, these represent probabilities in [0,1].


7) File-by-file Explanation (.py)
---------------------------------
orca/__init__.py
  - Package entrypoint: exposes public API:
      ORCAConfig, ORCA, ORCADataset, fit_orca, loaders.

orca/config.py
  - ORCAConfig dataclass: all hyperparameters and task settings in one place.
  - Includes training params, tfeat params, grid params, cross-fitting folds.

orca/utils.py
  - Utilities:
      set_all_seeds(), sigmoid_np(), logit_np(), ensure_dir().

orca/metrics.py
  - Evaluation metrics:
      rmse, sqrt_pehe, mise_grid.

orca/data.py
  - Defines ORCADataset (the minimal required data interface).
  - load_npz_dataset(): load NPZ with standard keys.
  - load_csv_dataset(): load CSV for quick demo.

orca/nuisance.py
  - Nuisance model factory for m(X) and e(X).
  - Supports ridge/rf/gbdt/mlp for both regression and classification.
  - TorchMLPRegressor / TorchMLPClassifier are included.

orca/model.py
  - Core ORCA model:
      ResidualNet (Encoder + treatment features + residual head)
      ORCA wrapper (fit + predict):
        predict_contT_grid() for contT tasks
        predict_binT() for binT tasks
  - Treatment feature options:
      direct / fourier / mlp

orca/trainer.py
  - fit_orca(): high-level training entry.
  - Handles:
      - creating nuisance models
      - experimental cross-fitting
      - running prediction
      - evaluation if ground truth exists

scripts/run_orca.py
  - CLI script:
      parses arguments, loads data, trains ORCA, saves outputs.

pyproject.toml
  - Packaging / dependencies so users can do:
      pip install -e .

README.md (optional)
  - A shorter markdown version of this USAGE.txt for GitHub homepage.


8) Notes for Maintainers (recommended)
--------------------------------------
- Keep comments ASCII-only if your cluster has encoding issues.
- For binary outcome training, the reference implementation keeps the residual-loss path compact; for exact
  reproduction of a specific paper experiment, verify the binary-outcome loss against the experiment script.
- For cross-fitting, the package includes a simple K-fold residualization workflow. For strict paper
  reproduction, match the fold construction and random seeds used in the corresponding experiment script.


9) Citation
-----------
If you use this repo, please cite our paper:
  (Add your BibTeX here)

============================================================
END
============================================================
