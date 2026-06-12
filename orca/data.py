from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd

Task = Literal["contT_contY", "contT_binY", "binT_contY", "binT_binY"]

@dataclass
class ORCADataset:
    """
    Minimal dataset contract.

    Required:
      X_train: (n_tr, d)
      T_train: (n_tr,)  float (cont) or {0,1} (bin)
      Y_train: (n_tr,)  float (cont) or {0,1} (bin)

      X_test:  (n_te, d)

    Optional (for evaluation with ground truth):
      - For binT tasks:
          mu0_test, mu1_test: (n_te,)
      - For contT tasks:
          mu_grid_test: (n_te, G) where G matches t_grid_size
    """
    task: Task
    X_train: np.ndarray
    T_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray

    # truth (optional)
    mu0_test: Optional[np.ndarray] = None
    mu1_test: Optional[np.ndarray] = None
    mu_grid_test: Optional[np.ndarray] = None


def load_npz_dataset(path: str, task: Task) -> ORCADataset:
    """
    Expected keys (recommended):
      X_train, T_train, Y_train, X_test
    Optional:
      mu0_test, mu1_test (binT)
      mu_grid_test (contT)

    All arrays should be float32; bin labels can be 0/1 float32.
    """
    z = np.load(path)
    Xtr = z["X_train"].astype(np.float32)
    Ttr = z["T_train"].astype(np.float32)
    Ytr = z["Y_train"].astype(np.float32)
    Xte = z["X_test"].astype(np.float32)

    mu0 = z["mu0_test"].astype(np.float32) if "mu0_test" in z else None
    mu1 = z["mu1_test"].astype(np.float32) if "mu1_test" in z else None
    mug = z["mu_grid_test"].astype(np.float32) if "mu_grid_test" in z else None

    return ORCADataset(
        task=task,
        X_train=Xtr, T_train=Ttr, Y_train=Ytr,
        X_test=Xte,
        mu0_test=mu0, mu1_test=mu1,
        mu_grid_test=mug
    )


def load_csv_dataset(path: str, task: Task,
                     x_cols: Optional[list[str]] = None,
                     t_col: str = "t",
                     y_col: str = "y",
                     split_col: Optional[str] = None) -> ORCADataset:
    """
    CSV loader for quick demos.
    Two options:
      (A) split_col given (values: 'train'/'test')
      (B) no split_col -> no true test; we will use last 10% rows as test (for demo)

    CSV must contain: features + t_col + y_col.
    """
    df = pd.read_csv(path)

    if x_cols is None:
        x_cols = [c for c in df.columns if c not in {t_col, y_col, split_col}]

    if split_col is not None and split_col in df.columns:
        tr = df[df[split_col] == "train"].copy()
        te = df[df[split_col] == "test"].copy()
    else:
        n = len(df)
        cut = int(n * 0.9)
        tr = df.iloc[:cut].copy()
        te = df.iloc[cut:].copy()

    Xtr = tr[x_cols].to_numpy(dtype=np.float32)
    Ttr = tr[t_col].to_numpy(dtype=np.float32)
    Ytr = tr[y_col].to_numpy(dtype=np.float32)
    Xte = te[x_cols].to_numpy(dtype=np.float32)

    return ORCADataset(task=task, X_train=Xtr, T_train=Ttr, Y_train=Ytr, X_test=Xte)
