from dataclasses import dataclass

@dataclass
class ORCAConfig:
    # task: contT_contY / contT_binY / binT_contY / binT_binY
    task: str = "binT_contY"

    # treatment feature
    tfeat: str = "fourier"   # direct / fourier / mlp
    n_freq: int = 10
    tmlp_dim: int = 32

    # network
    x_hidden: int = 256
    rep_dim: int = 128
    head_hidden: int = 256
    dropout: float = 0.1

    # training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 200
    patience: int = 15
    min_delta: float = 1e-5

    # nuisance
    nuisance: str = "rf"     # rf / ridge / gbdt / mlp

    # cross-fitting (experimental)
    crossfit_folds: int = 1  # 1 means no cross-fitting; >1 means K-fold cross-fitting

    # contT eval grid
    t_min: float = -2.0
    t_max: float = 2.0
    t_grid_size: int = 50
