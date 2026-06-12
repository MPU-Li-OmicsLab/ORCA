import os
import numpy as np
import torch

def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit_np(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
