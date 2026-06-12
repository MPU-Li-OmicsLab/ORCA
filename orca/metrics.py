import numpy as np

def rmse(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def sqrt_pehe(ite_hat, ite_true) -> float:
    ite_hat = np.asarray(ite_hat, dtype=np.float32)
    ite_true = np.asarray(ite_true, dtype=np.float32)
    return float(np.sqrt(np.mean((ite_hat - ite_true) ** 2)))

def mise_grid(mu_hat, mu_true, t_grid) -> float:
    # (n,G)
    dt = float(t_grid[1] - t_grid[0])
    mu_hat = np.asarray(mu_hat, dtype=np.float32)
    mu_true = np.asarray(mu_true, dtype=np.float32)
    return float(np.mean(np.sum((mu_hat - mu_true) ** 2, axis=1) * dt))
