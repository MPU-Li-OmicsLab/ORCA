import math
import numpy as np
import torch
import torch.nn as nn

from .utils import sigmoid_np, logit_np

class ResidualNet(nn.Module):
    """
    ORCA residual network:
      rep = Encoder(x)
      tf  = phi(t)
      out = Head([rep, tf, t_tilde])
    """
    def __init__(self, x_dim, x_hidden=256, rep_dim=128, head_hidden=256,
                 dropout=0.1, tfeat="fourier", n_freq=10, tmlp_dim=32):
        super().__init__()
        self.tfeat = tfeat
        self.n_freq = int(n_freq)

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, x_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(x_hidden, x_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(x_hidden, rep_dim),
        )

        if tfeat == "direct":
            self.basis_dim = 1
            self.tmlp = None
        elif tfeat == "fourier":
            self.basis_dim = 1 + 2 * self.n_freq
            self.tmlp = None
        elif tfeat == "mlp":
            self.basis_dim = int(tmlp_dim)
            self.tmlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, self.basis_dim),
            )
        else:
            raise ValueError("tfeat must be direct/fourier/mlp")

        self.head = nn.Sequential(
            nn.Linear(rep_dim + self.basis_dim + 1, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def t_features(self, t: torch.Tensor) -> torch.Tensor:
        # t: (b,)
        if self.tfeat == "direct":
            return t.unsqueeze(1)
        if self.tfeat == "mlp":
            return self.tmlp(t.unsqueeze(1))
        # fourier
        x = (t / 2.0) * math.pi
        feats = [torch.ones_like(x)]
        for k in range(1, self.n_freq + 1):
            feats.append(torch.sin(k * x))
            feats.append(torch.cos(k * x))
        return torch.stack(feats, dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, t_tilde: torch.Tensor) -> torch.Tensor:
        rep = self.encoder(x)
        tf = self.t_features(t)
        h = torch.cat([rep, tf, t_tilde.unsqueeze(1)], dim=1)
        return self.head(h).squeeze(1)


class ORCA:
    """
    High-level ORCA wrapper.
    - Fit: train nuisance m,e and residual net.
    - Predict:
        contT_*: returns grid curve (n,G) given t_grid
        binT_*: returns (mu0, mu1) (n,)
    """

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.net = None
        self.m_model = None
        self.e_model = None

    def _build_net(self, x_dim: int):
        self.net = ResidualNet(
            x_dim=x_dim,
            x_hidden=self.cfg.x_hidden,
            rep_dim=self.cfg.rep_dim,
            head_hidden=self.cfg.head_hidden,
            dropout=self.cfg.dropout,
            tfeat=self.cfg.tfeat,
            n_freq=self.cfg.n_freq,
            tmlp_dim=self.cfg.tmlp_dim
        ).to(self.device)

    @staticmethod
    def _is_binT(task: str) -> bool:
        return task.startswith("binT_")

    @staticmethod
    def _is_binY(task: str) -> bool:
        return task.endswith("_binY")

    @staticmethod
    def _is_contT(task: str) -> bool:
        return task.startswith("contT_")

    def fit_onefold(self, X, T, Y, m_model, e_model, seed: int):
        """
        Fit nuisance on full train and train residual net with early stopping on internal val split.
        For open-source skeleton: keep simple and robust.
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        task = self.cfg.task
        is_binT = self._is_binT(task)
        is_binY = self._is_binY(task)
        is_contT = self._is_contT(task)

        # fit nuisances
        if is_contT:
            # e(X): regression -> T
            e_model.fit(X, T)
            e_hat = e_model.predict(X).astype(np.float32)
        else:
            # e(X): classification -> P(T=1|X)
            e_model.fit(X, T.astype(int))
            e_hat = e_model.predict_proba(X)[:, 1].astype(np.float32)
            e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

        if not is_binY:
            # m(X): regression -> Y
            m_model.fit(X, Y)
            m_hat = m_model.predict(X).astype(np.float32)
            base_logit = None
        else:
            # m(X): classification -> P(Y=1|X) ; base_logit = logit(p)
            m_model.fit(X, Y.astype(int))
            p_hat = m_model.predict_proba(X)[:, 1].astype(np.float32)
            p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
            base_logit = logit_np(p_hat)
            m_hat = p_hat  # probability baseline

        # targets for residual net
        y_target = (Y.astype(np.float32) - m_hat).astype(np.float32)
        t_tilde = (T.astype(np.float32) - e_hat).astype(np.float32)

        # train/val split for net
        X_fit, X_va, T_fit, T_va, yt_fit, yt_va, tt_fit, tt_va, bl_fit, bl_va = train_test_split(
            X, T, y_target, t_tilde,
            (base_logit if base_logit is not None else np.zeros_like(Y, dtype=np.float32)),
            test_size=0.2, random_state=seed
        )

        # build net
        if self.net is None:
            self._build_net(x_dim=X.shape[1])

        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        ds = TensorDataset(
            torch.from_numpy(X_fit.astype(np.float32)),
            torch.from_numpy(T_fit.astype(np.float32)),
            torch.from_numpy(yt_fit.astype(np.float32)),
            torch.from_numpy(tt_fit.astype(np.float32)),
            torch.from_numpy(bl_fit.astype(np.float32)),
        )
        dl = DataLoader(ds, batch_size=min(self.cfg.batch_size, len(ds)), shuffle=True, drop_last=False)

        Xva_t = torch.from_numpy(X_va.astype(np.float32)).to(self.device)
        Tva_t = torch.from_numpy(T_va.astype(np.float32)).to(self.device)
        yva_t = torch.from_numpy(yt_va.astype(np.float32)).to(self.device)
        ttva_t = torch.from_numpy(tt_va.astype(np.float32)).to(self.device)
        blva_t = torch.from_numpy(bl_va.astype(np.float32)).to(self.device)

        best = float("inf"); best_state=None; wait=0
        for _ in range(self.cfg.epochs):
            self.net.train()
            for xb, tb, yb, ttb, blb in dl:
                xb = xb.to(self.device)
                tb = tb.to(self.device)
                yb = yb.to(self.device)
                ttb = ttb.to(self.device)
                blb = blb.to(self.device)

                opt.zero_grad(set_to_none=True)

                delta = self.net(xb, tb, ttb)

                if not is_binY:
                    loss = F.mse_loss(delta, yb)
                else:
                    # learn delta on top of base logit
                    logits = blb + delta
                    # y in {0,1} (we need original Y, but yb here is residual prob-space)
                    # we reconstruct y from residual + m_hat: y = y_target + m_hat
                    # in practice for binY, it's better to store original y in dataset;
                    # skeleton: approximate by clamp(y_target + m_hat) -> y01
                    y01 = torch.clamp(yb + torch.sigmoid(blb), 0.0, 1.0)
                    loss = F.binary_cross_entropy_with_logits(logits, y01)

                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                d_va = self.net(Xva_t, Tva_t, ttva_t)
                if not is_binY:
                    v = F.mse_loss(d_va, yva_t).item()
                else:
                    logits = blva_t + d_va
                    y01 = torch.clamp(yva_t + torch.sigmoid(blva_t), 0.0, 1.0)
                    v = F.binary_cross_entropy_with_logits(logits, y01).item()

            if v < best - self.cfg.min_delta:
                best = v
                best_state = {k: v_.detach().cpu().clone() for k, v_ in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.m_model = m_model
        self.e_model = e_model
        return self

    @torch.no_grad()
    def predict_binT(self, X):
        """
        Returns mu0_hat, mu1_hat (or p0_hat, p1_hat if binY)
        """
        import numpy as np
        task = self.cfg.task
        assert self._is_binT(task)

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)

        if self._is_binY(task):
            p_hat = self.m_model.predict_proba(X)[:, 1].astype(np.float32)
            base_logit = logit_np(p_hat)
            e_hat = self.e_model.predict_proba(X)[:, 1].astype(np.float32)
            e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

            t0 = np.zeros((X.shape[0],), dtype=np.float32)
            t1 = np.ones((X.shape[0],), dtype=np.float32)

            d0 = self.net(X_t,
                          torch.from_numpy(t0).to(self.device),
                          torch.from_numpy((t0 - e_hat).astype(np.float32)).to(self.device)
                          ).cpu().numpy().astype(np.float32)
            d1 = self.net(X_t,
                          torch.from_numpy(t1).to(self.device),
                          torch.from_numpy((t1 - e_hat).astype(np.float32)).to(self.device)
                          ).cpu().numpy().astype(np.float32)

            p0 = sigmoid_np(base_logit + d0).astype(np.float32)
            p1 = sigmoid_np(base_logit + d1).astype(np.float32)
            return np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)

        # contY
        m_hat = self.m_model.predict(X).astype(np.float32)
        e_hat = self.e_model.predict_proba(X)[:, 1].astype(np.float32)
        e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

        t0 = np.zeros((X.shape[0],), dtype=np.float32)
        t1 = np.ones((X.shape[0],), dtype=np.float32)

        y0 = self.net(X_t,
                      torch.from_numpy(t0).to(self.device),
                      torch.from_numpy((t0 - e_hat).astype(np.float32)).to(self.device)
                      ).cpu().numpy().astype(np.float32)
        y1 = self.net(X_t,
                      torch.from_numpy(t1).to(self.device),
                      torch.from_numpy((t1 - e_hat).astype(np.float32)).to(self.device)
                      ).cpu().numpy().astype(np.float32)
        return (m_hat + y0).astype(np.float32), (m_hat + y1).astype(np.float32)

    @torch.no_grad()
    def predict_contT_grid(self, X, t_grid: np.ndarray):
        """
        Returns mu_hat on grid: (n,G), (prob grid if binY)
        """
        task = self.cfg.task
        assert self._is_contT(task)

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)

        if self._is_binY(task):
            p_hat = self.m_model.predict_proba(X)[:, 1].astype(np.float32)
            base_logit = logit_np(p_hat)
            e_hat = self.e_model.predict(X).astype(np.float32)

            out = []
            for t in t_grid:
                tt = np.full((X.shape[0],), float(t), dtype=np.float32)
                ttil = (tt - e_hat).astype(np.float32)
                delta = self.net(X_t,
                                 torch.from_numpy(tt).to(self.device),
                                 torch.from_numpy(ttil).to(self.device)
                                 ).cpu().numpy().astype(np.float32)
                prob = sigmoid_np(base_logit + delta).astype(np.float32)
                out.append(np.clip(prob, 1e-6, 1 - 1e-6))
            return np.stack(out, axis=1)

        # contY
        m_hat = self.m_model.predict(X).astype(np.float32)
        e_hat = self.e_model.predict(X).astype(np.float32)

        out = []
        for t in t_grid:
            tt = np.full((X.shape[0],), float(t), dtype=np.float32)
            ttil = (tt - e_hat).astype(np.float32)
            ytil = self.net(X_t,
                            torch.from_numpy(tt).to(self.device),
                            torch.from_numpy(ttil).to(self.device)
                            ).cpu().numpy().astype(np.float32)
            out.append((m_hat + ytil).astype(np.float32))
        return np.stack(out, axis=1)
