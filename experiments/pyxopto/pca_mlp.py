"""PCA + MLP baseline for reflectance curves.

This class expects preprocessed inputs/targets:
- X: standardized continuous inputs
- Y: standardized log10 curves
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    out = {"float32": torch.float32, "float64": torch.float64}.get(str(dtype).lower())
    if out is None:
        raise ValueError(f"Unknown dtype {dtype!r} (expected 'float32' or 'float64')")
    return out


def _resolve_torch_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device)


ActivationName = Literal["silu", "relu"]


def _activation(name: ActivationName) -> torch.nn.Module:
    if name == "silu":
        return torch.nn.SiLU()
    if name == "relu":
        return torch.nn.ReLU()
    raise ValueError(f"Unknown activation {name!r} (expected 'silu' or 'relu')")


class _ScoreMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int, *, activation: ActivationName) -> None:
        super().__init__()
        width = int(hidden_width)
        act = _activation(activation)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, width),
            act,
            torch.nn.Linear(width, width),
            act,
            torch.nn.Linear(width, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PCAMLP:
    """PCA + two-hidden-layer MLP on [X, one_hot(task)] -> PCA scores."""

    grid_points: np.ndarray  # (N,2), standardized
    tasks: np.ndarray  # (N,)
    axis: np.ndarray  # (P,)
    eigencurves: np.ndarray  # (M,P)
    pca_mean: np.ndarray  # (P,)
    weights: np.ndarray  # (N,M)
    hidden_width: int = 128
    activation: ActivationName = "silu"
    dtype: str | torch.dtype = "float64"
    device: str | torch.device = "auto"

    torch_dtype: torch.dtype = field(init=False, repr=False)
    torch_device: torch.device = field(init=False, repr=False)
    _X_t: torch.Tensor = field(init=False, repr=False)
    _s_t: torch.Tensor = field(init=False, repr=False)
    _Xin_t: torch.Tensor = field(init=False, repr=False)
    _W_t: torch.Tensor = field(init=False, repr=False)
    _Phi_t: torch.Tensor = field(init=False, repr=False)
    _pca_mean_t: torch.Tensor = field(init=False, repr=False)
    _net: _ScoreMLP = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.torch_dtype = _resolve_torch_dtype(self.dtype)
        self.torch_device = _resolve_torch_device(self.device)
        self._X_t = torch.as_tensor(self.grid_points, dtype=self.torch_dtype, device=self.torch_device)
        self._s_t = torch.as_tensor(self.tasks, dtype=torch.long, device=self.torch_device)
        self._Xin_t = self._build_inputs_t(self._X_t, self._s_t)
        self._W_t = torch.as_tensor(self.weights, dtype=self.torch_dtype, device=self.torch_device)
        self._Phi_t = torch.as_tensor(self.eigencurves, dtype=self.torch_dtype, device=self.torch_device)
        self._pca_mean_t = torch.as_tensor(self.pca_mean, dtype=self.torch_dtype, device=self.torch_device)
        self._net = _ScoreMLP(
            self._Xin_t.shape[1], self.n_components, int(self.hidden_width), activation=self.activation
        ).to(
            device=self.torch_device, dtype=self.torch_dtype
        )

    @property
    def n_components(self) -> int:
        return int(self.eigencurves.shape[0])

    @property
    def n_tasks(self) -> int:
        return int(np.max(self.tasks)) + 1

    def _build_inputs_t(self, X_t: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(s_t, num_classes=self.n_tasks).to(dtype=self.torch_dtype)
        return torch.cat([X_t, one_hot], dim=1)

    @classmethod
    def from_curves(
        cls,
        *,
        grid_points: np.ndarray,
        tasks: np.ndarray,
        axis: np.ndarray,
        curves: np.ndarray,
        n_components: int = 25,
        hidden_width: int = 128,
        activation: ActivationName = "silu",
        dtype: str | torch.dtype = "float64",
        device: str | torch.device = "auto",
    ) -> "PCAMLP":
        grid_points = np.asarray(grid_points, dtype=float)
        tasks = np.asarray(tasks, dtype=int).reshape(-1)
        axis = np.asarray(axis, dtype=float)
        curves = np.asarray(curves, dtype=float)

        if grid_points.ndim != 2 or grid_points.shape[1] != 2:
            raise ValueError("grid_points must have shape (N,2) for (mua,musr)")
        if curves.ndim != 2 or curves.shape[0] != grid_points.shape[0]:
            raise ValueError("curves must have shape (N,P) matching grid_points")
        if tasks.shape[0] != grid_points.shape[0]:
            raise ValueError("tasks must have shape (N,) matching grid_points")
        if axis.ndim != 1 or axis.shape[0] != curves.shape[1]:
            raise ValueError("axis must have shape (P,) matching curves")
        if int(hidden_width) < 1:
            raise ValueError("hidden_width must be >= 1")

        pca = PCA(n_components=int(n_components), svd_solver="full")
        weights = pca.fit_transform(curves)
        eigencurves = pca.components_
        pca_mean = pca.mean_

        return cls(
            grid_points=grid_points,
            tasks=tasks,
            axis=axis,
            eigencurves=eigencurves,
            pca_mean=pca_mean,
            weights=weights,
            hidden_width=int(hidden_width),
            activation=activation,
            dtype=dtype,
            device=device,
        )

    def train(
        self,
        *,
        num_steps: int = 3000,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        seed: int = 0,
        val_X: np.ndarray | None = None,
        val_tasks: np.ndarray | None = None,
        val_Y: np.ndarray | None = None,
        val_y_std: np.ndarray | None = None,
        early_stop_metric: str = "mae_val",
        early_stop_patience_evals: int | None = None,
        early_stop_min_delta: float = 0.0,
        log_every: int = 100,
        verbose: bool = False,
    ) -> None:
        torch.manual_seed(int(seed))
        opt = torch.optim.AdamW(self._net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        patience = None if early_stop_patience_evals is None else int(early_stop_patience_evals)
        patience = None if patience is None or patience <= 0 else patience
        min_delta = float(early_stop_min_delta)

        Xin_v = Yv_t = y_std_t = None
        if patience is not None:
            if val_X is None or val_tasks is None or val_Y is None or val_y_std is None:
                raise ValueError("Early stopping requires val_X, val_tasks, val_Y, and val_y_std.")
            Xv = np.asarray(val_X, dtype=float)
            sv = np.asarray(val_tasks, dtype=int).reshape(-1)
            Yv = np.asarray(val_Y, dtype=float)
            y_std_v = np.asarray(val_y_std, dtype=float).reshape(-1)
            if Xv.ndim != 2 or Xv.shape[1] != 2:
                raise ValueError("val_X must have shape (Q,2)")
            if sv.shape[0] != Xv.shape[0] or Yv.shape[0] != Xv.shape[0]:
                raise ValueError("val_tasks and val_Y must match val_X rows")
            if Yv.shape[1] != self._pca_mean_t.shape[0] or y_std_v.shape[0] != Yv.shape[1]:
                raise ValueError("val_Y/val_y_std shape mismatch")
            Xv_t = torch.as_tensor(Xv, dtype=self.torch_dtype, device=self.torch_device)
            sv_t = torch.as_tensor(sv, dtype=torch.long, device=self.torch_device)
            Xin_v = self._build_inputs_t(Xv_t, sv_t)
            Yv_t = torch.as_tensor(Yv, dtype=self.torch_dtype, device=self.torch_device)
            y_std_t = torch.as_tensor(y_std_v, dtype=self.torch_dtype, device=self.torch_device)

        best_mae_val = float("inf")
        bad_evals = 0
        best_state = None

        metric = str(early_stop_metric).lower()
        metric = "mae_val" if metric in ("mae", "mae_val") else ("rmse_val" if metric in ("rmse", "rmse_val") else metric)
        if metric not in ("mae_val", "rmse_val"):
            raise ValueError("early_stop_metric must be 'mae_val' or 'rmse_val'")

        def eval_val_metric() -> float:
            assert Xin_v is not None and Yv_t is not None and y_std_t is not None
            with torch.no_grad():
                w_pred = self._net(Xin_v)
                y_pred = w_pred @ self._Phi_t + self._pca_mean_t[None, :]
                err = (y_pred - Yv_t) * y_std_t.unsqueeze(0)
                if metric == "mae_val":
                    return float(torch.mean(torch.abs(err)).detach().cpu().item())
                return float(torch.sqrt(torch.mean(err * err)).detach().cpu().item())

        best_val = float("inf")

        for step in range(int(num_steps)):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((self._net(self._Xin_t) - self._W_t) ** 2)
            loss.backward()
            opt.step()

            should_log = step % int(log_every) == 0 or step == int(num_steps) - 1
            if verbose and should_log:
                print(f"[PCAMLP/pyxopto] step={step}/{int(num_steps)} loss={float(loss.detach().cpu()):.6g}")

            if patience is None or not should_log:
                continue
            val_metric = eval_val_metric()
            if verbose:
                print(f"[PCAMLP/pyxopto] step={step}/{int(num_steps)} {metric}={val_metric:.6g}")
            if val_metric < best_val - min_delta:
                best_val = val_metric
                best_state = deepcopy(self._net.state_dict())
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    if verbose:
                        print(f"[PCAMLP/EARLY_STOP] stop at step={step} best_{metric}={best_val:.6g}")
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

    def predict_weights(self, X: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        Xq = np.asarray(X, dtype=float)
        sq = np.asarray(tasks, dtype=int).reshape(-1)
        if Xq.ndim != 2 or Xq.shape[1] != 2:
            raise ValueError("X must have shape (Q,2)")
        if sq.shape[0] != Xq.shape[0]:
            raise ValueError("tasks must have shape (Q,) matching X")

        Xq_t = torch.as_tensor(Xq, dtype=self.torch_dtype, device=self.torch_device)
        sq_t = torch.as_tensor(sq, dtype=torch.long, device=self.torch_device)
        Xin_t = self._build_inputs_t(Xq_t, sq_t)
        with torch.no_grad():
            return self._net(Xin_t).detach().cpu().numpy()

    def predict_curve(self, X: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        w = self.predict_weights(X, tasks)
        w_t = torch.as_tensor(np.atleast_2d(w), dtype=self.torch_dtype, device=self.torch_device)
        y = w_t @ self._Phi_t + self._pca_mean_t[None, :]
        return y.detach().cpu().numpy().squeeze()
