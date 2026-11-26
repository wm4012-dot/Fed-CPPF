"""Local antisymmetric scalar-potential ranker, CSO optimizer, and Rank-KNN arm."""

from __future__ import annotations

import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .model import rank01

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class AntisymmetricPotentialMLP:
    """Small tanh scalar-potential network trained with antisymmetric pairwise loss.

    For any pair (x_a, x_b) where y_a < y_b (label y_ab = +1), the margin is
    d_loc(x_a, x_b) = s_loc(x_b) - s_loc(x_a), so smaller s_loc(x) denotes a
    more promising solution and reversing a pair negates the decision exactly.
    Pure NumPy implementation (zero PyTorch dependency).
    """

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, int] = (32, 16),
        learning_rate: float = 1.0e-2,
        l2: float = 1.0e-4,
        epochs: int = 60,
        batch_size: int = 512,
        patience: int = 10,
        min_delta: float = 1.0e-5,
        grad_clip: float = 5.0,
        seed: int = 20260812,
    ):
        self.input_dim = int(input_dim)
        self.hidden = (max(2, int(hidden[0])), max(2, int(hidden[1])))
        self.learning_rate = max(1.0e-6, float(learning_rate))
        self.l2 = max(0.0, float(l2))
        self.epochs = max(1, int(epochs))
        self.batch_size = max(1, int(batch_size))
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.grad_clip = max(0.0, float(grad_clip))
        self.seed = int(seed)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._params: dict[str, np.ndarray] | None = None
        self.epochs_run: int = 0
        self.final_loss: float = float("nan")

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        out = np.empty_like(values)
        pos = values >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
        ev = np.exp(values[~pos])
        out[~pos] = ev / (1.0 + ev)
        return out

    def _initialize(self, rng: np.random.Generator) -> None:
        h1, h2 = self.hidden
        s1 = np.sqrt(2.0 / max(1, self.input_dim + h1))
        s2 = np.sqrt(2.0 / max(1, h1 + h2))
        s3 = np.sqrt(2.0 / max(1, h2 + 1))
        self._params = {
            "w1": rng.normal(0.0, s1, size=(self.input_dim, h1)),
            "b1": np.zeros(h1, dtype=float),
            "w2": rng.normal(0.0, s2, size=(h1, h2)),
            "b2": np.zeros(h2, dtype=float),
            "w3": rng.normal(0.0, s3, size=(h2, 1)),
            "b3": np.zeros(1, dtype=float),
        }

    def _forward_normalized(self, x: np.ndarray):
        assert self._params is not None
        h1 = np.tanh(x @ self._params["w1"] + self._params["b1"])
        h2 = np.tanh(h1 @ self._params["w2"] + self._params["b2"])
        score = h2 @ self._params["w3"] + self._params["b3"]
        return score.reshape(-1), h1, h2

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._scale is not None
        return (np.asarray(x, dtype=float) - self._mean) / self._scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AntisymmetricPotentialMLP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] != 2 * self.input_dim:
            raise ValueError("Invalid pairwise training shape.")
        keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & (y != 0.0)
        X = X[keep]
        y = np.where(y[keep] > 0.0, 1.0, -1.0)
        if X.shape[0] < 2 or np.unique(y).size < 2:
            raise ValueError("Pairwise training requires both positive and negative classes.")

        left = X[:, : self.input_dim]
        right = X[:, self.input_dim :]
        pts = np.vstack((left, right))
        self._mean = np.mean(pts, axis=0)
        self._scale = np.std(pts, axis=0)
        self._scale = np.where(self._scale > 1.0e-8, self._scale, 1.0)
        left = self._normalize(left)
        right = self._normalize(right)

        rng = np.random.default_rng(self.seed)
        self._initialize(rng)
        assert self._params is not None
        first = {k: np.zeros_like(v) for k, v in self._params.items()}
        second = {k: np.zeros_like(v) for k, v in self._params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1.0e-8
        step = 0
        best_loss = float("inf")
        best_params = None
        stale = 0
        n_rows = y.shape[0]

        for epoch in range(self.epochs):
            order = rng.permutation(n_rows)
            loss_sum = 0.0
            seen = 0
            for start in range(0, n_rows, self.batch_size):
                idx = order[start : start + self.batch_size]
                lb, rb, yb = left[idx], right[idx], y[idx]
                both = np.vstack((lb, rb))
                scores, h1, h2 = self._forward_normalized(both)
                nb = idx.size
                # Margin: y_ab * (s_loc(x_b) - s_loc(x_a)) -> lower score = better point
                diff = scores[nb:] - scores[:nb]
                margin = yb * diff
                loss_sum += float(np.sum(np.logaddexp(0.0, -margin)))
                seen += nb

                grad_diff = -yb * self._sigmoid(-margin) / nb
                grad_score = np.concatenate((-grad_diff, grad_diff))[:, None]
                grads = {
                    "w3": h2.T @ grad_score + self.l2 * self._params["w3"],
                    "b3": np.sum(grad_score, axis=0),
                }
                grad_z2 = (grad_score @ self._params["w3"].T) * (1.0 - h2 * h2)
                grads["w2"] = h1.T @ grad_z2 + self.l2 * self._params["w2"]
                grads["b2"] = np.sum(grad_z2, axis=0)
                grad_z1 = (grad_z2 @ self._params["w2"].T) * (1.0 - h1 * h1)
                grads["w1"] = both.T @ grad_z1 + self.l2 * self._params["w1"]
                grads["b1"] = np.sum(grad_z1, axis=0)

                if self.grad_clip > 0.0:
                    norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
                    if norm > self.grad_clip:
                        scale = self.grad_clip / max(norm, 1.0e-12)
                        grads = {k: g * scale for k, g in grads.items()}

                step += 1
                for k, g in grads.items():
                    first[k] = beta1 * first[k] + (1.0 - beta1) * g
                    second[k] = beta2 * second[k] + (1.0 - beta2) * (g * g)
                    m_hat = first[k] / (1.0 - beta1**step)
                    v_hat = second[k] / (1.0 - beta2**step)
                    self._params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            epoch_loss = loss_sum / max(1, seen) + 0.5 * self.l2 * sum(
                float(np.sum(self._params[k] ** 2)) for k in ("w1", "w2", "w3")
            )
            self.epochs_run = epoch + 1
            if epoch_loss < best_loss - self.min_delta:
                best_loss = epoch_loss
                best_params = {k: v.copy() for k, v in self._params.items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_params is not None:
            self._params = best_params
        self.final_loss = float(best_loss)
        return self

    def score_points(self, x: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("Ranker is not fitted.")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        scores, _, _ = self._forward_normalized(self._normalize(x))
        return scores

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.score_points(X[:, self.input_dim :]) - self.score_points(
            X[:, : self.input_dim]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)


def build_gp_lcb_probes(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    n_probes: int,
    lcb_kappa: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate GP-LCB auxiliary probe points in [0, 1]^D for ordinal training."""
    dim = x_hist.shape[1]
    y_norm = rank01(y_hist)
    probes = rng.uniform(0.0, 1.0, size=(max(16, int(n_probes)), dim))
    try:
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(dim) * 0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=int(rng.integers(0, 10**6))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x_hist, y_norm)
            mu, sd = gp.predict(probes, return_std=True)
        pseudo_y = mu - float(lcb_kappa) * np.maximum(sd, 0.0)
    except Exception:
        d = cdist(probes, x_hist)
        nn = np.argmin(d, axis=1)
        pseudo_y = y_norm[nn]
    return probes, np.asarray(pseudo_y, dtype=float)


def build_pairwise_dataset(
    x_pts: np.ndarray,
    y_vals: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct antisymmetric pairs (x_a, x_b) excluding ties (|y_a - y_b| <= 1e-12)."""
    n = x_pts.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = y_vals[j_idx] - y_vals[i_idx]
    keep = np.abs(diff) > 1.0e-12
    i_idx, j_idx, diff = i_idx[keep], j_idx[keep], diff[keep]
    if i_idx.size == 0:
        d = x_pts.shape[1]
        return np.empty((0, 2 * d)), np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)

    # Randomly orient pairs so +/- classes are balanced
    flip = rng.random(i_idx.size) < 0.5
    a_idx = np.where(flip, j_idx, i_idx)
    b_idx = np.where(flip, i_idx, j_idx)
    labels = np.where(y_vals[a_idx] < y_vals[b_idx], 1.0, -1.0)

    if a_idx.size > max_pairs:
        sel = rng.choice(a_idx.size, size=max_pairs, replace=False)
        a_idx, b_idx, labels = a_idx[sel], b_idx[sel], labels[sel]

    X_pair = np.hstack((x_pts[a_idx], x_pts[b_idx]))
    return X_pair, labels, a_idx, b_idx


def fit_local_ranker_with_oof(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    *,
    n_folds: int = 3,
    n_probes: int = 100,
    lcb_kappa: float = 2.0,
    max_pairs: int = 10000,
    min_bacc: float = 0.60,
    seed: int = 20260812,
):
    """Fit full local AntisymmetricPotentialMLP and point-disjoint K-fold OOF models.

    Returns ``(full_model, oof_scores_on_x_hist, health_bacc, accepted)``.
    """
    rng = np.random.default_rng(seed)
    n_arch = x_hist.shape[0]
    probes, probe_y = build_gp_lcb_probes(x_hist, y_hist, n_probes, lcb_kappa, rng)
    y_arch_rank = rank01(y_hist)

    x_all = np.vstack((x_hist, probes))
    y_all = np.concatenate((y_arch_rank, probe_y))
    n_total = x_all.shape[0]

    X_full, y_full, a_idx, b_idx = build_pairwise_dataset(x_all, y_all, max_pairs, rng)
    if X_full.shape[0] < 32:
        return None, None, 0.0, False

    full_model = AntisymmetricPotentialMLP(input_dim=x_hist.shape[1], seed=seed).fit(X_full, y_full)

    # Point-disjoint K-fold partition of unique points
    perm = rng.permutation(n_total)
    fold_of = np.empty(n_total, dtype=int)
    for idx_pos, pt_id in enumerate(perm):
        fold_of[pt_id] = idx_pos % n_folds

    val_preds = np.zeros(X_full.shape[0], dtype=float)
    val_mask = np.zeros(X_full.shape[0], dtype=bool)
    fold_models: list[AntisymmetricPotentialMLP] = []

    for f in range(n_folds):
        tr_mask = (fold_of[a_idx] != f) & (fold_of[b_idx] != f)
        va_mask = (fold_of[a_idx] == f) & (fold_of[b_idx] == f)
        if np.sum(tr_mask) < 16 or np.sum(va_mask) < 4:
            return full_model, None, 0.0, False
        fm = AntisymmetricPotentialMLP(input_dim=x_hist.shape[1], seed=seed + 17 * (f + 1))
        fm.fit(X_full[tr_mask], y_full[tr_mask])
        fold_models.append(fm)
        val_preds[va_mask] = fm.predict(X_full[va_mask])
        val_mask |= va_mask

    if not np.any(val_mask):
        return full_model, None, 0.0, False
    vy = y_full[val_mask]
    vp = val_preds[val_mask]
    pos_m, neg_m = vy > 0.0, vy < 0.0
    if not np.any(pos_m) or not np.any(neg_m):
        return full_model, None, 0.0, False
    bacc = 0.5 * (float(np.mean(vp[pos_m] > 0.0)) + float(np.mean(vp[neg_m] < 0.0)))
    accepted = bool(np.isfinite(bacc) and bacc >= min_bacc)

    # Per-fold z-score aligned point-disjoint OOF predictions on x_hist
    oof_scores = None
    if accepted and len(fold_models) == n_folds:
        z_preds = []
        ok = True
        for fm in fold_models:
            s = fm.score_points(x_hist)
            sd = float(np.std(s))
            if sd <= 1e-12 or not np.all(np.isfinite(s)):
                ok = False
                break
            z_preds.append((s - float(np.mean(s))) / sd)
        if ok:
            oof_scores = np.array([z_preds[fold_of[i]][i] for i in range(n_arch)], dtype=float)

    return full_model, oof_scores, bacc, accepted


def cso_scalar_search(
    scorer,
    dim: int,
    *,
    pop_size: int = 100,
    iters: int = 100,
    phi: float = 0.1,
    seeds: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Competitive Swarm Optimizer using O(N_p) scalar potential evaluations per generation."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_p = max(16, int(pop_size))
    if n_p % 2 == 1:
        n_p += 1
    pos = rng.uniform(0.0, 1.0, size=(n_p, dim))
    if seeds is not None and seeds.size > 0:
        s_arr = np.clip(np.atleast_2d(seeds), 0.0, 1.0)
        k = min(s_arr.shape[0], n_p // 4)
        pos[:k] = s_arr[:k]
    vel = np.zeros_like(pos)

    for _ in range(max(1, int(iters))):
        scores = np.asarray(scorer.score_points(pos), dtype=float).reshape(-1)
        mean_pos = np.mean(pos, axis=0, keepdims=True)
        perm = rng.permutation(n_p)
        p1, p2 = perm[: n_p // 2], perm[n_p // 2 :]
        win_1 = scores[p1] <= scores[p2]
        winners = np.where(win_1, p1, p2)
        losers = np.where(win_1, p2, p1)

        r1 = rng.random((losers.size, dim))
        r2 = rng.random((losers.size, dim))
        r3 = rng.random((losers.size, dim))
        vel[losers] = (
            r1 * vel[losers]
            + r2 * (pos[winners] - pos[losers])
            + float(phi) * r3 * (mean_pos - pos[losers])
        )
        pos[losers] = np.clip(pos[losers] + vel[losers], 0.0, 1.0)

    final_scores = np.asarray(scorer.score_points(pos), dtype=float).reshape(-1)
    return pos[int(np.argmin(final_scores))].copy()


def rank_knn_scores(
    x_query: np.ndarray,
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    k: int = 7,
    kappa: float = 0.35,
) -> np.ndarray:
    """Rank-KNN LCB acquisition score in [0, 1]^D (lower is better)."""
    n = x_hist.shape[0]
    kk = min(max(1, int(k)), n)
    targets = rank01(y_hist)
    dists = cdist(np.atleast_2d(x_query), x_hist)
    nn_idx = np.argpartition(dists, kth=kk - 1, axis=1)[:, :kk]
    rows = np.arange(dists.shape[0])[:, None]
    dsel = dists[rows, nn_idx]
    w = 1.0 / np.maximum(dsel, 1e-12)
    ysel = targets[nn_idx]
    ymean = (w * ysel).sum(axis=1) / w.sum(axis=1)
    yvar = (w * (ysel - ymean[:, None]) ** 2).sum(axis=1) / w.sum(axis=1)
    return ymean - float(kappa) * np.sqrt(np.maximum(yvar, 1e-12))


def loo_rank_knn_scores(
    x_hist: np.ndarray, y_hist: np.ndarray, k: int = 7, kappa: float = 0.35
) -> np.ndarray | None:
    """Leave-one-out Rank-KNN scores on the local archive for OOF fusion calibration."""
    n = x_hist.shape[0]
    kk = min(max(1, int(k)), n - 1)
    if n < kk + 2:
        return None
    targets = rank01(y_hist)
    dists = cdist(x_hist, x_hist)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=kk - 1, axis=1)[:, :kk]
    rows = np.arange(n)[:, None]
    dsel = dists[rows, nn_idx]
    w = 1.0 / np.maximum(dsel, 1e-12)
    ysel = targets[nn_idx]
    ymean = (w * ysel).sum(axis=1) / w.sum(axis=1)
    yvar = (w * (ysel - ymean[:, None]) ** 2).sum(axis=1) / w.sum(axis=1)
    return ymean - float(kappa) * np.sqrt(np.maximum(yvar, 1e-12))
