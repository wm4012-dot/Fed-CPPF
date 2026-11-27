"""Public-probe personalized potential modeling and OOF fusion calibration (Fed-CPPF).

Data-sharing constraints:
- Clients upload only scale-invariant functional summaries derived from their
  local archive:
    1) `poly_w` (2*D + 1 floats): elite-weighted separable quadratic trend fit
       to a rank + log-IQR hybrid target.
    2) `pub_pred` (M floats): local surrogate predictions evaluated strictly at
       a deterministic public scrambled Sobol probe grid X_pub in [0, 1]^D.
- Raw evaluated points `x`, raw objective values `y`, and private kernel centers
  never leave the client.
- On the server, the personalized potential model for target client `i` excludes
  client `i` from parameter aggregation (Leave-Target-Out), using client `i`
  only to measure peer Spearman affinity on X_pub.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc, spearmanr

MODEL_KIND = "server_mtm_poly_rbf_v3"
TOL = 1e-12


def grid_key(grid: dict | None) -> tuple[int, int, int] | None:
    if not isinstance(grid, dict):
        return None
    try:
        seed = int(grid["seed"])
        size = int(grid["size"])
        dim = int(grid["dim"])
    except (KeyError, TypeError, ValueError):
        return None
    if size < 16 or dim < 1:
        return None
    return seed, size, dim


@lru_cache(maxsize=32)
def _sobol_grid_array(seed: int, size: int, dim: int) -> np.ndarray:
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    pts = np.asarray(sampler.random(size), dtype=float)
    pts.setflags(write=False)
    return pts


@lru_cache(maxsize=32)
def _public_kernel_factor(
    key: tuple[int, int, int], ridge: float
) -> tuple[float, tuple[np.ndarray, bool]]:
    x_pub = _sobol_grid_array(*key)
    d2 = cdist(x_pub, x_pub, metric="sqeuclidean")
    pos = d2[d2 > 0.0]
    bw2 = max(float(np.median(pos)) if pos.size else 1.0, key[2] / 8.0)
    k_mat = np.exp(-d2 / (2.0 * bw2))
    k_mat.flat[:: x_pub.shape[0] + 1] += float(max(1e-6, ridge))
    return bw2, cho_factor(k_mat, lower=True)


def rank01(values: np.ndarray) -> np.ndarray:
    """Average-tie rank transform mapped to [0, 1] (lower y -> lower rank)."""
    vals = np.asarray(values, dtype=float).reshape(-1)
    n = vals.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=float)
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    ranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / float(n - 1)


def scale_invariant_target(y_raw: np.ndarray, mode: str = "hybrid") -> tuple[np.ndarray, np.ndarray]:
    """Scale-invariant monotone target (invariant to positive affine rescaling of y).

    Returns `(target, r)` where `r = rank01(y_raw)`.
    - `rank`: centered rank `r - 0.5`
    - `log`: standardized `log1p((y - y_min) / IQR)` (preserves convex basin curvature)
    - `hybrid`: 50/50 blend of standardized rank and standardized log-IQR excess
    """
    y = np.asarray(y_raw, dtype=float).reshape(-1)
    r = rank01(y)
    mode_l = str(mode).strip().lower()
    if mode_l == "rank":
        return r - 0.5, r

    q25, q75 = np.percentile(y, [25.0, 75.0])
    iqr = max(float(q75 - q25), float(np.std(y)) * 0.1, TOL)
    z_log = np.log1p(np.maximum(0.0, y - float(np.min(y))) / iqr)
    z_log_sd = float(np.std(z_log))
    if z_log_sd <= TOL:
        return r - 0.5, r
    z_log = (z_log - float(np.mean(z_log))) / z_log_sd

    if mode_l == "log":
        return z_log, r

    r_sd = float(np.std(r))
    if r_sd <= TOL:
        return z_log, r
    z_rk = (r - float(np.mean(r))) / r_sd
    return 0.5 * z_rk + 0.5 * z_log, r


def poly_features(x: np.ndarray) -> np.ndarray:
    """Separable quadratic trend basis [1, x_1..x_D, x_1^2..x_D^2]."""
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    return np.hstack((np.ones((arr.shape[0], 1), dtype=float), arr, arr * arr))


def fit_client_private_summary(
    x_norm: np.ndarray,
    y_raw: np.ndarray,
    grid: dict,
    *,
    target_mode: str = "hybrid",
    poly_ridge: float = 0.05,
    rbf_ridge: float = 0.03,
    elite_weighting: bool = True,
) -> dict | None:
    """Fit a scale-invariant local surrogate and return ONLY privacy-safe outputs.

    Neither x_norm nor y_raw nor local kernel centers leave the client.
    """
    key = grid_key(grid)
    if key is None:
        return None
    x = np.atleast_2d(np.asarray(x_norm, dtype=float))
    y = np.asarray(y_raw, dtype=float).reshape(-1)
    n, dim = x.shape
    if dim != key[2] or n < max(12, dim + 2) or y.shape[0] != n:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.ptp(y) <= TOL:
        return None

    x_pub = _sobol_grid_array(*key)
    yr, r = scale_invariant_target(y, mode=target_mode)
    if elite_weighting:
        # Mild elite emphasis preserves global conditioning even at late FE90-110
        sw = 1.0 + 1.5 * np.exp(-3.0 * r)
    else:
        sw = np.ones(n, dtype=float)

    # 1. Elite-weighted separable quadratic trend (2*D + 1 floats)
    phi = poly_features(x)
    w_phi = phi * sw[:, None]
    reg = float(max(1e-6, poly_ridge)) * np.eye(phi.shape[1], dtype=float)
    reg[0, 0] = 1e-8
    try:
        w_poly = np.linalg.solve(phi.T @ w_phi + reg, w_phi.T @ yr)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(w_poly)):
        return None

    # 2. Local residual RBF evaluated ONLY at the fixed public Sobol grid X_pub
    res = yr - phi @ w_poly
    alpha_local = None
    bw2 = dim / 8.0
    try:
        d2 = cdist(x, x, metric="sqeuclidean")
        pos = d2[d2 > 0.0]
        bw2 = max(float(np.median(pos)) if pos.size else 1.0, dim / 8.0)
        k_mat = np.exp(-d2 / (2.0 * bw2))
        k_mat.flat[:: n + 1] += float(max(1e-6, rbf_ridge))
        alpha_local = cho_solve(cho_factor(k_mat, lower=True), res)
        if not np.all(np.isfinite(alpha_local)):
            alpha_local = None
    except (np.linalg.LinAlgError, ValueError):
        alpha_local = None

    pub_pred = poly_features(x_pub) @ w_poly
    if alpha_local is not None:
        k_q = np.exp(-cdist(x_pub, x, metric="sqeuclidean") / (2.0 * bw2))
        pub_pred = pub_pred + k_q @ alpha_local
    if not np.all(np.isfinite(pub_pred)) or float(np.std(pub_pred)) <= TOL:
        return None

    return {
        "grid": dict(zip(("seed", "size", "dim"), key)),
        "poly_w": w_poly.tolist(),
        "pub_pred": pub_pred.tolist(),
    }


def _valid_summary(summary, key: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(summary, dict) or grid_key(summary.get("grid")) != key:
        return None
    try:
        w_poly = np.asarray(summary["poly_w"], dtype=float).reshape(-1)
        pub_pred = np.asarray(summary["pub_pred"], dtype=float).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        w_poly.shape != (2 * key[2] + 1,)
        or pub_pred.shape != (key[1],)
        or not np.all(np.isfinite(w_poly))
        or not np.all(np.isfinite(pub_pred))
        or float(np.std(pub_pred)) <= TOL
    ):
        return None
    return w_poly, pub_pred


def _optimize_server_proposal(
    w_poly: np.ndarray,
    alpha: np.ndarray,
    centers: np.ndarray,
    bw2: float,
    y_centers: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract analytic quadratic consensus optimum x_quad and regularized L-BFGS-B x_srv."""
    a = w_poly[1 : dim + 1]
    b = w_poly[dim + 1 :]
    x_quad = np.where(
        b > 1e-6,
        np.clip(-a / (2.0 * np.maximum(b, 1e-6)), 0.0, 1.0),
        np.where(a + b < 0.0, 1.0, 0.0),
    )
    has_rbf = bool(np.any(alpha != 0.0) and centers.shape[0] > 0)

    def _val_and_grad(z: np.ndarray) -> tuple[float, np.ndarray]:
        val = float(w_poly[0] + np.dot(a, z) + np.dot(b, z * z))
        grad = a + 2.0 * b * z
        if has_rbf:
            diff = z[None, :] - centers
            d2 = np.sum(diff * diff, axis=1)
            k = np.exp(-d2 / (2.0 * bw2))
            wk = alpha * k
            val += float(np.sum(wk))
            grad = grad - (wk[:, None] * diff).sum(axis=0) / bw2
        return val, grad

    best_x = x_quad.copy()
    best_v, _ = _val_and_grad(best_x)
    starts = [x_quad]
    if centers.shape[0] > 0 and y_centers.shape[0] == centers.shape[0]:
        starts.append(np.clip(centers[int(np.argmin(y_centers))], 0.0, 1.0))

    bounds = [(0.0, 1.0)] * dim
    for x0 in starts:
        try:
            res = minimize(
                _val_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 35, "ftol": 1e-9},
            )
            if np.all(np.isfinite(res.x)) and float(res.fun) < best_v:
                best_v = float(res.fun)
                best_x = np.clip(res.x, 0.0, 1.0)
        except Exception:
            continue
    return best_x, x_quad


def server_fit_personalized_model(
    target_cid: int,
    summaries_by_cid: dict[int, dict],
    *,
    version: int,
    topk: int = 5,
    min_sim: float = 0.40,
    srv_ridge: float = 0.25,
    similarity_weighting: bool = True,
    residual_enabled: bool = True,
    proposal_enabled: bool = True,
    shuffle_summary: bool = False,
) -> dict | None:
    """Train a peer-only personalized semi-parametric potential model & proposal on the server.

    Target client `target_cid` is used ONLY to compute peer functional Spearman
    affinity on X_pub; its own trend weights and predictions are strictly
    excluded from the fitted model (zero-leakage leave-target-out construction).
    """
    target_summary = summaries_by_cid.get(int(target_cid))
    if not isinstance(target_summary, dict):
        return None
    key = grid_key(target_summary.get("grid"))
    if key is None:
        return None
    parsed_target = _valid_summary(target_summary, key)
    if parsed_target is None:
        return None
    _, target_pub = parsed_target

    peers = []
    for cid, raw in sorted(summaries_by_cid.items()):
        if int(cid) == int(target_cid):
            continue
        parsed = _valid_summary(raw, key)
        if parsed is None:
            continue
        w_k, pub_k = parsed
        rho = float(spearmanr(target_pub, pub_k).statistic)
        if np.isfinite(rho) and rho >= float(min_sim):
            peers.append((rho, int(cid), w_k, pub_k))

    if not peers:
        return None
    peers.sort(key=lambda item: (-item[0], item[1]))
    peers = peers[: max(1, int(topk))]

    sims = np.asarray([p[0] for p in peers], dtype=float)
    raw_w = (sims * sims) if similarity_weighting else np.ones_like(sims)
    w_sum = float(np.sum(raw_w))
    if not np.isfinite(w_sum) or w_sum <= TOL:
        return None
    weights = raw_w / w_sum

    dim = key[2]
    x_pub = _sobol_grid_array(*key)
    w_poly_srv = np.zeros(2 * dim + 1, dtype=float)
    pub_cons = np.zeros(key[1], dtype=float)

    for wk, (_, _, w_k, pub_k) in zip(weights, peers):
        mu = float(np.mean(pub_k))
        sd = float(np.std(pub_k))
        if sd <= TOL:
            continue
        pub_cons += wk * ((pub_k - mu) / sd)
        w_std = w_k.copy()
        w_std[0] = (w_std[0] - mu) / sd
        w_std[1:] = w_std[1:] / sd
        w_poly_srv += wk * w_std

    if shuffle_summary:
        # Causal counterfactual: preserve exact peer selection, marginal distribution,
        # and packet schema, while randomly permuting spatial correspondence on X_pub.
        rng_shuf = np.random.default_rng(
            20261122 + 1009 * int(target_cid) + 37 * int(version)
        )
        pub_cons = rng_shuf.permutation(pub_cons)
        phi_pub = poly_features(x_pub)
        reg_p = 0.05 * np.eye(phi_pub.shape[1], dtype=float)
        reg_p[0, 0] = 1e-8
        try:
            w_poly_srv = np.linalg.solve(phi_pub.T @ phi_pub + reg_p, phi_pub.T @ pub_cons)
        except np.linalg.LinAlgError:
            w_poly_srv = np.zeros(2 * dim + 1, dtype=float)

    if residual_enabled:
        res_all = pub_cons - poly_features(x_pub) @ w_poly_srv
        bw2, factor = _public_kernel_factor(key, float(max(1e-6, srv_ridge)))
        try:
            alpha_srv = cho_solve(factor, res_all)
        except np.linalg.LinAlgError:
            alpha_srv = np.zeros(key[1], dtype=float)
        if not np.all(np.isfinite(alpha_srv)):
            alpha_srv = np.zeros(key[1], dtype=float)
    else:
        bw2 = max(1.0, dim / 8.0)
        alpha_srv = np.zeros(key[1], dtype=float)

    eff_sim = float(np.sum(weights * sims))
    top1_sim = float(sims[0])
    peer_conf = float(np.clip(eff_sim * (0.55 + 0.45 * min(1.0, len(peers) / 3.0)), 0.0, 1.0))

    x_srv_list = None
    x_quad_list = None
    if proposal_enabled:
        x_srv, x_quad = _optimize_server_proposal(
            w_poly_srv, alpha_srv, x_pub, float(bw2), pub_cons, dim
        )
        x_srv_list = x_srv.tolist()
        x_quad_list = x_quad.tolist()

    return {
        "kind": MODEL_KIND,
        "target_cid": int(target_cid),
        "model_version": int(version),
        "grid": dict(zip(("seed", "size", "dim"), key)),
        "source_cids": [p[1] for p in peers],
        "peer_sims": [round(float(s), 4) for s in sims],
        "peer_weights": [round(float(w), 4) for w in weights],
        "top1_sim": round(top1_sim, 4),
        "eff_sim": round(eff_sim, 4),
        "peer_confidence": round(peer_conf, 4),
        "w_poly": w_poly_srv.tolist(),
        "alpha": alpha_srv.tolist(),
        "bw2": float(bw2),
        "x_srv": x_srv_list,
        "x_quad": x_quad_list,
    }


def predict_server_model(packet: dict | None, x_norm: np.ndarray) -> np.ndarray | None:
    """Evaluate the server's personalized multi-task potential on normalized x in [0, 1]^D."""
    if not isinstance(packet, dict) or packet.get("kind") != MODEL_KIND:
        return None
    key = grid_key(packet.get("grid"))
    if key is None:
        return None
    try:
        w_poly = np.asarray(packet["w_poly"], dtype=float).reshape(-1)
        alpha = np.asarray(packet["alpha"], dtype=float).reshape(-1)
        bw2 = float(packet["bw2"])
        x = np.atleast_2d(np.asarray(x_norm, dtype=float))
    except (KeyError, TypeError, ValueError):
        return None
    if (
        w_poly.shape != (2 * key[2] + 1,)
        or alpha.shape != (key[1],)
        or x.ndim != 2
        or x.shape[1] != key[2]
        or not np.isfinite(bw2)
        or bw2 <= 0.0
        or not np.all(np.isfinite(w_poly))
        or not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(x))
    ):
        return None

    scores = poly_features(x) @ w_poly
    if np.any(alpha != 0.0):
        x_pub = _sobol_grid_array(*key)
        k_q = np.exp(-cdist(x, x_pub, metric="sqeuclidean") / (2.0 * bw2))
        scores = scores + k_q @ alpha
    if not np.all(np.isfinite(scores)):
        return None
    return scores


def pairwise_concordance(scores: np.ndarray, truth: np.ndarray) -> float:
    """Pairwise ranking concordance (lower score predicts lower truth)."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    n = truth.shape[0]
    if n < 2 or scores.shape[0] != n:
        return 0.5
    idx_i, idx_j = np.triu_indices(n, k=1)
    dt = truth[idx_i] - truth[idx_j]
    keep = np.abs(dt) > TOL
    if not np.any(keep):
        return 0.5
    ds = scores[idx_i[keep]] - scores[idx_j[keep]]
    dt = dt[keep]
    good = np.where(np.abs(ds) <= TOL, 0.5, (ds * dt > 0.0).astype(float))
    return float(np.mean(good))


def top_vs_all_concordance(scores: np.ndarray, truth: np.ndarray, top_frac: float = 0.25) -> float:
    """Macro-basin concordance: pairwise accuracy of top-frac points against all points."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    n = truth.shape[0]
    if n < 4 or scores.shape[0] != n:
        return 0.5
    k = max(4, min(n - 1, int(round(float(top_frac) * n))))
    top_idx = np.argsort(truth, kind="mergesort")[:k]
    dt = truth[top_idx, None] - truth[None, :]
    keep = dt < -TOL
    if not np.any(keep):
        return 0.5
    ds = scores[top_idx, None] - scores[None, :]
    good = (ds[keep] < -TOL).astype(float) + 0.5 * (np.abs(ds[keep]) <= TOL).astype(float)
    return float(np.mean(good))


def calibrate_fusion_weight(
    local_oof: np.ndarray,
    server_scores: np.ndarray,
    y_archive: np.ndarray,
    *,
    max_weight: float = 0.50,
    peer_confidence: float = 1.0,
    min_srv_top_acc: float = 0.55,
    min_gain: float = 0.005,
    top_frac: float = 0.25,
    elite_k: int = 12,
    elite_weight: float = 0.55,
    elite_tol: float = 0.005,
) -> tuple[float, dict]:
    """Client-side dual-scale (macro top-25% + fine elite top-12) OOF calibration.

    Measures whether the peer-only server potential improves both macro-basin
    identification and fine-basin ranking among the client's top `elite_k` points.
    Reduces lambda toward 0 once local elite ranking outperforms the peer consensus
    inside the client's refined basin.
    """
    loc = np.asarray(local_oof, dtype=float).reshape(-1)
    srv = np.asarray(server_scores, dtype=float).reshape(-1)
    y = np.asarray(y_archive, dtype=float).reshape(-1)
    n = y.shape[0]
    w_e = float(np.clip(elite_weight, 0.0, 1.0))
    w_m = 1.0 - w_e

    diag = {
        "lam": 0.0,
        "reason": "invalid_input",
        "eff_max": 0.0,
        "srv_top_acc": 0.5,
        "srv_elite_acc": 0.5,
        "loc_top_acc": 0.5,
        "loc_elite_acc": 0.5,
        "fus_top_acc": 0.5,
        "fus_elite_acc": 0.5,
        "dual_gain": 0.0,
    }
    if (
        n < 16
        or loc.shape[0] != n
        or srv.shape[0] != n
        or not np.all(np.isfinite(loc))
        or not np.all(np.isfinite(srv))
        or not np.all(np.isfinite(y))
    ):
        return 0.0, diag

    loc_sd = float(np.std(loc))
    srv_sd = float(np.std(srv))
    if loc_sd <= TOL or srv_sd <= TOL:
        diag["reason"] = "constant_scores"
        return 0.0, diag

    loc_z = (loc - float(np.mean(loc))) / loc_sd
    srv_z = (srv - float(np.mean(srv))) / srv_sd

    eff_max = float(np.clip(min(max_weight, peer_confidence), 0.0, 0.90))
    ek = min(n, max(8, int(elite_k)))
    elite_idx = np.argsort(y, kind="mergesort")[:ek]

    srv_top = top_vs_all_concordance(srv_z, y, top_frac=top_frac)
    srv_elite = pairwise_concordance(srv_z[elite_idx], y[elite_idx])
    loc_top = top_vs_all_concordance(loc_z, y, top_frac=top_frac)
    loc_elite = pairwise_concordance(loc_z[elite_idx], y[elite_idx])
    base_obj = w_m * loc_top + w_e * loc_elite

    diag.update({
        "eff_max": eff_max,
        "srv_top_acc": srv_top,
        "srv_elite_acc": srv_elite,
        "loc_top_acc": loc_top,
        "loc_elite_acc": loc_elite,
        "fus_top_acc": loc_top,
        "fus_elite_acc": loc_elite,
    })
    if eff_max < 0.04:
        diag["reason"] = "low_peer_confidence"
        return 0.0, diag
    if srv_top < float(min_srv_top_acc):
        diag["reason"] = "low_server_macro_concordance"
        return 0.0, diag

    n_steps = max(2, int(round(eff_max / 0.04)))
    grid = np.linspace(0.04, eff_max, n_steps)
    best_lam = 0.0
    best_obj = base_obj
    best_top, best_elite = loc_top, loc_elite

    for lam in grid:
        lam_f = float(lam)
        comb = (1.0 - lam_f) * loc_z + lam_f * srv_z
        c_top = top_vs_all_concordance(comb, y, top_frac=top_frac)
        c_elite = pairwise_concordance(comb[elite_idx], y[elite_idx])
        if c_top < loc_top - 1e-9 or c_elite < loc_elite - float(elite_tol):
            continue
        obj = w_m * c_top + w_e * c_elite
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_lam = lam_f
            best_top, best_elite = c_top, c_elite

    gain = best_obj - base_obj
    if best_lam <= 0.0 or gain < float(min_gain):
        diag["reason"] = "insufficient_oof_gain"
        return 0.0, diag

    diag.update({
        "lam": best_lam,
        "reason": "accepted",
        "fus_top_acc": best_top,
        "fus_elite_acc": best_elite,
        "dual_gain": gain,
    })
    return best_lam, diag


class FusedPotentialRanker:
    """Antisymmetric transitive scalar-potential ranker fusing local & server models."""

    def __init__(
        self,
        local_ranker,
        server_packet: dict,
        lam: float,
        loc_mu: float,
        loc_sd: float,
        srv_mu: float,
        srv_sd: float,
    ):
        self.local_ranker = local_ranker
        self.server_packet = server_packet
        self.lam = float(np.clip(lam, 0.0, 1.0))
        self.loc_mu = float(loc_mu)
        self.loc_sd = max(float(loc_sd), 1e-8)
        self.srv_mu = float(srv_mu)
        self.srv_sd = max(float(srv_sd), 1e-8)
        self.input_dim = int(local_ranker.input_dim)
        self.epochs_run = getattr(local_ranker, "epochs_run", 0)
        self.final_loss = getattr(local_ranker, "final_loss", float("nan"))
        self.training_accuracy = getattr(local_ranker, "training_accuracy", float("nan"))

    def score_points(self, x: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(x, dtype=float))
        s_loc = (np.asarray(self.local_ranker.score_points(pts), dtype=float) - self.loc_mu) / self.loc_sd
        if self.lam <= 0.0:
            return s_loc
        s_srv_raw = predict_server_model(self.server_packet, pts)
        if s_srv_raw is None:
            return s_loc
        s_srv = (s_srv_raw - self.srv_mu) / self.srv_sd
        return (1.0 - self.lam) * s_loc + self.lam * s_srv

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 * self.input_dim:
            raise ValueError("invalid pair dimension")
        return self.score_points(arr[:, self.input_dim :]) - self.score_points(
            arr[:, : self.input_dim]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
