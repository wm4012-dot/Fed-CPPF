"""Client-side probe encoding, OOF potential calibration, and dual-arm search."""

from __future__ import annotations

import numpy as np

from .framework import Client, ClientPackage
from .model import (
    FusedPotentialRanker,
    calibrate_fusion_weight,
    fit_client_private_summary,
    predict_server_model,
    rank01,
)
from .ranker import (
    cso_scalar_search,
    fit_local_ranker_with_oof,
    loo_rank_knn_scores,
    rank_knn_scores,
)


class FedCPPFClient(Client):
    """Fed-CPPF client implementing Algorithm 1.

    Parameters match the default hyperparameter registry in the paper:
    - Public probe encoding: ``grid_size=128``, ``grid_seed=2026``,
      ``poly_ridge=0.05``, ``rbf_ridge=0.03``, ``elite_weighting=True``.
    - OOF fusion calibration: ``max_weight=0.50``, ``min_gain=0.005``,
      ``min_top_acc=0.55``, ``top_frac=0.25``, ``elite_k=12``,
      ``elite_weight=0.55``, ``elite_tol=0.005``, ``late_floor_ratio=0.15``.
    - Dual-arm search: Antisymmetric scalar-potential MLP + CSO (``pop_size=100``,
      ``cso_iters=100``, ``cso_phi=0.1``) paired with Rank-KNN (``knn_k=7``,
      ``knn_kappa=0.35``) selected via Thompson sampling.
    """

    def __init__(
        self,
        problem,
        *,
        enabled: bool = True,
        grid_size: int = 128,
        grid_seed: int = 2026,
        target_mode: str = "hybrid",
        poly_ridge: float = 0.05,
        rbf_ridge: float = 0.03,
        elite_weighting: bool = True,
        max_weight: float = 0.50,
        min_gain: float = 0.005,
        min_top_acc: float = 0.55,
        top_frac: float = 0.25,
        elite_k: int = 12,
        elite_weight: float = 0.55,
        elite_tol: float = 0.005,
        late_taper: bool = True,
        late_floor_ratio: float = 0.15,
        oof_calibration: bool = True,
        fixed_weight: float = 0.35,
        guide_ordinal: bool = True,
        guide_knn: bool = True,
        seed_injection: bool = True,
        seed_max_progress: float = 0.45,
        seed_min_conf: float = 0.45,
        seed_max_local_rank: float = 0.35,
        seed_min_dist: float = 0.08,
        seed_min_adv: float = 0.25,
        dual_arm_start: float = 0.45,
        ordinal_min_points: int = 50,
        n_probes: int = 100,
        lcb_kappa: float = 2.0,
        min_bacc: float = 0.60,
        cso_pop_size: int = 100,
        cso_iters: int = 100,
        cso_phi: float = 0.1,
        knn_k: int = 7,
        knn_kappa: float = 0.35,
        seed: int = 20260812,
    ):
        super().__init__(problem)
        self.enabled = bool(enabled)
        self.grid_size = max(16, int(grid_size))
        self.grid_seed = int(grid_seed)
        self.target_mode = str(target_mode)
        self.poly_ridge = float(poly_ridge)
        self.rbf_ridge = float(rbf_ridge)
        self.elite_weighting = bool(elite_weighting)
        self.max_weight = float(max_weight)
        self.min_gain = float(min_gain)
        self.min_top_acc = float(min_top_acc)
        self.top_frac = float(top_frac)
        self.elite_k = int(elite_k)
        self.elite_weight = float(elite_weight)
        self.elite_tol = float(elite_tol)
        self.late_taper = bool(late_taper)
        self.late_floor_ratio = float(late_floor_ratio)
        self.oof_calibration = bool(oof_calibration)
        self.fixed_weight = float(fixed_weight)
        self.guide_ordinal = bool(guide_ordinal)
        self.guide_knn = bool(guide_knn)
        self.seed_injection = bool(seed_injection)
        self.seed_max_progress = float(seed_max_progress)
        self.seed_min_conf = float(seed_min_conf)
        self.seed_max_local_rank = float(seed_max_local_rank)
        self.seed_min_dist = float(seed_min_dist)
        self.seed_min_adv = float(seed_min_adv)
        self.dual_arm_start = float(np.clip(dual_arm_start, 0.0, 1.0))
        self.ordinal_min_points = int(ordinal_min_points)
        self.n_probes = int(n_probes)
        self.lcb_kappa = float(lcb_kappa)
        self.min_bacc = float(min_bacc)
        self.cso_pop_size = int(cso_pop_size)
        self.cso_iters = int(cso_iters)
        self.cso_phi = float(cso_phi)
        self.knn_k = int(knn_k)
        self.knn_kappa = float(knn_kappa)
        self._rng = np.random.default_rng(int(seed) + 1009 * int(self.id))

        # Beta-Bernoulli Thompson sampling state for (Rank-KNN, Fused Ordinal CSO)
        self._ts_alpha = {"knn": 2.0, "ordinal": 1.0}
        self._ts_beta = {"knn": 1.0, "ordinal": 1.0}
        self._round = 0
        self.lambda_history: list[float] = []
        self.arm_history: list[str] = []

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        span = np.maximum(self.ub - self.lb, 1e-12)
        return np.clip((np.asarray(x, dtype=float) - self.lb) / span, 0.0, 1.0)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lb + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (self.ub - self.lb)

    def _progress(self) -> float:
        budget = max(1, int(self.fe_max) - int(self.fe_init))
        used = max(0, int(self.solutions.size) - int(self.fe_init))
        return float(np.clip(used / budget, 0.0, 1.0))

    def _local_knn_rank_pct(
        self, x_hist: np.ndarray, y_hist: np.ndarray, pt: np.ndarray
    ) -> float:
        """Inverse-distance KNN predicted rank percentile in [0, 1] (lower = better)."""
        k = min(max(3, int(self.knn_k)), x_hist.shape[0])
        dists = np.linalg.norm(x_hist - pt[None, :], axis=1)
        idx = np.argpartition(dists, k - 1)[:k]
        w = 1.0 / np.maximum(dists[idx], 1e-6)
        r = rank01(y_hist)
        return float(np.sum(w * r[idx]) / np.sum(w))

    def _gated_server_seeds(
        self,
        server_model: dict | None,
        lam: float,
        progress: float,
        x_hist: np.ndarray,
        y_hist: np.ndarray,
    ) -> np.ndarray | None:
        if (
            not self.seed_injection
            or server_model is None
            or lam <= 0.0
            or progress > self.seed_max_progress
        ):
            return None
        conf = float(server_model.get("peer_confidence", 0.0))
        if conf < self.seed_min_conf:
            return None
        srv_arch = predict_server_model(server_model, x_hist)
        if srv_arch is None:
            return None
        srv_sd = float(np.std(srv_arch))
        if srv_sd <= 1e-12 or not np.all(np.isfinite(srv_arch)):
            return None

        best_idx = int(np.argmin(y_hist))
        x_best = np.clip(x_hist[best_idx], 0.0, 1.0)
        srv_best = float(srv_arch[best_idx])

        raw_candidates: list[np.ndarray] = []
        for key in ("x_quad", "x_srv"):
            pt = server_model.get(key)
            if pt is None:
                continue
            arr = np.asarray(pt, dtype=float).reshape(-1)
            if arr.shape[0] == self.dim and np.all(np.isfinite(arr)):
                raw_candidates.append(np.clip(arr, 0.0, 1.0))
        if not raw_candidates:
            return None
        # Midpoint bridge between current best x_best and primary server optimum
        raw_candidates.append(np.clip(0.5 * x_best + 0.5 * raw_candidates[0], 0.0, 1.0))

        accepted: list[np.ndarray] = []
        for pt in raw_candidates:
            if any(float(np.linalg.norm(pt - prev)) < 0.04 for prev in accepted):
                continue
            d_min = float(np.min(np.linalg.norm(x_hist - pt[None, :], axis=1)))
            if d_min < self.seed_min_dist:
                continue
            if self._local_knn_rank_pct(x_hist, y_hist, pt) > self.seed_max_local_rank:
                continue
            sp = predict_server_model(server_model, pt[None, :])
            if sp is None or not np.isfinite(sp[0]):
                continue
            adv = (srv_best - float(sp[0])) / srv_sd
            if adv < self.seed_min_adv:
                continue
            accepted.append(pt)
        return np.asarray(accepted, dtype=float) if accepted else None

    def _calibrate_lam(
        self,
        local_oof: np.ndarray,
        srv_arch: np.ndarray,
        y_hist: np.ndarray,
        server_model: dict,
        progress: float,
    ) -> float:
        conf = float(server_model.get("peer_confidence", 0.0))
        eff_max = self.max_weight
        if self.late_taper:
            t = float(np.clip(progress, 0.0, 1.0))
            taper = 1.0 - (1.0 - self.late_floor_ratio) * (t * t)
            eff_max = eff_max * taper
        if not self.oof_calibration:
            return float(np.clip(min(self.fixed_weight, eff_max, conf), 0.0, 0.90))
        lam, _ = calibrate_fusion_weight(
            local_oof=local_oof,
            server_scores=srv_arch,
            y_archive=y_hist,
            max_weight=eff_max,
            peer_confidence=conf,
            min_srv_top_acc=self.min_top_acc,
            min_gain=self.min_gain,
            top_frac=self.top_frac,
            elite_k=self.elite_k,
            elite_weight=self.elite_weight,
            elite_tol=self.elite_tol,
        )
        return float(lam)

    def _propose_knn(
        self,
        x_hist: np.ndarray,
        y_hist: np.ndarray,
        server_model: dict | None,
        progress: float,
    ) -> tuple[np.ndarray, float]:
        # Trust-region + global candidate pool around incumbent
        best_x = x_hist[int(np.argmin(y_hist))]
        rad = 0.60 * (1.0 - progress) + 0.15 * progress
        n_loc, n_glob = 64, 32
        local_cands = np.clip(
            best_x + self._rng.normal(0.0, rad * 0.35, size=(n_loc, self.dim)),
            0.0,
            1.0,
        )
        glob_cands = self._rng.uniform(0.0, 1.0, size=(n_glob, self.dim))
        pool = np.vstack((local_cands, glob_cands))

        loc_pool = rank_knn_scores(pool, x_hist, y_hist, k=self.knn_k, kappa=self.knn_kappa)
        lam = 0.0
        if self.enabled and self.guide_knn and server_model is not None:
            loo_loc = loo_rank_knn_scores(x_hist, y_hist, k=self.knn_k, kappa=self.knn_kappa)
            srv_arch = predict_server_model(server_model, x_hist)
            srv_pool = predict_server_model(server_model, pool)
            if loo_loc is not None and srv_arch is not None and srv_pool is not None:
                lam = self._calibrate_lam(loo_loc, srv_arch, y_hist, server_model, progress)
                if lam > 0.0:
                    seeds = self._gated_server_seeds(server_model, lam, progress, x_hist, y_hist)
                    if seeds is not None and seeds.size > 0:
                        pool = np.vstack((pool, seeds))
                        loc_pool = rank_knn_scores(pool, x_hist, y_hist, k=self.knn_k, kappa=self.knn_kappa)
                        srv_pool = predict_server_model(server_model, pool)
                    loc_sd = max(float(np.std(loo_loc)), 1e-12)
                    srv_sd = max(float(np.std(srv_arch)), 1e-12)
                    loc_z = (loc_pool - float(np.mean(loo_loc))) / loc_sd
                    srv_z = (srv_pool - float(np.mean(srv_arch))) / srv_sd
                    fused = (1.0 - lam) * loc_z + lam * srv_z
                    return pool[int(np.argmin(fused))], lam
        return pool[int(np.argmin(loc_pool))], lam

    def optimize(self) -> None:
        """Execute one federated optimization round (Algorithm 1)."""
        if self.solutions.size >= self.fe_max:
            return
        self._round += 1
        progress = self._progress()
        x_hist = self._to_unit(self.solutions.x)
        y_hist = np.asarray(self.solutions.y, dtype=float).reshape(-1)
        prev_best = float(np.min(y_hist))

        # Step 1: Private probe-grid encoding & upload (never sends raw x, y, or centers)
        summary = None
        if self.enabled:
            grid = {"seed": self.grid_seed, "size": self.grid_size, "dim": self.dim}
            summary = fit_client_private_summary(
                x_norm=x_hist,
                y_raw=y_hist,
                grid=grid,
                target_mode=self.target_mode,
                poly_ridge=self.poly_ridge,
                rbf_ridge=self.rbf_ridge,
                elite_weighting=self.elite_weighting,
            )
        self.request_server(
            ClientPackage(cid=self.id, action="upload", version=self._round, data={"summary": summary})
        )

        # Step 2: Receive Leave-Target-Out personalized server model (never reuse stale model)
        resp = self.request_server(
            ClientPackage(cid=self.id, action="download", version=self._round)
        )
        server_model = resp.get("model") if isinstance(resp, dict) else None

        # Step 3: Fit local antisymmetric scalar-potential ranker + 3-fold point-disjoint OOF
        use_ordinal = False
        full_ranker = None
        oof_scores = None
        if progress >= self.dual_arm_start and x_hist.shape[0] >= self.ordinal_min_points:
            full_ranker, oof_scores, _, accepted = fit_local_ranker_with_oof(
                x_hist,
                y_hist,
                n_folds=3,
                n_probes=self.n_probes,
                lcb_kappa=self.lcb_kappa,
                min_bacc=self.min_bacc,
                seed=int(self._rng.integers(0, 10**7)),
            )
            use_ordinal = bool(accepted and full_ranker is not None)

        # Step 4: Thompson-sampling arm selection once progress >= dual_arm_start (0.45)
        if use_ordinal:
            draw_knn = float(self._rng.beta(self._ts_alpha["knn"], self._ts_beta["knn"]))
            draw_ord = float(self._rng.beta(self._ts_alpha["ordinal"], self._ts_beta["ordinal"]))
            chosen_arm = "ordinal" if draw_ord >= draw_knn else "knn"
        else:
            chosen_arm = "knn"

        lam = 0.0
        if chosen_arm == "ordinal" and full_ranker is not None:
            scorer = full_ranker
            if self.enabled and self.guide_ordinal and server_model is not None and oof_scores is not None:
                srv_arch = predict_server_model(server_model, x_hist)
                loc_arch = np.asarray(full_ranker.score_points(x_hist), dtype=float).reshape(-1)
                loc_sd = float(np.std(loc_arch))
                srv_sd = float(np.std(srv_arch)) if srv_arch is not None else 0.0
                if srv_arch is not None and loc_sd > 1e-12 and srv_sd > 1e-12:
                    lam = self._calibrate_lam(oof_scores, srv_arch, y_hist, server_model, progress)
                    if lam > 0.0:
                        scorer = FusedPotentialRanker(
                            local_ranker=full_ranker,
                            server_packet=server_model,
                            lam=lam,
                            loc_mu=float(np.mean(loc_arch)),
                            loc_sd=loc_sd,
                            srv_mu=float(np.mean(srv_arch)),
                            srv_sd=srv_sd,
                        )
            seeds = self._gated_server_seeds(server_model, lam, progress, x_hist, y_hist)
            u_next = cso_scalar_search(
                scorer,
                self.dim,
                pop_size=self.cso_pop_size,
                iters=self.cso_iters,
                phi=self.cso_phi,
                seeds=seeds,
                rng=self._rng,
            )
        else:
            chosen_arm = "knn"
            u_next, lam = self._propose_knn(x_hist, y_hist, server_model, progress)

        # Step 5: Evaluate proposed solution and update Thompson-sampling posteriors
        x_eval = self._from_unit(u_next).reshape(1, -1)
        y_new = float(np.asarray(self.problem.evaluate(x_eval)).ravel()[0])
        improved = y_new < prev_best
        self._ts_alpha[chosen_arm] = 0.98 * self._ts_alpha[chosen_arm] + (1.0 if improved else 0.0)
        self._ts_beta[chosen_arm] = 0.98 * self._ts_beta[chosen_arm] + (0.0 if improved else 1.0)
        self.lambda_history.append(lam)
        self.arm_history.append(chosen_arm)
        self.record_round_info("arm", chosen_arm)
        self.record_round_info("lambda", f"{lam:.3f}")
