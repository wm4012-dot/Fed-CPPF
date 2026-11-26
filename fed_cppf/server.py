"""Server-side Leave-Target-Out filtering and personalized potential synthesis."""

from __future__ import annotations

from typing import Any

from .framework import Server
from .model import server_fit_personalized_model


class FedCPPFServer(Server):
    """Federated server for Fed-CPPF.

    Parameters
    ----------
    topk : int, default=5
        Maximum number of top-ranked peers (``P``) retained after Leave-Target-Out
        Spearman filtering.
    min_similarity : float, default=0.40
        Minimum Spearman rank correlation ``rho_min`` required for peer inclusion.
    ridge : float, default=0.25
        Ridge regularization ``gamma`` for the public-probe residual RBF fit.
    similarity_weighting : bool, default=True
        Weight selected peers by squared Spearman correlation ``omega_{i,k} ~ rho_{i,k}^2``.
    residual_enabled : bool, default=True
        Include the non-separable public-probe RBF residual layer.
    proposal_enabled : bool, default=True
        Compute and attach analytic quadratic and L-BFGS-B server minima ``(x_quad, x_srv)``.
    shuffle_summary : bool, default=False
        Ablation flag: permute probe predictions before aggregation.
    """

    def __init__(
        self,
        *,
        topk: int = 5,
        min_similarity: float = 0.40,
        ridge: float = 0.25,
        similarity_weighting: bool = True,
        residual_enabled: bool = True,
        proposal_enabled: bool = True,
        shuffle_summary: bool = False,
    ):
        super().__init__()
        self.topk = max(1, int(topk))
        self.min_similarity = float(min_similarity)
        self.ridge = float(ridge)
        self.similarity_weighting = bool(similarity_weighting)
        self.residual_enabled = bool(residual_enabled)
        self.proposal_enabled = bool(proposal_enabled)
        self.shuffle_summary = bool(shuffle_summary)
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [-1, 1].")
        if self.ridge <= 0.0:
            raise ValueError("ridge must be positive.")

    def aggregate_version(self, version: int, uploads_by_cid: dict[int, Any]) -> None:
        summaries_by_cid = {
            int(cid): payload["summary"]
            for cid, payload in uploads_by_cid.items()
            if isinstance(payload, dict) and payload.get("summary") is not None
        }
        for cid in self.sorted_ids:
            if cid not in summaries_by_cid or len(summaries_by_cid) < 2:
                self._responses[version][cid] = {"model": None}
                continue
            model = server_fit_personalized_model(
                target_cid=cid,
                summaries_by_cid=summaries_by_cid,
                version=int(version),
                topk=self.topk,
                min_sim=self.min_similarity,
                srv_ridge=self.ridge,
                similarity_weighting=self.similarity_weighting,
                residual_enabled=self.residual_enabled,
                proposal_enabled=self.proposal_enabled,
                shuffle_summary=self.shuffle_summary,
            )
            self._responses[version][cid] = {"model": model}
