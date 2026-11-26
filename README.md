# Fed-CPPF

This repository contains the reference implementation of **Fed-CPPF** (*Privacy-Preserving Federated Many-Task Optimization via Calibrated Personalized Potential Fusion*).

Each client owns an expensive black-box optimization task and never transmits raw evaluated coordinates, raw objective values, or private kernel centers. Instead, clients exchange scale-invariant functional summaries on a shared deterministic public Sobol probe grid, and every client calibrates server fusion locally via out-of-fold (OOF) concordance before search.

## Algorithmic Core

- **Scale-Invariant Public-Probe Encoding (`fed_cppf/model.py`)**: Each client maps its private objective archive to a rank--log-IQR hybrid target (invariant to positive affine transformations when the stabilization floor is inactive) and uploads only an elite-weighted separable quadratic trend vector (`2D + 1` floats) plus local surrogate predictions evaluated on a public scrambled Sobol probe grid (`M = 128` floats).
- **Leave-Target-Out Personalized Potential Synthesis (`fed_cppf/server.py`)**: For each target client `i`, the server excludes client `i` from parameter aggregation, selects up to Top-`P` peers whose Spearman probe correlation meets `rho_min`, and builds a personalized quadratic + public-grid RBF potential model together with candidate minima `(x_quad, x_srv)`.
- **OOF-Calibrated Potential Fusion & Dual-Arm Search (`fed_cppf/client.py`, `fed_cppf/ranker.py`)**: Each client fits a pure-NumPy antisymmetric scalar-potential ranker (`D -> 32 -> 16 -> 1`, `tanh`) with 3-fold point-disjoint OOF calibration and dual-scale elite guardrails (G1)--(G3) to determine the fusion weight `lambda* in [0, lambda_max]`, falling back cleanly to `lambda* = 0` (pure local search) whenever server alignment fails. Competitive Swarm Optimization (CSO) requires one score evaluation per particle (`O(N_p)` per generation) and is paired with a Rank-KNN secondary arm via Thompson sampling.

**Data-Sharing Boundary**: Clients share only a public Sobol seed `s_pub` and scale-invariant summaries `(w_poly, f_pub)` on `X_pub`; raw coordinates, objective values, and private kernel centers remain strictly local.

## Package Layout

- `fed_cppf/model.py`: Scrambled Sobol public probe grid, rank--log-IQR encoding, Leave-Target-Out peer filtering, personalized server potential model, dual-scale OOF weight calibration, and `FusedPotentialRanker`.
- `fed_cppf/ranker.py`: Pure-NumPy antisymmetric scalar-potential MLP, GP-LCB probe labeling, point-disjoint 3-fold OOF validation, CSO scalar-potential search, and Rank-KNN acquisition.
- `fed_cppf/client.py`: `FedCPPFClient` implementing Algorithm 1 (upload, download, OOF calibration, Thompson-sampling dual-arm selection, gated early-stage server seed injection).
- `fed_cppf/server.py`: `FedCPPFServer` implementing Leave-Target-Out Spearman Top-`P` aggregation.
- `fed_cppf/framework.py`: Minimal in-process client/server synchronization layer (no external framework required to run the demo).
- `examples/minimal_demo.py`: Self-contained four-client multi-task example with mixed objective scales and distinct basins.

## Installation

Python 3.10 or newer is recommended:

```bash
pip install -r requirements.txt
```

No GPU or PyTorch installation is required.

## Minimal Example

Run the standalone demo from the repository root:

```bash
python examples/minimal_demo.py
```

The demo runs four clients in separate threads against an in-process `FedCPPFServer` (two similar shifted Sphere tasks scaled by `1x` and `1000x` plus two shifted Rosenbrock tasks) and prints each client's evaluation count, best objective value, active transfer rounds, and mean calibrated fusion weight `lambda*`.

## Default Hyperparameters

| Module | Parameters | Defaults |
| :--- | :--- | :--- |
| Public-Probe Encoding | `grid_size (M)`, `grid_seed`, `poly_ridge`, `rbf_ridge` | `128`, `2026`, `0.05`, `0.03` |
| Server LTO Model | `min_similarity (rho_min)`, `topk (P)`, `ridge (gamma)` | `0.40`, `5`, `0.25` |
| Local Ranker & Arms | MLP widths, `n_probes`, `min_bacc`, `knn_k`, `knn_kappa` | `32-16-1`, `100`, `0.60`, `7`, `0.35` |
| OOF Calibration | `top_frac`, `elite_k`, `elite_weight`, `max_weight`, `late_floor_ratio` | `0.25`, `12`, `0.55`, `0.50`, `0.15` |
| Guardrails (G1)--(G3) | `min_top_acc`, `elite_tol`, `min_gain` | `0.55`, `0.005`, `0.005` |

Ablation flags (`enabled=False`, `shuffle_summary=True`, `guide_ordinal=False`, `oof_calibration=False`, `residual_enabled=False`) are exposed directly on `FedCPPFClient` and `FedCPPFServer`.

## License

Released under the [MIT License](LICENSE).
