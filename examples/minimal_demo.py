"""Standalone in-process Fed-CPPF example with four heterogeneous tasks.

Two clients solve similar shifted-and-scaled Sphere tasks (different objective
scales of 1x and 1000x to verify positive-affine invariance), while two other
clients solve shifted Rosenbrock tasks in a distant basin. Each client runs in
its own thread against an in-process FedCPPFServer.
"""

from pathlib import Path
import sys
import threading
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fed_cppf import FedCPPFClient, FedCPPFServer


class SolutionArchive:
    def __init__(self, dim: int):
        self.x = np.empty((0, dim), dtype=float)
        self.y = np.empty((0, 1), dtype=float)

    def append(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.vstack([self.x, np.asarray(x, dtype=float).reshape(-1, self.x.shape[1])])
        self.y = np.vstack([self.y, np.asarray(y, dtype=float).reshape(-1, 1)])

    size = property(lambda self: len(self.y))
    y_min = property(lambda self: float(np.min(self.y)))


class SyntheticTask:
    """Simple bounded black-box task matching the client archive interface."""

    def __init__(
        self,
        cid: int,
        kind: str,
        shift: np.ndarray,
        scale: float = 1.0,
        offset: float = 0.0,
        dim: int = 5,
        fe_init: int = 20,
        fe_max: int = 35,
        seed: int = 0,
    ):
        self.id = cid
        self.name = f"{kind}_{cid}"
        self.kind = kind
        self.dim = dim
        self.obj = 1
        self.fe_init = fe_init
        self.fe_max = fe_max
        self.lb = -5.0 * np.ones(dim, dtype=float)
        self.ub = 5.0 * np.ones(dim, dtype=float)
        self.shift = np.asarray(shift, dtype=float).reshape(1, dim)
        self.scale = float(scale)
        self.offset = float(offset)
        self.rng = np.random.default_rng(seed)
        self.solutions = SolutionArchive(dim)
        init_x = self.rng.uniform(self.lb, self.ub, size=(fe_init, dim))
        self.evaluate(init_x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float).reshape(-1, self.dim), self.lb, self.ub)
        if self.solutions.size + len(x) > self.fe_max:
            raise ValueError("Evaluation budget exceeded.")
        z = x - self.shift
        if self.kind == "sphere":
            raw = np.sum(z**2, axis=1, keepdims=True)
        else:
            raw = np.sum(100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1.0 - z[:, :-1]) ** 2, axis=1, keepdims=True)
        y = self.scale * raw + self.offset
        self.solutions.append(x, y)
        return y


def run_client(client: FedCPPFClient) -> None:
    while client.solutions.size < client.fe_max:
        client.optimize()


def main() -> None:
    dim = 5
    tasks = [
        SyntheticTask(1, "sphere", shift=np.array([-1.5, -1.5, -1.5, -1.5, -1.5]), scale=1.0, offset=0.0, dim=dim, seed=10),
        SyntheticTask(2, "sphere", shift=np.array([-1.4, -1.5, -1.6, -1.5, -1.4]), scale=1000.0, offset=50.0, dim=dim, seed=20),
        SyntheticTask(3, "rosen", shift=np.array([2.0, 2.0, 2.0, 2.0, 2.0]), scale=1.0, offset=0.0, dim=dim, seed=30),
        SyntheticTask(4, "rosen", shift=np.array([2.1, 1.9, 2.0, 2.1, 1.9]), scale=10.0, offset=-20.0, dim=dim, seed=40),
    ]
    server = FedCPPFServer(topk=5, min_similarity=0.40, ridge=0.25)
    clients = [
        FedCPPFClient(t, ordinal_min_points=24, n_probes=32, cso_pop_size=40, cso_iters=40)
        for t in tasks
    ]
    for c in clients:
        c.bind_server(server)

    threads = [threading.Thread(target=run_client, args=(c,)) for c in clients]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    print("Fed-CPPF Minimal Demo Results:")
    for c in clients:
        active = sum(1 for lam in c.lambda_history if lam > 0.0)
        mean_lam = float(np.mean(c.lambda_history)) if c.lambda_history else 0.0
        print(
            f"  Client {c.id} ({c.problem.name:9s}): FE={c.solutions.size:2d}, "
            f"best_y={c.solutions.y_min:10.4f}, active_transfer={active:2d}/{len(c.lambda_history)}, "
            f"mean_lambda={mean_lam:.3f}"
        )


if __name__ == "__main__":
    main()
