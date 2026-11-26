"""Minimal in-process client-server interfaces for standalone Fed-CPPF execution.

Each client runs in its own thread against a shared in-process server (see
``examples/minimal_demo.py``). Every upload is stored by round version under a
lock; once all registered clients upload version ``r``, the server triggers
Leave-Target-Out personalized aggregation for every client and serves the
resulting packet ``M_i``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class ClientPackage:
    cid: int
    action: str
    version: int | None = None
    data: Any = None


class Client:
    """Minimal client base compatible with in-process multi-threaded execution."""

    def __init__(self, problem: Any):
        self.problem = problem
        self._server: Server | None = None
        self.rounds_info: dict[str, list[str]] = defaultdict(list)

    def bind_server(self, server: "Server") -> None:
        self._server = server
        server.register_client(self.id)

    def request_server(
        self,
        package: ClientPackage,
        repeat: int = 1500,
        interval: float = 0.05,
    ) -> Any:
        if self._server is None:
            raise RuntimeError("Call bind_server(server) before optimizing.")
        for _ in range(max(1, int(repeat))):
            response = self._server.serve(package)
            if response is not None:
                return response
            time.sleep(interval)
        return None

    def record_round_info(self, name: str, value: Any) -> None:
        self.rounds_info[name].append(str(value))

    id = property(lambda self: self.problem.id)
    dim = property(lambda self: self.problem.dim)
    obj = property(lambda self: self.problem.obj)
    fe_init = property(lambda self: self.problem.fe_init)
    fe_max = property(lambda self: self.problem.fe_max)
    lb = property(lambda self: self.problem.lb)
    ub = property(lambda self: self.problem.ub)
    solutions = property(lambda self: self.problem.solutions)


class Server:
    """Thread-safe synchronous in-process server with per-round aggregation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client_ids: set[int] = set()
        self._uploads: dict[int, dict[int, Any]] = defaultdict(dict)
        self._responses: dict[int, dict[int, Any]] = defaultdict(dict)

    def register_client(self, cid: int) -> None:
        with self._lock:
            self._client_ids.add(int(cid))

    @property
    def sorted_ids(self) -> list[int]:
        return sorted(self._client_ids)

    def serve(self, package: ClientPackage) -> Any:
        with self._lock:
            cid = int(package.cid)
            ver = int(package.version or 1)
            if package.action == "upload":
                self._uploads[ver][cid] = package.data
                if len(self._uploads[ver]) >= len(self._client_ids):
                    self.aggregate_version(ver, self._uploads[ver])
                return {"status": "ok", "version": ver}
            if package.action == "download":
                if ver in self._responses and cid in self._responses[ver]:
                    return self._responses[ver][cid]
                return None
            raise ValueError(f"Unknown action: {package.action}")

    def aggregate_version(self, version: int, uploads_by_cid: dict[int, Any]) -> None:
        raise NotImplementedError
