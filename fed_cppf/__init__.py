"""Fed-CPPF: Privacy-Preserving Federated Many-Task Optimization via Calibrated Personalized Potential Fusion."""

from .client import FedCPPFClient
from .server import FedCPPFServer

__all__ = ["FedCPPFClient", "FedCPPFServer"]
