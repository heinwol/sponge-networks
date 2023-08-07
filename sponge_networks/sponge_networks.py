from typing import Literal
from .utils.utils import *
from .resource_networks import *


class ResourceNetworkGreedy(ResourceNetwork):
    @override
    def flow(self, q: NDarrayT[AnyFloat]) -> NDarrayT[AnyFloat]:
        q = np.asarray(q)
        q = q.reshape((-1, 1))
        adj_diag: NDarrayT[AnyFloat] = np.diag(self.adjacency_matrix).reshape((-1, 1))
        q_contained: NDarrayT[AnyFloat] = np.minimum(q, adj_diag)
        q_rest: NDarrayT[AnyFloat] = q - q_contained  # type: ignore
        return np.minimum(  # type: ignore
            q_rest * self.stochastic_matrix, self.adjacency_matrix
        ) + np.diag(q_contained)


GridType: TypeAlias = Literal["triangular", "hexagonal", "grid_2d"]

# grid types are as follows:
# =========================
# (visually inaccurate, but are isomorphic and orientation is preserved)
#
#    hexagonal:
# ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
# │ (0, 1) │ ─── │ (0, 2) │ ─── │ (1, 2) │ ─── │ (1, 1) │
# └────────┘     └────────┘     └────────┘     └────────┘
#   │                                            │
#   │                                            │
#   │                                            │
# ┌────────┐     ┌────────┐                      │
# │ (0, 0) │ ─── │ (1, 0) │──────────────────────┘
# └────────┘     └────────┘
#
#    triangular:
#           ┌────────┐
#   ┌───────│ (0, 1) │────┐
#   │       └────────┘    │
# ┌────────┐          ┌────────┐
# │ (0, 0) │ ──────── │ (1, 0) │
# └────────┘          └────────┘
#
#
#    grid_2d:
#
# ┌────────┐     ┌────────┐
# │ (0, 1) │ ─── │ (1, 1) │
# └────────┘     └────────┘
#     │              │
#     │              │
# ┌────────┐     ┌────────┐
# │ (0, 0) │ ─── │ (1, 0) │
# └────────┘     └────────┘


# def f(t: GridType) -> int:
#     match t:
#         case "grid_2d":
#             return 1
#         case "hexagonal":
#             return 2
#         case "triangular":
#             return 3
#         case _:
#             raise ValueError("wrong type of grid")
