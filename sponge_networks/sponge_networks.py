from typing import Literal

from networkx import Graph
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


def _grid_2d_assign_positions(G: nx.Graph, n_cols: int, n_rows: int) -> nx.Graph:
    G = G.copy()
    for node in G.nodes:
        x, y = node
        G.nodes[node]["pos"] = (x, y)
    return G


def grid_with_positions(n_cols: int, n_rows: int, grid_type: GridType) -> nx.DiGraph:
    match grid_type:
        case "grid_2d":
            grid_no_positions = nx.grid_2d_graph(
                n_cols + 1, n_rows + 1, create_using=nx.Graph
            )
            grid_undirected = _grid_2d_assign_positions(
                grid_no_positions, n_cols, n_rows
            )
        case "hexagonal":
            grid_undirected = nx.hexagonal_lattice_graph(
                n_rows, n_cols, create_using=nx.Graph
            )

        case "triangular":
            grid_undirected = nx.triangular_lattice_graph(
                n_rows, n_cols, create_using=nx.Graph
            )

        case _:
            raise ValueError(f'unknown grid type: "{grid_type}"')
    grid_directed = cast(nx.DiGraph, grid_undirected.to_directed())
    return grid_directed


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
