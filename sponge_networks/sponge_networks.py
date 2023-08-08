from typing import Literal, final
from abc import ABC, abstractmethod
import sponge_networks

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


SpongeNode: TypeAlias = tuple[int, int]
SpongeSinkNode: TypeAlias = tuple[int, Literal[-1]]


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


class AbstractSpongeNetworkProvider(ABC):
    @property
    @abstractmethod
    def grid(self) -> nx.DiGraph:
        ...

    @staticmethod
    @abstractmethod
    def _generate_sinks(grid: nx.DiGraph) -> nx.DiGraph:
        """
        ## Warning
        all sink nodes should be of type `SpongeSinkNode`
        """
        ...


class AbstractSpongeNetwork(ABC):
    def __init__(self, provider: AbstractSpongeNetworkProvider) -> None:
        grid = cast(nx.DiGraph, provider.grid.copy())
        grid = provider._generate_sinks(grid)

        self.resource_network = ResourceNetworkGreedy(grid)
        # self.grid = provider.grid

    @abstractmethod
    def run_sponge_simulation(
        self, initial_state: dict[SpongeNode, float] | list[float], n_iters: int = 30
    ) -> StateArray[SpongeNode]:
        """
        ## Warning
        `initial_state` applies only to the upper nodes. In case of list,
        the order of nodes shold be from left to right
        """
        ...

    def plot_simulation(self, sim: StateArray[SpongeNode], scale: float = 1.0) -> None:
        plot_simulation(self.resource_network, sim, scale)


@final
class SpongeNetwork2dProvider(AbstractSpongeNetworkProvider):
    def __init__(self, n_cols: int, n_rows: int) -> None:
        self.n_cols = n_cols
        self.n_rows = n_rows
        self._grid = grid_with_positions(n_cols, n_rows, grid_type="grid_2d")

    @property
    def grid(self) -> nx.DiGraph:
        return self._grid

    # @staticmethod
    # def _generate_sinks(grid: nx.DiGraph) -> nx.DiGraph:

    # return
