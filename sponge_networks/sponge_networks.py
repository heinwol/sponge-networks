from dataclasses import field
from typing import Literal, final
from abc import ABC, abstractmethod
import sponge_networks
from sponge_networks.resource_networks import nx
from sponge_networks.utils.utils import nx

from .utils.utils import *
from .resource_networks import *
import abc


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


# grid layouts are as follows:
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
# SpongeSinkNode: TypeAlias = tuple[int, Literal[-1]]


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


@dataclass
class SpongeNetworkLayout:
    weights_sink_edge: float
    weights_loop: float


LayoutT = TypeVar("LayoutT", bound=SpongeNetworkLayout)


@dataclass
class AbstractSpongeNetworkBuilder(ABC, Generic[LayoutT]):
    n_cols: int
    n_rows: int
    layout: LayoutT
    # grid: nx.DiGraph = field(init=False)

    @abstractmethod
    def generate_initial_grid(self) -> nx.DiGraph:
        ...

    @abstractmethod
    def upper_nodes(self) -> list[SpongeNode]:
        ...

    @abstractmethod
    def bottom_nodes(self) -> list[SpongeNode]:
        """
        ## Warning
        these nodes does not include sink nodes
        """
        ...

    @abstractmethod
    def generate_weights_from_layout(self, grid: nx.DiGraph) -> nx.DiGraph:
        ...

    def generate_sinks(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())
        grid.add_edges_from(
            (bottom_node, (i, -1), {"weight": self.layout.weights_sink_edge})
            for i, bottom_node in enumerate(self.bottom_nodes())
        )
        return grid

    def generate_loops(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())
        grid.add_edges_from(
            (node, node, {"weight": self.layout.weights_loop}) for node in grid.nodes
        )
        return grid

    def final_grid_hook(self, grid: nx.DiGraph) -> nx.DiGraph:
        return grid


class AbstractSpongeNetwork(ABC):
    def __init__(self, builder: AbstractSpongeNetworkBuilder) -> None:
        grid = builder.generate_initial_grid()
        grid = builder.generate_loops(grid)
        grid = builder.generate_weights_from_layout(grid)
        grid = builder.generate_sinks(grid)
        grid = builder.final_grid_hook(grid)

        self.resource_network = ResourceNetworkGreedy(grid)

    @abstractmethod
    def run_sponge_simulation(
        self, initial_state: dict[SpongeNode, float] | list[float], n_iters: int = 30
    ) -> StateArray[SpongeNode]:
        """
        ## Warning
        `initial_state` applies only to the upper nodes. In case of list,
        the order of nodes shold go from left to right
        """
        ...

    def plot_simulation(self, sim: StateArray[SpongeNode], scale: float = 1.0) -> None:
        plot_simulation(self.resource_network, sim, scale)


@dataclass
class Layout2d(SpongeNetworkLayout):
    horizontal_weight: float


@final
@dataclass
class SpongeNetwork2dBuilder(AbstractSpongeNetworkBuilder[Layout2d]):
    @override
    def generate_initial_grid(self) -> nx.DiGraph:
        return grid_with_positions(self.n_cols, self.n_rows, "grid_2d")

    @override
    def upper_nodes(self) -> list[SpongeNode]:
        return [(i, 0) for i in range(self.n_cols + 1)]

    @override
    def bottom_nodes(self) -> list[SpongeNode]:
        return [(i, self.n_rows) for i in range(self.n_cols + 1)]

    @override
    def generate_weights_from_layout(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())

        # grid.add_edges_from(
        #     (bottom_node, (i, -1), {"weight": self.layout.sink_edge_weights})
        #     for i, bottom_node in enumerate(self.bottom_nodes())
        # )

        return grid
