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

        def gen_pos(node: SpongeNode) -> tuple[float, float]:
            node_d = grid.nodes[node]
            if "pos" in node_d:
                x, y = node_d["pos"]
            else:
                x, y = node
            return (x, y - 1)

        bottom_nodes = self.bottom_nodes()
        grid.add_nodes_from(
            ((i, -1), {"pos": gen_pos(bottom_node)})
            for i, bottom_node in enumerate(bottom_nodes)
        )
        grid.add_edges_from(
            (bottom_node, (i, -1), {"weight": self.layout.weights_sink_edge})
            for i, bottom_node in enumerate(bottom_nodes)
        )
        return grid

    def generate_loops(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())
        grid.add_edges_from(
            (node, node, {"weight": self.layout.weights_loop}) for node in grid.nodes
        )
        return grid

    def initial_state_processor_provider(
        self,
    ) -> Callable[[list[float] | dict[SpongeNode, float]], dict[SpongeNode, float]]:
        upper_nodes = self.upper_nodes()

        def initial_state_processor(
            initial_state_short: list[float] | dict[SpongeNode, float],
        ) -> dict[SpongeNode, float]:
            if isinstance(initial_state_short, dict):
                return initial_state_short
            elif len(upper_nodes) == len(initial_state_short):
                return dict(zip(upper_nodes, initial_state_short))
            else:
                raise ValueError(
                    f"length of upper nodes ({len(upper_nodes)}) and initial state ({len(initial_state_short)}) should be equal"
                )

        return initial_state_processor

    def final_grid_hook(self, grid: nx.DiGraph) -> nx.DiGraph:
        return grid


class SpongeNetwork:
    def __init__(
        self, builder: AbstractSpongeNetworkBuilder, generate_sinks: bool = True
    ) -> None:
        grid = builder.generate_initial_grid()
        grid = builder.generate_weights_from_layout(grid)
        grid = builder.generate_loops(grid)
        if generate_sinks:
            grid = builder.generate_sinks(grid)
        grid = builder.final_grid_hook(grid)

        self.resource_network = ResourceNetworkGreedy(grid)

        self.upper_nodes = builder.upper_nodes()
        self.bottom_nodes = builder.bottom_nodes()
        self.initial_state_processor = builder.initial_state_processor_provider()

    def altered(self, callback: Callable[[nx.DiGraph], Optional[nx.DiGraph]]) -> Self:
        ...

    def run_sponge_simulation(
        self, initial_state: dict[SpongeNode, float] | list[float], n_iters: int = 30
    ) -> StateArray[SpongeNode]:
        """
        ## Warning
        `initial_state` applies only to the upper nodes. In case of list,
        the order of nodes shold go from left to right
        """
        return self.resource_network.run_simulation(
            self.initial_state_processor(initial_state), n_iters=n_iters
        )

    def plot_simulation(self, sim: StateArray[SpongeNode], scale: float = 1.0) -> None:
        self.resource_network.plot_simulation(sim, scale)


@dataclass
class Layout2d(SpongeNetworkLayout):
    weights_horizontal: float
    weights_up_down: float
    weights_down_up: float


@final
@dataclass
class SpongeNetwork2dBuilder(AbstractSpongeNetworkBuilder[Layout2d]):
    @override
    def generate_initial_grid(self) -> nx.DiGraph:
        return grid_with_positions(self.n_cols, self.n_rows, "grid_2d")

    @override
    def upper_nodes(self) -> list[SpongeNode]:
        return [(i, self.n_rows) for i in range(self.n_cols + 1)]

    @override
    def bottom_nodes(self) -> list[SpongeNode]:
        return [(i, 0) for i in range(self.n_cols + 1)]

    @override
    def generate_weights_from_layout(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())
        u: tuple[int, int]
        v: tuple[int, int]
        d: dict
        for u, v, d in grid.edges(data=True):
            (i1, j1), (i2, j2) = (u, v)
            if j1 == j2:
                d["weight"] = self.layout.weights_horizontal
            elif i1 == i2 and j1 > j2:
                d["weight"] = self.layout.weights_up_down
            elif i1 == i2 and j1 < j2:
                d["weight"] = self.layout.weights_down_up
            else:
                raise ValueError(
                    f"some strange edge encountered while building sponge network: {u} -> {v}"
                )
        return grid


@dataclass
class LayoutTriangular(SpongeNetworkLayout):
    weights_horizontal: float
    weights_up_down: float
    weights_down_up: float


@final
@dataclass
class SpongeNetworkTriangularBuilder(AbstractSpongeNetworkBuilder[LayoutTriangular]):
    @override
    def generate_initial_grid(self) -> nx.DiGraph:
        return grid_with_positions(self.n_cols, self.n_rows, "triangular")

    @override
    def upper_nodes(self) -> list[SpongeNode]:
        upper_nodes_len = (self.n_cols + (self.n_rows + 1) % 2) // 2 + 1
        return [(i, self.n_rows) for i in range(upper_nodes_len)]

    @override
    def bottom_nodes(self) -> list[SpongeNode]:
        bottom_nodes_len = (self.n_cols + 1) // 2 + 1
        return [(i, 0) for i in range(bottom_nodes_len)]

    @override
    def generate_weights_from_layout(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())
        u: tuple[int, int]
        v: tuple[int, int]
        d: dict
        for u, v, d in grid.edges(data=True):
            (i1, j1), (i2, j2) = (u, v)
            if j1 == j2:
                d["weight"] = self.layout.weights_horizontal
            elif j1 > j2:
                d["weight"] = self.layout.weights_up_down
            elif j1 < j2:
                d["weight"] = self.layout.weights_down_up
            else:
                raise ValueError(
                    f"some strange edge encountered while building sponge network: {u} -> {v}"
                )
        return grid


@dataclass
class LayoutHexagonal(SpongeNetworkLayout):
    weights_horizontal: float
    weights_up_down: float
    weights_down_up: float


@final
@dataclass
class SpongeNetworkHexagonalBuilder(AbstractSpongeNetworkBuilder[LayoutHexagonal]):
    @override
    def generate_initial_grid(self) -> nx.DiGraph:
        return grid_with_positions(self.n_cols, self.n_rows, "hexagonal")

    @override
    def upper_nodes(self) -> list[SpongeNode]:
        if self.n_cols == 1:
            return [(i, self.n_rows * 2) for i in range(2)]
        upper_nodes_len = (self.n_cols) // 2 * 2
        return [(i + 1, self.n_rows * 2 + 1) for i in range(upper_nodes_len)]

    @override
    def bottom_nodes(self) -> list[SpongeNode]:
        bottom_nodes_len = (self.n_cols + 1) // 2 * 2
        return [(i, 0) for i in range(bottom_nodes_len)]

    @override
    def generate_weights_from_layout(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())
        u: tuple[int, int]
        v: tuple[int, int]
        d: dict
        for u, v, d in grid.edges(data=True):
            (i1, j1), (i2, j2) = (u, v)
            if j1 == j2:
                d["weight"] = self.layout.weights_horizontal
            elif j1 > j2:
                d["weight"] = self.layout.weights_up_down
            elif j1 < j2:
                d["weight"] = self.layout.weights_down_up
            else:
                raise ValueError(
                    f"some strange edge encountered while building sponge network: {u} -> {v}"
                )
        return grid


class _LayoutDict(TypedDict):
    weights_sink_edge: float
    weights_loop: float
    weights_horizontal: float
    weights_up_down: float
    weights_down_up: float


def build_sponge_network(
    grid_type: GridType,
    n_cols: int,
    n_rows: int,
    layout: _LayoutDict,
    generate_sinks: bool = True,
) -> SpongeNetwork:
    match grid_type:
        case "grid_2d":
            return SpongeNetwork(
                SpongeNetwork2dBuilder(
                    n_cols=n_cols,
                    n_rows=n_rows,
                    layout=Layout2d(**layout),
                ),
                generate_sinks=generate_sinks,
            )
        case "hexagonal":
            return SpongeNetwork(
                SpongeNetworkHexagonalBuilder(
                    n_cols=n_cols,
                    n_rows=n_rows,
                    layout=LayoutHexagonal(**layout),
                ),
                generate_sinks=generate_sinks,
            )
        case "triangular":
            return SpongeNetwork(
                SpongeNetworkTriangularBuilder(
                    n_cols=n_cols,
                    n_rows=n_rows,
                    layout=LayoutTriangular(**layout),
                ),
                generate_sinks=generate_sinks,
            )
        case _:
            raise ValueError(f'unknown grid type: "{grid_type}"')
