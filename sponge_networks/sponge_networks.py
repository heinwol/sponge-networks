import warnings
from abc import ABC, abstractmethod
from typing import Literal, final
import ipywidgets

import networkx as nx

from .resource_networks import *
from .utils.utils import *


class ResourceNetworkGreedy(ResourceNetwork):
    def __init__(self, G: nx.DiGraph | None = None):
        self.adjacency_matrix_without_loops: NDArrayT[AnyFloat]
        self.stochastic_matrix_without_loops: NDArrayT[AnyFloat]
        super().__init__(G)

    @override
    def _recalculate_matrices(self) -> None:
        super()._recalculate_matrices()
        adj_diag: NDArrayT[AnyFloat] = np.diag(self.adjacency_matrix).reshape((-1, 1))
        self.adjacency_matrix_without_loops = self.adjacency_matrix - np.diagflat(
            adj_diag
        )
        M_sum = self.adjacency_matrix_without_loops.sum(axis=1).reshape(  # type: ignore
            (-1, 1)
        )
        M_sum = np.where(np.isclose(M_sum, 0), np.inf, M_sum)
        self.stochastic_matrix_without_loops = (
            self.adjacency_matrix_without_loops / M_sum
        )

    @override
    def flow(self, q: NDArrayT[AnyFloat]) -> NDArrayT[AnyFloat]:
        q = np.asarray(q)
        q = q.reshape((-1, 1))
        adj_diag: NDArrayT[AnyFloat] = np.diag(self.adjacency_matrix).reshape((-1, 1))
        q_contained: NDArrayT[AnyFloat] = np.minimum(q, adj_diag)
        q_rest: NDArrayT[AnyFloat] = q - q_contained  # type: ignore
        return np.minimum(  # type: ignore
            q_rest * self.stochastic_matrix_without_loops, self.adjacency_matrix
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
    generate_sinks: bool


LayoutT = TypeVar("LayoutT", bound=SpongeNetworkLayout)
SpongeNetworkT = TypeVar("SpongeNetworkT", bound="SpongeNetwork")


@dataclass
class AbstractSpongeNetworkBuilder(ABC, Generic[LayoutT]):
    n_cols: int
    n_rows: int
    layout: LayoutT
    visual_sink_edge_length: float = 1
    # grid: nx.DiGraph = field(init=False)

    @abstractmethod
    def generate_initial_grid(self) -> nx.DiGraph: ...

    @abstractmethod
    def upper_nodes(self) -> list[SpongeNode]: ...

    @abstractmethod
    def bottom_nodes(self) -> list[SpongeNode]:
        """
        ## Warning
        these nodes do not include sink nodes
        """
        ...

    @abstractmethod
    def generate_weights_from_layout(self, grid: nx.DiGraph) -> nx.DiGraph: ...

    def generate_sinks(self, grid: nx.DiGraph) -> nx.DiGraph:
        grid = cast(nx.DiGraph, grid.copy())

        def gen_pos(node: SpongeNode) -> tuple[float, float]:
            node_d = grid.nodes[node]
            if "pos" in node_d:
                x, y = node_d["pos"]
            else:
                x, y = node
            return (x, y - self.visual_sink_edge_length)

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

    def final_grid_hook(self, grid: nx.DiGraph) -> nx.DiGraph:
        return grid

    def build(self, cls: type[SpongeNetworkT]) -> SpongeNetworkT:
        grid = self.generate_initial_grid()
        grid = self.generate_weights_from_layout(grid)
        grid = self.generate_loops(grid)
        if self.layout.generate_sinks:
            grid = self.generate_sinks(grid)
        grid = self.final_grid_hook(grid)

        resource_network = ResourceNetworkGreedy(grid)

        upper_nodes = self.upper_nodes()
        bottom_nodes = self.bottom_nodes()
        return cls(
            resource_network=resource_network,
            upper_nodes=upper_nodes,
            bottom_nodes=bottom_nodes,
        )


class SpongeNetwork:
    def __init__(
        self,
        resource_network: ResourceNetworkGreedy,
        upper_nodes: list[SpongeNode],
        bottom_nodes: Optional[list[SpongeNode]],
    ) -> None:
        """
        ## Warning!
        this constructor should **not** be used as is, unless you really know what you're doing
        """
        self.resource_network = resource_network
        self.upper_nodes = upper_nodes
        self.bottom_nodes = bottom_nodes

    @classmethod
    def build(cls, builder: AbstractSpongeNetworkBuilder) -> Self:
        return builder.build(cls)

    def initial_state_processor(
        self,
        initial_state_short: list[float] | dict[SpongeNode, float],
    ) -> dict[SpongeNode, float]:
        if isinstance(initial_state_short, dict):
            return initial_state_short
        elif len(self.upper_nodes) == len(initial_state_short):
            return dict(zip(self.upper_nodes, initial_state_short))
        else:
            raise ValueError(
                f"length of upper nodes ({len(self.upper_nodes)}) and initial state ({len(initial_state_short)}) should be equal"
            )

    def altered(
        self,
        callback: Callable[[nx.DiGraph], Optional[nx.DiGraph]],
        new_upper_nodes: Optional[list[SpongeNode]] = None,
    ) -> Self:
        G = self.resource_network.G
        res = callback(G)
        if res:
            G = res
        if not all("pos" in G.nodes[node] for node in G.nodes):
            warnings.warn(
                'Not all resulting nodes have "pos" attribute, network will be drawn incorrectly',
                RuntimeWarning,
            )
        if isinstance(new_upper_nodes, Iterable):
            upper_nodes = list(new_upper_nodes)
        else:
            upper_nodes = [node for node in self.upper_nodes if node in G.nodes]
        bottom_nodes = (
            [node for node in self.bottom_nodes if node in G.nodes]
            if self.bottom_nodes
            else None
        )
        rn = type(self.resource_network)(G)
        return type(self)(
            resource_network=rn, upper_nodes=upper_nodes, bottom_nodes=bottom_nodes
        )

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

    def plot_simulation(
        self,
        sim: StateArray[SpongeNode],
        prop_setter: Optional[Callable[[nx.DiGraph], None]] = None,
        scale: float = 1.0,
    ) -> ipywidgets.interactive:
        return self.resource_network.plot_simulation(
            sim, prop_setter=prop_setter, scale=scale
        )


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
    generate_sinks: bool


def build_sponge_network(
    grid_type: GridType,
    n_cols: int,
    n_rows: int,
    layout: _LayoutDict,
    visual_sink_edge_length: Optional[float] = None,
) -> SpongeNetwork:
    visual_sink_edge_length_dict = (
        {}
        if visual_sink_edge_length is None
        else {"visual_sink_edge_length": visual_sink_edge_length}
    )
    match grid_type:
        case "grid_2d":
            return SpongeNetwork.build(
                SpongeNetwork2dBuilder(
                    n_cols=n_cols,
                    n_rows=n_rows,
                    layout=Layout2d(**layout),
                    **visual_sink_edge_length_dict,
                ),
            )
        case "hexagonal":
            return SpongeNetwork.build(
                SpongeNetworkHexagonalBuilder(
                    n_cols=n_cols,
                    n_rows=n_rows,
                    layout=LayoutHexagonal(**layout),
                    **visual_sink_edge_length_dict,
                ),
            )
        case "triangular":
            return SpongeNetwork.build(
                SpongeNetworkTriangularBuilder(
                    n_cols=n_cols,
                    n_rows=n_rows,
                    layout=LayoutTriangular(**layout),
                    **visual_sink_edge_length_dict,
                ),
            )
        case _:
            raise ValueError(f'unknown grid type: "{grid_type}"')
