from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt

from IPython.lib import pretty
from ipywidgets import widgets

from scipy.sparse.linalg import eigs as sparce_eigs

from sponge_networks.display import (
    SimulationWithChangingWidthDrawable,
    display_svgs_interactively,
    JustDrawable,
)

from .utils.utils import *

# from scipy.sparse import sparray


class ResourceNetwork(Generic[Node]):
    def __init__(self, G: Optional[nx.DiGraph] = None):
        if G is None:
            G = nx.DiGraph()
        self._G = copy_graph(G)
        dp: tuple[dict[Node, int], list[Node]] = ResourceNetwork._descriptor_pair(
            self._G.nodes
        )
        self.node_descriptor, self.idx_descriptor = dp
        for u, v, d in self._G.edges(data=True):
            if "weight" not in d:
                d["weight"] = np.random.randint(1, 10)
        self.stochastic_matrix: NDArrayT[AnyFloat]
        self.adjacency_matrix: NDArrayT[AnyFloat]
        self._recalculate_matrices()

    @staticmethod
    def _descriptor_pair(
        nodes: nx.reportviews.NodeView,
    ) -> tuple[dict[Node, int], list[Node]]:
        """
        ## Warning:
        the first returned value is node descriptor,
        the second is index descriptor
        """
        node_descriptor: dict[Node, int] = {node: i for i, node in enumerate(nodes)}
        idx_descriptor: list[Node] = [None] * len(nodes)  # type: ignore
        for node, i in node_descriptor.items():
            idx_descriptor[i] = node
        return (node_descriptor, idx_descriptor)

    def _recalculate_matrices(self) -> None:
        M: NDArrayT[AnyFloat] = nx.adjacency_matrix(self._G).toarray()
        self.adjacency_matrix = M
        M_sum = M.sum(axis=1).reshape((-1, 1))
        M_sum: Any = np.where(np.isclose(M_sum, 0), np.inf, M_sum)
        self.stochastic_matrix = M / M_sum
        for i in range(len(M)):
            if M_sum[i] == np.inf:
                self.stochastic_matrix[i, i] = 1

    def _repr_pretty_(self, p: pretty.RepresentationPrinter, cycle: bool) -> None:
        ctor = pretty.CallExpression.factory(type(self).__name__)
        p.pretty(
            ctor(
                adjacency_matrix=self.adjacency_matrix,
                idx_descriptor=self.idx_descriptor,
            )
        )

    def __repr__(self) -> str:
        return pretty.pretty(self)

    @property
    def G(self) -> nx.DiGraph:
        """
        ## Warning:
        a copy of the underlying graph is returned,
        all operations on the result will have no effect on
        the ResourceNetwork instance
        """
        return copy_graph(self._G)

    def r_in(self) -> NDArrayT[AnyFloat]:
        return self.adjacency_matrix.sum(axis=0)

    def r_out(self) -> NDArrayT[AnyFloat]:
        return self.adjacency_matrix.sum(axis=1)

    def one_limit_state(self) -> NDArrayT[AnyFloat]:
        if not nx.is_aperiodic(self._G):
            raise ValueError(
                "Graph must be aperiodic for calculation of one limit state"
            )
        n = len(self.adjacency_matrix)

        eigval, eigvect = cast(
            tuple[NDArrayT[AnyFloat], NDArrayT[AnyFloat]],
            sparce_eigs(self.stochastic_matrix.T, k=1, sigma=1.1),
        )

        if not np.allclose(eigval, 1, atol=1e-7):
            raise RuntimeError(f"bad calculation of eigenvalue, it is {eigval}, not 1")
        if eigvect.shape == (n, 1):
            eigvect = np.real(eigvect.reshape(-1))
        else:
            raise RuntimeError(f"strange eigenvectors: {eigvect}")
        return eigvect / eigvect.sum()

    def T(self) -> float:
        q1_star = self.one_limit_state()
        r_out = self.r_out()
        T_i = r_out / q1_star
        min_ = T_i.min()
        return min_

    def state_to_normal_form(
        self, q: dict[Node, float] | list[float]
    ) -> dict[Node, float]:
        if isinstance(q, dict):
            for node in q.keys():
                if node not in self._G.nodes:
                    raise ValueError(f"unknown node: '{node}'")
            return {node: (q[node] if node in q else 0) for node in self._G.nodes}
        elif len(q) == len(self):
            return {node: x for node, x in zip(self.node_descriptor.keys(), q)}
        else:
            raise ValueError(
                f"expected list of proper length ({len(self)}), while got {q}"
            )

    def flow(self, q: NDArrayT[AnyFloat]) -> NDArrayT[AnyFloat]:
        q = np.asarray(q)
        q = q.reshape((-1, 1))
        return np.minimum(q * self.stochastic_matrix, self.adjacency_matrix)

    def S(
        self, q: npt.ArrayLike, flow: Optional[NDArrayT[AnyFloat]] = None
    ) -> NDArrayT[AnyFloat]:
        q = np.asarray(q)
        flow = self.flow(q) if flow is None else flow
        return q + flow.sum(axis=0) - flow.sum(axis=1)

    def __len__(self):
        return len(self.adjacency_matrix)

    def altered(self, callback: Callable[[nx.DiGraph], Optional[nx.DiGraph]]) -> Self:
        """
        This function provides the only interface to modify the underlying graph.
        If you want to get a copy of the graph, refer to the
        `ResourceNetwork.G` property

        Function `f` should either modify the graph in place and return None
        or return `networkx.DiGraph` which is supposed to be changed
        """

        G = self.G
        result = callback(G)
        if result is None:
            return type(self)(G)
        elif not isinstance(result, nx.DiGraph):
            raise ValueError(
                f"""the input function {callback} was supposed to return networkx.DiGraph,
                however it returned type '{type(result)}'
                with value '{result}'"""
            )
        else:
            return type(self)(result)

    def run_simulation(
        self, initial_state: dict[Node, float] | list[float], n_iters: int = 30
    ) -> StateArray[Node]:
        if not isinstance(initial_state, dict) and len(initial_state) != len(
            self._G.nodes
        ):
            raise ValueError(
                "Incorrect initial states: expected states for "
                + str(self._G.nodes)
                + ", while got:"
                + str(initial_state)
            )
        state_dict = self.state_to_normal_form(initial_state)
        n = len(state_dict)
        state_arr = np.zeros((n_iters, n))
        flow_arr = np.zeros((n_iters, n, n))

        for j in range(n):
            state_arr[0, j] = state_dict[self.idx_descriptor[j]]
        for i in range(1, n_iters):
            flow_arr[i] = self.flow(state_arr[i - 1])
            state_arr[i] = self.S(state_arr[i - 1], flow=flow_arr[i])
        total_output_res: NDArrayT[AnyFloat] = np.array(
            [
                sum(map(lambda v: self._G[u][v]["weight"], self._G[u]))
                for u in self.idx_descriptor
            ]
        )
        return StateArray[Node](
            self.node_descriptor,
            self.idx_descriptor,
            state_arr,
            flow_arr,
            total_output_res,
        )

    def plot_with_states(
        self,
        states: StateArray[Node],
        prop_setter: Optional[Callable[[nx.DiGraph], None]] = None,
        scale: Optional[float] = None,
        max_node_width: Optional[float] = None,
    ) -> list[SVG]:
        return SimulationWithChangingWidthDrawable.new(
            self._G, sim=states, scale=scale, max_node_width=max_node_width
        ).plot(prop_setter=prop_setter)

    def plot(
        self,
        scale: float = 1.7,
        prop_setter: Optional[Callable[[nx.DiGraph], None]] = None,
    ) -> SVG:
        return JustDrawable.new(self._G).plot(scale, prop_setter)

    def plot_simulation(
        self,
        simulation: StateArray[Node],
        prop_setter: Optional[Callable[[nx.DiGraph], None]] = None,
        scale: Optional[float] = None,
    ) -> widgets.interactive:
        pl = self.plot_with_states(
            simulation,
            prop_setter=prop_setter,
            scale=scale,
        )
        return display_svgs_interactively(pl)


class ResourceNetworkWithIncome(ResourceNetwork):
    @override
    def run_simulation(
        self,
        initial_state: dict[Node, float] | list[float],
        n_iters: int = 30,
        income_seq_func: Optional[Callable[[int], list[float]]] = None,
    ) -> StateArray[Node]:
        if len(initial_state) != len(self._G.nodes):
            raise ValueError(
                "Incorrect initial states: expected states for "
                + str(self._G.nodes)
                + ", while got:"
                + str(initial_state)
            )
        n = len(initial_state)
        state_arr = np.zeros((n_iters, n))
        flow_arr = np.zeros((n_iters, n, n))

        if income_seq_func is None:
            income_seq_func = lambda t: [0] * n

        if isinstance(initial_state, dict):
            state_dict = initial_state
        else:
            state_dict = {
                node: x for node, x in zip(self.node_descriptor.keys(), initial_state)
            }
        for j in range(n):
            state_arr[0, j] = state_dict[self.idx_descriptor[j]]

        total_output_res = cast(
            NDArrayT[float],  # type: ignore
            np.array(
                [
                    sum(map(lambda v: self._G[u][v]["weight"], self._G[u]))
                    for u in self.idx_descriptor
                ]
            ),
        )

        for i in range(1, n_iters):
            income_seq = income_seq_func(i - 1)
            for u in self._G.nodes:
                u_i = self.node_descriptor[u]
                for v in self._G[u]:
                    v_i = self.node_descriptor[v]
                    transferred_res = min(
                        self._G[u][v]["weight"]
                        / total_output_res[u_i]
                        * state_arr[i - 1, u_i],
                        self._G[u][v]["weight"],
                    )
                    flow_arr[i, u_i, v_i] = transferred_res
                    state_arr[i, v_i] += transferred_res
                state_arr[i, u_i] += (
                    max(state_arr[i - 1, u_i] - total_output_res[u_i], 0)
                    + income_seq[u_i]
                )

        return StateArray[Node](
            self.node_descriptor,
            self.idx_descriptor,
            state_arr,
            flow_arr,
            np.asarray(total_output_res),
        )
