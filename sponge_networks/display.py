import multiprocessing
import os
from typing import Callable, Optional, Sequence, cast
import networkx as nx
import numpy as np
import numpy.typing as npt
from returns.curry import curry
from toolz import valmap
from IPython.core.display import SVG
from ipywidgets import interact, widgets

from sponge_networks.utils.utils import (
    AnyFloat,
    Node,
    StateArray,
    const,
    const_iter,
    flatten,
    linear_func_from_2_points,
    parallelize_range,
)


def preserve_pos_when_plotting(G: nx.DiGraph):
    if all(map(lambda node: "pos" in G.nodes[node], G.nodes)):
        for node in G.nodes:
            node_d: dict = G.nodes[node]
            if isinstance(node_d["pos"], Sequence):
                node_d["pos"] = f"{node_d['pos'][0]},{node_d['pos'][1]}!"
            elif not isinstance(node_d["pos"], str):
                raise ValueError(
                    f"""
                    Attribute "pos" of every node should be either Sequence or str,
                    however on node '{node}' attribute "pos" is of type '{type(node_d["pos"])}'
                    with value '{node_d["pos"]}'
                    """
                )
    else:
        layout = nx.nx_pydot.pydot_layout(G, prog="neato")
        layout_new = valmap(lambda x: (x[0] / 45, x[1] / 45), layout)
        for v in G.nodes:
            G.nodes[v]["pos"] = f"{layout_new[v][0]},{layout_new[v][1]}!"


def parallel_plot(
    G: nx.DiGraph, states: StateArray[Node], rng: Sequence[int]
) -> list[SVG]:
    def my_fmt(x: AnyFloat | int) -> str:
        if np.isclose(x, 0):
            return "0"
        if isinstance(x, int):
            return str(x)
        x_int, x_frac = int(x), x % 1
        if x_frac < 1e-3 or x >= 1e3:
            rem = ""
        else:
            len_int = len(str(x_int))
            rem = str(int(x_frac * 10 ** (4 - len_int)))
            rem = ("0" * (4 - len_int - len(rem)) + rem).rstrip("0")
        return str(x_int) + "." + rem

    total_sum = states.states_arr[-1].sum()
    calc_node_width = (
        linear_func_from_2_points((0, 0.35), (total_sum, 1.1))
        if total_sum > 0
        else const(0.35)
    )
    res: list[SVG] = [None] * len(rng)  # type: ignore
    for n_it, idx in enumerate(rng):
        state = states[idx]
        for v in G.nodes:
            v = cast(Node, v)
            if "color" not in G.nodes[v] or G.nodes[v]["color"] != "transparent":
                G.nodes[v]["label"] = my_fmt(cast(AnyFloat, state["states"][[v]]))
                G.nodes[v]["width"] = calc_node_width(cast(float, state["states"][[v]]))  # type: ignore

                G.nodes[v]["fillcolor"] = (
                    "#f0fff4"
                    if (
                        state["states"][[v]] < state["total_output_res"][[v]]
                        and not np.isclose(state["total_output_res"][[v]], 0)
                    )
                    else "#b48ead"
                )

        for u, v, d in G.edges(data=True):
            d["label"] = d["weight"]
        res[n_it] = SVG(nx.nx_pydot.to_pydot(G).create_svg())
    return cast(list[SVG], res)


class DisplayableGraph:
    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G

    def plot_with_states(
        self,
        states: StateArray[Node],
        prop_setter: Optional[Callable[[nx.DiGraph], None]] = None,
        scale: float = 1.0,
        max_node_width: float = 1.1,
    ) -> list[SVG]:
        G: nx.DiGraph = self.G
        res = [None] * len(states)

        G.graph["graph"] = {"layout": "neato", "scale": scale}  # type: ignore

        G.graph["node"] = {  # type: ignore
            "fontsize": 10,
            "shape": "circle",
            "style": "filled",
            "fillcolor": "#f0fff4",
            "fixedsize": True,
        }

        if prop_setter is not None:
            prop_setter(G)

        max_weight: float = max(map(lambda x: x[2]["weight"], G.edges(data=True)))
        min_weight: float = min(map(lambda x: x[2]["weight"], G.edges(data=True)))
        if np.allclose(max_weight, min_weight):
            calc_edge_width: Callable[[float], float] = lambda x: 2.5
        else:
            calc_edge_width = linear_func_from_2_points(
                (min_weight, 0.8), (max_weight, 4.5)
            )

        preserve_pos_when_plotting(G)

        # adding big "void" transparent nodes to preserve layout when
        # width of a node is changed dynamically
        void_node_dict = {}
        for v in G.nodes:
            G.nodes[v]["tooltip"] = str(v)
            void_node_dict[("void", v)] = {
                "pos": G.nodes[v]["pos"],
                "style": "invis",
                "label": "",
                "color": "transparent",
                "fillcolor": "transparent",
                "tooltip": str(v),
                "width": max_node_width,
            }
        G.add_nodes_from(void_node_dict.items())

        for u, v in G.edges:
            weight = self._G.edges[u, v]["weight"]
            G.edges[u, v]["label"] = f"<<B>{weight}</B>>"
            G.edges[u, v]["penwidth"] = calc_edge_width(weight)
            G.edges[u, v]["arrowsize"] = 0.5
            G.edges[u, v]["fontsize"] = 14
            G.edges[u, v]["fontcolor"] = "black"
            G.edges[u, v]["color"] = "#f3ad5c99"

        cpu_count = os.cpu_count()
        n_pools = min(cpu_count if cpu_count else 1, len(states.states_arr))
        pool_obj = multiprocessing.Pool(n_pools)
        answer: list[list[SVG]] = pool_obj.starmap(
            parallel_plot,
            zip(
                const_iter(G),
                const_iter(states),
                parallelize_range(n_pools, range(len(res))),
            ),
        )
        return flatten(answer)
