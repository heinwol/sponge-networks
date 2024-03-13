from abc import ABC, abstractmethod
import copy

from dataclasses import dataclass, field
import os
import io
import re
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Literal,
    Optional,
    Protocol,
    Self,
    Sequence,
    TypeAlias,
    TypeVarTuple,
    Union,
    cast,
    TypeVar,
    final,
)
import networkx as nx
import numpy as np
from returns.primitives.hkt import Kind1
from toolz import valmap
import cairosvg
from IPython.core.display import SVG
from IPython.display import display
from PIL import Image
from ipywidgets import widgets

from svgpathtools import svg2paths2

from sponge_networks.utils.utils import (
    T,
    T1,
    T2,
    AnyFloat,
    Mutated,
    NDArrayT,
    Node,
    StateArray,
    StateArraySlice,
    check_cast,
    const,
    const_iter,
    copy_graph,
    do_multiple,
    flatten,
    linear_func_from_2_points,
    parallelize_range,
    set_object_property_nested,
)
import dill, multiprocessing

# see https://stackoverflow.com/a/69253561/10240583
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads  # type: ignore
multiprocessing.reduction.ForkingPickler = dill.Pickler  # type: ignore
multiprocessing.reduction.dump = dill.dump  # type: ignore

__all__ = [
    "DrawableGraphWithContext",
    "MaybeDrawablePropertySetterOrFn",
    "JustDrawable",
    "SimulationContext",
    "SimulationWithChangingWidthContext",
    "SimulationWithChangingWidthDrawable",
    "display_svgs_interactively",
    "svgs_to_gif",
]


def _parse_svg_distance_attribute(attr: str) -> tuple[int, str]:
    rx = re.compile(r"(\d+)(\w*)")
    num, metrics = rx.match(attr).groups()  # type: ignore
    return (int(num), metrics)


def _format_vertex_resource_for_simulation_display(x: AnyFloat | int) -> str:
    if np.isclose(x, 0):
        return "0"
    if isinstance(x, int):
        return str(x)
    x_int, x_frac = int(x), x % 1  # type: ignore
    if x_frac < 1e-3 or x >= 1e3:  # type: ignore
        rem = ""
    else:
        len_int = len(str(x_int))
        rem = str(int(x_frac * 10 ** (4 - len_int)))
        rem = ("0" * (4 - len_int - len(rem)) + rem).rstrip("0")
    return str(x_int) + "." + rem


def _gen_graph_pydot_layout(G: nx.DiGraph) -> None:
    layout = nx.nx_pydot.pydot_layout(G, prog="neato")
    layout_new = valmap(lambda x: (x[0] / 45, x[1] / 45), layout)
    for v in G.nodes:
        G.nodes[v]["pos"] = f"{layout_new[v][0]},{layout_new[v][1]}!"


def _ensure_graph_layout(G: nx.DiGraph) -> None:
    set_object_property_nested(G.graph, {"graph": {"layout": "neato"}}, priority="left")
    # if not ("graph" in G.graph and "layout" in G.graph["graph"]):
    #     G.graph["graph"]["layout"] = "neato"  # some magic happens here
    if all(map(lambda node: "pos" in G.nodes[node], G.nodes)):
        for node in G.nodes:
            node_d: dict = G.nodes[node]
            if isinstance(node_d["pos"], Sequence):
                node_d["pos"] = f"{node_d['pos'][0]},{node_d['pos'][1]}!"
            elif isinstance(node_d["pos"], str):
                pass
            else:
                raise ValueError(
                    f"""
                        Attribute "pos" of every node should be either Sequence or str,
                        however on node '{node}' attribute "pos" is of type '{type(node_d["pos"])}'
                        with value '{node_d["pos"]}'
                        """
                )
    else:
        _gen_graph_pydot_layout(G)


def _add_void_nodes(G: nx.DiGraph, node_width: float, scale: float) -> None:
    """
    adding big "void" transparent nodes to preserve layout when
    width of a node is changed dynamically
    """
    void_node_dict = {}
    void_edges = []
    for v in G.nodes:
        G.nodes[v]["tooltip"] = str(v)
        void_node_dict[("void", v)] = {
            "pos": G.nodes[v]["pos"],
            "style": "invis",
            "label": "",
            "color": "transparent",
            "fillcolor": "transparent",
            "tooltip": str(v),
            "width": node_width * scale,
        }
        if (v, v) in G.edges:
            e: dict = G.edges[v, v]
            void_edges.append(
                (
                    ("void", v),
                    ("void", v),
                    {
                        "weight": e["weight"] if "weight" in e else 1,
                        "style": "invis",
                        "label": "",
                        "color": "transparent",
                        "fillcolor": "transparent",
                        "arrowsize": 10 * scale,
                    },
                )
            )
    G.add_nodes_from(void_node_dict.items())
    G.add_edges_from(void_edges)


_ContextT = TypeVar("_ContextT", bound=Any)


@dataclass
class DrawableGraphWithContext(Generic[_ContextT]):
    """
    Here `original_graph` is the graph we want to be displayed,
    it should not be modified in any way (copy if needed)

    `drawing_graph`, on the contrary, is the graph we're modifying
    to pass to some display handler. In current architecture it's
    supposed to be unique, though it's not always true. Anyway,
    copy this graph if needed, since it's obviously not e.g. thread
    safe
    """

    original_graph: nx.DiGraph
    display_context: _ContextT
    drawing_graph: nx.DiGraph = field(init=False)

    def __post_init__(self) -> None:
        self.drawing_graph = copy_graph(self.original_graph)

    def property_setter(self) -> None: ...

    def compose_property_setter(
        self, x: "DrawableGraphWithContext[Any]" | Callable[[nx.DiGraph], None] | None
    ) -> Self:
        if not x:
            return self
        if isinstance(x, type(self)):
            assert x.drawing_graph is self.drawing_graph
            self.property_setter = do_multiple(self.property_setter, x.property_setter)
            return self
        if isinstance(x, Callable):
            self.property_setter = do_multiple(
                self.property_setter, lambda: x(self.drawing_graph)
            )
            return self
        raise ValueError(f"invalid parameter: {x}")


MaybeDrawablePropertySetterOrFn: TypeAlias = (
    Callable[[nx.DiGraph], None] | DrawableGraphWithContext[_ContextT] | None
)


@dataclass
class JustDrawableContext:
    scale: float = 1.0


@dataclass
class JustDrawable(DrawableGraphWithContext[JustDrawableContext]):
    @classmethod
    def new(cls, G: nx.DiGraph, scale: Optional[float]) -> Self:
        return cls(G, JustDrawableContext(scale or JustDrawableContext.scale))

    def plot(self) -> SVG:
        G_d = self.drawing_graph
        scale = self.display_context.scale
        set_object_property_nested(
            G_d.graph,
            {
                "graph": {"scale": scale},
                "node": {"fontsize": 8 * scale},
                "edge": {"arrowsize": 0.4 * scale, "fontsize": 8 * scale},
            },
            priority="right",
        )

        _ensure_graph_layout(G_d)

        for u, v in G_d.edges:
            G_d.edges[u, v]["label"] = G_d.edges[u, v]["weight"]

        self.property_setter()

        return SVG(nx.nx_pydot.to_pydot(G_d).create_svg())


@dataclass
class SimulationContext(Generic[Node]):
    sim: StateArray[Node]
    scale: float = 1.0


class _SimulationDrawableT(Protocol[Node]):
    @property
    def display_context(self) -> SimulationContext[Node]: ...


def _display_states_as_list_parrallel(
    drawable: _SimulationDrawableT[Node],
    batch_processor: Callable[
        [_SimulationDrawableT[Node], list[StateArraySlice[Node]]],
        list[T],
    ],
) -> list[T]:
    sim = drawable.display_context.sim
    cpu_count = os.cpu_count()
    n_pools = min(cpu_count or 1, len(sim.states_arr))
    pool_obj = multiprocessing.Pool(n_pools)
    rngs = parallelize_range(n_pools, range(len(sim)))
    states_packs: list[list[StateArraySlice[Node]]] = [
        [sim[i] for i in pack] for pack in rngs
    ]
    res: list[list[T]] = pool_obj.starmap(
        batch_processor,
        zip(const_iter(drawable), states_packs),
    )
    return flatten(res)


@dataclass
class SimulationWithChangingWidthContext(Generic[Node], SimulationContext[Node]):
    max_node_width: float = 1.1


@dataclass
class SimulationWithChangingWidthDrawable(
    Generic[Node], DrawableGraphWithContext[SimulationWithChangingWidthContext[Node]]
):
    total_sum: float = field(init=False)
    min_weight: float = field(init=False)
    max_weight: float = field(init=False)
    calc_edge_width: Callable[[float], float] = field(init=False)
    calc_node_width: Callable[[float], float] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        G = self.drawing_graph
        scale = self.display_context.scale
        self.total_sum = self.display_context.sim[0]["states"].arr.sum()
        self.min_weight = min(map(lambda x: x[2]["weight"], G.edges(data=True)))
        self.max_weight = max(map(lambda x: x[2]["weight"], G.edges(data=True)))
        self.calc_edge_width = (
            const(2.5 * scale)
            if np.allclose(self.max_weight, self.min_weight)
            else linear_func_from_2_points(
                (self.min_weight, 0.6 * scale), (self.max_weight, 4 * scale)
            )
        )
        self.calc_node_width = (
            linear_func_from_2_points((0, 0.3 * scale), (self.total_sum, 1.0 * scale))
            if self.total_sum > 0
            else const(0.3 * scale)
        )

    @classmethod
    def new(
        cls,
        G: nx.DiGraph,
        sim: StateArray[Node],
        scale: Optional[float] = None,
        max_node_width: Optional[float] = None,
    ) -> Self:
        return cls(
            G,
            SimulationWithChangingWidthContext(
                sim=sim,
                scale=(scale or SimulationWithChangingWidthContext.scale),
                max_node_width=(
                    max_node_width or SimulationWithChangingWidthContext.max_node_width
                ),
            ),
        )

    def plot(self) -> list[SVG]:
        G = self.drawing_graph
        scale = self.display_context.scale
        max_node_width = self.display_context.max_node_width

        _ensure_graph_layout(G)

        set_object_property_nested(
            G.graph,
            {
                "graph": {"scale": scale},
                "node": {  # type: ignore
                    "fontsize": 10 * scale,
                    "shape": "circle",
                    "style": "filled",
                    "fillcolor": "#f0fff4",
                    "fixedsize": True,
                },
                "edge": {
                    "arrowsize": 0.4 * scale,
                    "fontsize": 10 * scale,
                    "fontcolor": "black",
                    "color": "#f3ad5c99",
                },
            },
            priority="right",
        )

        for u, v in G.edges:
            weight = self.original_graph.edges[u, v]["weight"]
            G.edges[u, v]["label"] = f"<<B>{weight}</B>>"
            G.edges[u, v]["penwidth"] = self.calc_edge_width(weight)

        self.property_setter()

        _add_void_nodes(G, max_node_width, scale)

        return _display_states_as_list_parrallel(
            self,
            type(self)._plot_batches,  # type: ignore
        )

    def _plot_batches(self, states: list[StateArraySlice[Node]]) -> list[SVG]:
        G = self.drawing_graph
        self.calc_node_width
        res: list[SVG] = [None] * len(states)  # type: ignore
        for n_it, state in enumerate(states):
            for v in G.nodes:
                v = cast(Node, v)
                if "color" not in G.nodes[v] or G.nodes[v]["color"] != "transparent":
                    set_object_property_nested(
                        G.nodes[v],
                        {
                            "label": _format_vertex_resource_for_simulation_display(
                                cast(AnyFloat, state["states"][[v]])
                            ),
                            "width": self.calc_node_width(
                                cast(float, state["states"][[v]])
                            ),
                            "fillcolor": (
                                "#f0fff4"
                                if (
                                    state["states"][[v]] < state["total_output_res"][[v]]  # type: ignore
                                    and not np.isclose(
                                        state["total_output_res"][[v]], 0
                                    )
                                )
                                else "#b48ead"
                            ),
                        },
                        "right",
                    )
            res[n_it] = SVG(nx.nx_pydot.to_pydot(G).create_svg())
        return res


def display_svgs_interactively(
    svgs: list[SVG],
) -> widgets.interactive:

    f = lambda i: display(svgs[i])
    interactive_plot = widgets.interactive(
        f,
        i=widgets.IntSlider(
            min=0,
            max=len(svgs) - 1,
            step=1,
            value=0,
            description="№ of iteration",
        ),
    )
    try:
        all_attrs: list[dict[str, str]] = [
            svg2paths2(io.StringIO(svg.data))[2] for svg in svgs  # type: ignore
        ]
        max_height = max(
            _parse_svg_distance_attribute(attr["height"])[0] for attr in all_attrs
        )
        max_width = max(
            _parse_svg_distance_attribute(attr["width"])[0] for attr in all_attrs
        )
        height_metrics = _parse_svg_distance_attribute(all_attrs[0]["height"])[1]
        width_metrics = _parse_svg_distance_attribute(all_attrs[0]["width"])[1]
        interactive_plot.children[-1].layout.height = str(max_height + 2) + height_metrics  # type: ignore
        # interactive_plot.children[-1].layout.width = str(max_width + 2) + width_metrics  # type: ignore
    finally:
        return interactive_plot


def svgs_to_gif(svgs: list[SVG], fp: str) -> None:
    images = [
        Image.open(
            io.BytesIO(check_cast(bytes, cairosvg.svg2png(bytestring=svg.data))),
            formats=["png"],
        )
        for svg in svgs
    ]

    with open(fp, "wb") as gif:
        images[0].save(
            gif,
            format="gif",
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=800,
            loop=0,
        )
