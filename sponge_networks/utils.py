from dataclasses import dataclass
from operator import getitem
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    TypedDict,
    Union,
    cast,
)

import networkx as nx
import numpy as np
from IPython.core.display import SVG, Image
import toolz
from toolz import curry, compose, partition_all

AnyFloat: TypeAlias = Union[float, np.float64]
Node: TypeAlias = Hashable
FlowMatrix: TypeAlias = Sequence[Sequence[Sequence[float]]]
# Dims = TypeVarTuple("Dims", bound=int)
# Dtype = TypeVar("Dtype", bound=np.dtype)
# NDarrayShaped = np.ndarray[Any, np.dtype[Dtype]]
T = TypeVar("T", bound=Any)
T1 = TypeVar("T1", bound=Any)
T2 = TypeVar("T2", bound=Any)
ValT = TypeVar("ValT", bound=Any)
# NDarrayT[np.float64]: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
# NDarrayAny: TypeAlias = np.ndarray[Any, np.dtype[Any]]
NDarrayT = np.ndarray[Any, np.dtype[T]]

MAX_NODE_WIDTH = 1.1


def lmap(f: Callable[[T1], T2], x: Iterable[T1]) -> List[T2]:
    return list(map(f, x))


def tmap(f: Callable[[T1], T2], x: Iterable[T1]) -> Tuple[T2, ...]:
    return tuple(map(f, x))


get = curry(getitem)


def linear_func_from_2_points(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Callable[[float], float]:
    if np.allclose(p2[0], p1[0]):
        if np.allclose(p2[1], p1[1]):
            return lambda x: p1[1]
        else:
            raise ValueError(f"Invalid points for linear function: {p1}, {p2}")
    else:
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = (p1[1] * p2[0] - p2[1] * p1[0]) / (p2[0] - p1[0])
        return lambda x: k * x + b


def parallelize_range(n_pools: int, rng: range) -> Iterator[Tuple[int, ...]]:
    rng_ = list(rng)
    total_len = len(rng_)
    size_of_pool = total_len // n_pools + int(bool(total_len % n_pools))
    return partition_all(size_of_pool, rng_)


def const_iter(x: T) -> Iterator[T]:
    while True:
        yield x


class SimpleNodeArrayDescriptor(Generic[ValT]):
    def __init__(
        self,
        val_descriptor: Mapping[Hashable, int],
        arr: NDarrayT[ValT],
        dims_affected: Optional[Tuple[int, ...]] = None,
    ):
        self.val_descriptor = val_descriptor
        self.arr = arr
        self.dims_affected = set(
            dims_affected if dims_affected is not None else range(len(arr.shape))
        )

    def __getitem__(
        self, key: Union[int, Tuple[int, ...]]
    ) -> Union[NDarrayT[ValT], ValT]:
        # [0, 2] => (arr['lala', 5, 1] => arr[desc['lala'], 5, desc[12]])
        key_ = (key,) if not isinstance(key, tuple) else key
        new_key = list(key_)
        for i in range(len(key_)):
            if i in self.dims_affected:
                new_key[i] = self.val_descriptor[key_[i]]
        new_key = tuple(new_key)
        return self.arr[new_key if len(new_key) != 1 else new_key[0]]


StateArraySlice = TypedDict(
    "StateArraySlice",
    {
        "states": SimpleNodeArrayDescriptor[np.float64],
        "flow": SimpleNodeArrayDescriptor[np.float64],
        "total_output_res": SimpleNodeArrayDescriptor[np.float64],
    },
)


@dataclass
class StateArray:
    node_descriptor: Mapping[Node, int]
    idx_descriptor: Union[List[Node], Mapping[int, Node]]
    states_arr: NDarrayT[np.float64]  # N x M
    flow_arr: NDarrayT[np.float64]  # N x M x M
    total_output_res: NDarrayT[np.float64]  # M

    def __len__(self) -> int:
        return len(self.states_arr)

    def __getitem__(self, time: int) -> StateArraySlice:
        return {
            "states": SimpleNodeArrayDescriptor(
                self.node_descriptor, self.states_arr[time], (0,)
            ),
            "flow": SimpleNodeArrayDescriptor(
                self.node_descriptor, self.flow_arr[time], (0, 1)
            ),
            "total_output_res": SimpleNodeArrayDescriptor(
                self.node_descriptor, self.total_output_res, (0,)
            ),
        }


def parallel_plot(G: nx.DiGraph, states: StateArray, rng: List[int]) -> List[SVG]:
    def my_fmt(x: Union[AnyFloat, int]) -> str:
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
    calc_node_width = linear_func_from_2_points((0, 0.35), (total_sum, 1.1))
    res: List[Optional[SVG]] = [None] * len(rng)
    n_it = 0
    for idx in rng:
        state = states[idx]
        for v in G.nodes:
            if "color" not in G.nodes[v] or G.nodes[v]["color"] != "transparent":
                G.nodes[v]["label"] = my_fmt(cast(float, state["states"][v]))
                G.nodes[v]["width"] = calc_node_width(cast(float, state["states"][v]))

                G.nodes[v]["fillcolor"] = (
                    "#f0fff4"
                    if state["states"][v] < state["total_output_res"][v]
                    else "#b48ead"
                )

        for u, v, d in G.edges(data=True):
            d["label"] = d["weight"]
        res[n_it] = SVG(nx.nx_pydot.to_pydot(G).create_svg())
        n_it += 1
    return cast(List[SVG], res)
