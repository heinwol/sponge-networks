from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
import itertools
from operator import getitem
from os import PathLike
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from IPython.core.display import SVG
from toolz import curry, partition_all, valmap
from typing_extensions import override

# Num = TypeVar("Num", bound=npt.NBitBase)
Node = TypeVar("Node", bound=Hashable)
FlowMatrix: TypeAlias = Sequence[Sequence[Sequence[float]]]
T = TypeVar("T", bound=Any)
T1 = TypeVar("T1", bound=Any)
T2 = TypeVar("T2", bound=Any)
ValT = TypeVar("ValT", bound=Any)
K = TypeVar("K", bound=Any, contravariant=True)
V = TypeVar("V", bound=Any, covariant=True)
V1 = TypeVar("V1", bound=Any, covariant=True)
V2 = TypeVar("V2", bound=Any, covariant=True)
# NDarrayT = np.ndarray[Any, np.dtype[T]]
NDArrayT: TypeAlias = npt.NDArray[T]
Mutated = Annotated[T, "this variable is subject to change"]

AnyFloat: TypeAlias = np.float64


def lmap(f: Callable[[T1], T2], x: Iterable[T1]) -> list[T2]:
    return list(map(f, x))


def tmap(f: Callable[[T1], T2], x: Iterable[T1]) -> tuple[T2, ...]:
    return tuple(map(f, x))


class TypedMapping(Protocol[K, V]):
    def __getitem__(self, key: K, /) -> V: ...

    def __len__(self) -> int: ...

    def __contains__(self, key: K, /) -> bool: ...


@overload
def get(container: TypedMapping[K, V]) -> Callable[[K], V]: ...


@overload
def get(container: TypedMapping[K, V], key: K) -> V: ...


def get(*args, **kwargs):
    _get = curry(getitem)
    return _get(*args, **kwargs)


def linear_func_from_2_points(
    p1: tuple[float, float], p2: tuple[float, float]
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


def parallelize_range(n_pools: int, rng: range) -> Iterator[tuple[int, ...]]:
    rng_ = list(rng)
    total_len = len(rng_)
    size_of_pool = total_len // n_pools + int(bool(total_len % n_pools))
    return partition_all(size_of_pool, rng_)


def const_iter(x: T) -> Iterator[T]:
    while True:
        yield x


def const(x: T) -> Callable[..., T]:
    def inner(*args, **kwargs):
        return x

    return inner


def flatten(x: Iterable[Iterable[T]]) -> list[T]:
    return list(itertools.chain.from_iterable(x))


class NeverError(BaseException):
    """
    You should never encounter this
    """


def set_object_property_nested(
    left: dict,
    right: dict[Hashable, Any | dict],
    priority: Literal["left", "right", "error"],
) -> None:
    for k, v in right.items():
        if k not in left:
            left[k] = v
        elif isinstance(left[k], dict) and isinstance(v, dict):
            set_object_property_nested(left[k], v, priority=priority)
        elif left[k] != v:
            match priority:
                case "error":
                    raise ValueError(f"Cannot merge attributes '{left[k]}' and '{v}'")
                case "left":
                    pass
                case "right":
                    left[k] = v


class Empty: ...


DictMergePolicy: TypeAlias = Callable[[K, list[V1]], V2 | Empty]


def merge_dicts_with_policy(
    ds: Iterable[dict[K, V1]], policy: DictMergePolicy[K, V1, V2]
) -> dict[K, V2]:
    res = {}
    all_keys = set(flatten(d.keys() for d in ds))

    for key in all_keys:
        valid_dicts = list(filter(lambda d: key in d, ds))
        if len(valid_dicts) == 1:
            res[key] = valid_dicts[0][key]
        elif all(isinstance(d[key], dict) for d in valid_dicts):
            res[key] = merge_dicts_with_policy((d[key] for d in valid_dicts), policy)
        else:
            policy_res = policy(key, [d[key] for d in valid_dicts])
            if not isinstance(policy_res, Empty):
                res[key] = policy_res
    return res


def dict_without_keys(d: dict[K, V], keys: Iterable[K]) -> dict[K, V]:
    d_ = copy.copy(d)
    for key in keys:
        if key in d_:
            del d_[key]
    return d_


_GraphT = TypeVar("_GraphT", bound=nx.Graph)


def copy_graph(G: _GraphT, as_view: bool = False) -> _GraphT:
    return cast(_GraphT, G.copy(as_view=as_view))


class DescriptorPair(TypedDict, Generic[Node]):
    node_descriptor: dict[Node, int]
    idx_descriptor: TypedMapping[int, Node]


class SimpleNodeArrayDescriptor(Generic[Node, ValT]):
    """
    This class provides a way to support array operations on graphs which
    set nodes is not `set(range(n))` for some n. E.g. Nodes can be enumerated
    in an unconstrained way or even be of any arbitrary (though `Hashable`) class
    (marked as TypeVar `K`)

    The `val_descriptor` field is what's used to map arbitrary data to indices
    in a (multidinensional) array `arr`. *Warning* The main constraint is that
    `val_descriptor` is the same for each (affected; see below) axis of `arr`.

    One can specify the `dims_affected` parameter to mark the axes where
    the index mapping should occur. The other axes will be accessed as-is.
    """

    def __init__(
        self,
        val_descriptor: dict[Node, int],
        arr: NDArrayT[ValT],
        dims_affected: Optional[tuple[int, ...]] = None,
    ):
        self.val_descriptor = val_descriptor
        self.arr = arr
        self.dims_affected = set(
            dims_affected if dims_affected is not None else range(len(arr.shape))
        )
        self._tuple_is_used = any(isinstance(x, tuple) for x in val_descriptor.keys())

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, key: Node | list[Node]) -> NDArrayT[ValT] | ValT:
        # [0, 2] => (arr['lala', 5, 1] => arr[desc['lala'], 5, desc[12]])
        #
        if self._tuple_is_used and type(key) == tuple:
            raise ValueError(
                f"""
                It seems like a tuple is used as one of value descriptor keys.
                At the same time, a tuple ('{key}') is passed to __getitem__.
                To avoid ambiguity, such behavior is prohibited. Please use
                list ('{list(key)}' if you meant to refer to several elements
                or '[{key}]' if you desired to access element with index 
                '{key}') instead.
                """
            )

        key_: list[Node] = [key] if type(key) not in (list, tuple) else list(key)  # type: ignore
        new_key: list[int] = [0] * len(key_)
        for i in range(len(key_)):
            if i in self.dims_affected:
                new_key[i] = (  # type: ignore
                    self.val_descriptor[key_[i]]
                    if key_[i] in self.val_descriptor
                    else key_[i]
                )
        new_key_t = tuple(new_key)
        return self.arr[new_key_t if len(new_key_t) != 1 else new_key_t[0]]


class StateArraySlice(TypedDict, Generic[Node]):
    states: SimpleNodeArrayDescriptor[Node, AnyFloat]
    flow: SimpleNodeArrayDescriptor[Node, AnyFloat]
    total_output_res: SimpleNodeArrayDescriptor[Node, AnyFloat]


@dataclass
class StateArray(Generic[Node]):
    node_descriptor: dict[Node, int]
    idx_descriptor: TypedMapping[int, Node]
    states_arr: NDArrayT[AnyFloat]  # N x M
    flow_arr: NDArrayT[AnyFloat] = field(repr=False)  # N x M x M
    total_output_res: NDArrayT[AnyFloat]  # M

    def __len__(self) -> int:
        return len(self.states_arr)

    @overload
    def __getitem__(self, time: int) -> StateArraySlice[Node]: ...

    @overload
    def __getitem__(self, time: slice) -> list[StateArraySlice[Node]]: ...

    def __getitem__(
        self, time: int | slice
    ) -> StateArraySlice[Node] | list[StateArraySlice[Node]]:
        def get_StateArraySlice_at_time(t: int) -> StateArraySlice[Node]:
            return {
                "states": SimpleNodeArrayDescriptor(
                    self.node_descriptor,
                    self.states_arr[time],
                    (0,),
                ),
                "flow": SimpleNodeArrayDescriptor(
                    self.node_descriptor,
                    self.flow_arr[time],
                    (0, 1),
                ),
                "total_output_res": SimpleNodeArrayDescriptor(
                    self.node_descriptor,
                    self.total_output_res,
                    (0,),
                ),
            }

        match time:
            case int():
                return get_StateArraySlice_at_time(time)
            case slice():
                start = time.start if time.start else 0
                step = time.step if time.step else 1
                rng = range(start, time.stop, step)
                return list(map(get_StateArraySlice_at_time, rng))
            case _:
                raise ValueError(f"unknown slice type: {type(time)}")

    def simple_protocol(self) -> pd.DataFrame:
        n_iters = len(self)
        vertices = lmap(get(self.idx_descriptor), range(len(self.idx_descriptor)))
        cols = ["t"] + vertices
        data: list[Any] = [None] * n_iters
        for i in range(n_iters):
            data[i] = [i] + [self[i]["states"][[v]] for v in vertices]
        df = pd.DataFrame(columns=cols, data=data)
        return df.set_index("t")

    def to_excel(self, filename: str) -> None:
        df = self.simple_protocol()
        df.to_excel(filename)
