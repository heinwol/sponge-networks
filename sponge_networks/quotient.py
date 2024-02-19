"""
This module is devoted to quotient networks and means of dealing with them.

Quotient network is a network resulting from taking a quotient graph of the
original graph.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
import operator
from typing import (
    Any,
    Callable,
    Generic,
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
from copy import copy
import toolz
from sponge_networks.resource_networks import ResourceNetwork

from sponge_networks.utils.utils import (
    K,
    T,
    T1,
    T2,
    V1,
    V2,
    AnyFloat,
    DictMergePolicy,
    Empty,
    Mutated,
    NDArrayT,
    Node,
    StateArray,
    StateArraySlice,
    const,
    const_iter,
    copy_graph,
    flatten,
    linear_func_from_2_points,
    merge_dicts_with_policy,
    parallelize_range,
    set_object_property_nested,
)
from sponge_networks.sponge_networks import SpongeNetwork

QuotientNode: TypeAlias = Node | frozenset[Node]


class QuotientNetwork(Generic[Node]):
    def __init__(
        self, original_network: ResourceNetwork[Node], quotient_vertices: set[set[Node]]
    ) -> None:
        self.original_network = original_network

    @staticmethod
    def generate_quotient_graph(
        G: nx.DiGraph, quotient_vertices: set[set[Node]]
    ) -> nx.DiGraph:
        all_quotient_vertices = flatten(quotient_vertices)
        if not all(v in G.nodes for v in all_quotient_vertices):
            raise ValueError("some unknown vertices encountered")
        if len(set(all_quotient_vertices)) != len(all_quotient_vertices):
            raise ValueError("repeated vertices encountered")

        quotient_entry = reduce(
            lambda x, y: x | y,
            (
                {node: frozenset(nodeset) for node in nodeset}
                for nodeset in quotient_vertices
            ),
            dict[Node, QuotientNode[Node]](),
        )
        for node in G.nodes:
            if node not in quotient_entry:
                quotient_entry[node] = node

        def nodes_merge_policy(key: Any, xs: list[T]) -> T | Empty:
            if key == "pos":
                return Empty()
            if not all(map(lambda x, y: x == y, xs[:-1], xs[1:])):
                raise ValueError(f"cannot merge parameters '{xs=}'")
            return xs[0]

        nodes_to_add: dict[QuotientNode[Node], Any] = {}
        for nodeset in quotient_vertices:
            nodes_to_add[frozenset(nodeset)] = merge_dicts_with_policy(
                (G.nodes[node] for node in nodeset),
                nodes_merge_policy,
            )

        #     node_or_set: copy(G.nodes[node])
        #     for node, node_or_set in quotient_entry.items()
        # }

        G_q = nx.DiGraph()

        # Graph and nodes properties are simply copied
        G_q.graph = copy(G.graph)
        G_q.add_nodes_from(nodes_to_add.items())

        # when it comes to edges it's interesting
        return G_q
