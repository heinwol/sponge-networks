"""
This module is devoted to quotient networks and means of dealing with them.

Quotient network is a network resulting from taking a quotient graph of the
original graph.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from itertools import product
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
    Pair,
    StateArray,
    StateArraySlice,
    all_equals,
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

__all__ = ["QuotientNode", "QuotientNetwork"]

QuotientNode: TypeAlias = Node | frozenset[Node]


class QuotientNetwork(Generic[Node]):
    def __init__(
        self, original_network: ResourceNetwork[Node], quotient_nodes: set[set[Node]]
    ) -> None:
        self.original_network = original_network
        # self.

    @staticmethod
    def _nodes_merge_policy(key: Any, xs: list[T]) -> T | Empty:
        if key == "pos":
            return Empty()
        if not all_equals(xs):
            raise ValueError(f"cannot merge parameters '{xs=}'")
        return xs[0]

    @staticmethod
    def _merge_edges_policy(key: Any, xs: list[T]) -> list[T] | float:
        return float(np.average(xs)) if key == "weight" else xs

    @classmethod
    def _create_edge_by_vertex_sets(
        cls, G: nx.DiGraph, start: frozenset[Node], end: frozenset[Node]
    ) -> dict:
        is_loop = start == end
        edges_to_consider: set[Pair[Node]] = set()
        for start_vertex in start:
            for incident_to_start_vertex in G[start_vertex]:
                if incident_to_start_vertex in end:
                    if not is_loop or start_vertex == incident_to_start_vertex:
                        edges_to_consider.add((start_vertex, incident_to_start_vertex))

        # avg_weight = float(
        #     np.average(
        #         [
        #             (G.edges[e]["weight"] if "weight" in G.edges[e] else 0.0)
        #             for e in edges_to_consider
        #         ]
        #     )
        # )
        res_edge = merge_dicts_with_policy(
            (G.edges[e] for e in edges_to_consider), cls._merge_edges_policy
        )
        # res_edge["weight"] = avg_weight  # type: ignore
        return res_edge

    def generate_quotient_graph(
        self, G: nx.DiGraph, quotient_nodes: set[set[Node]]
    ) -> tuple[nx.DiGraph, dict[Node, frozenset[Node]]]:
        all_quotient_nodes_ = flatten(quotient_nodes)
        if not all(v in G.nodes for v in all_quotient_nodes_):
            raise ValueError("some unknown vertices encountered")
        if len(set(all_quotient_nodes_)) != len(all_quotient_nodes_):
            raise ValueError("repeated vertices encountered")
        all_quotient_nodes = set(all_quotient_nodes_)
        nonquotient_nodes: list[Node] = list(
            filter(lambda v: v not in all_quotient_nodes_, G.nodes)
        )
        quotient_entry: dict[Node, frozenset[Node]] = reduce(
            lambda x, y: x | y,
            (
                {node: frozenset(nodeset) for node in nodeset}
                for nodeset in quotient_nodes
            ),
            dict[Node, frozenset[Node]](),
        )
        quotient_entry |= {node: frozenset({node}) for node in nonquotient_nodes}

        nodes_to_add: dict[QuotientNode[Node], Any] = {}
        for nodeset in quotient_nodes:
            nodes_to_add[frozenset(nodeset)] = merge_dicts_with_policy(
                (G.nodes[node] for node in nodeset),
                self._nodes_merge_policy,
            )

        G_q = nx.DiGraph()

        # Graph and nodes properties are copied more or less without change
        G_q.graph = copy(G.graph)
        G_q.add_nodes_from(nodes_to_add.items())

        # when it comes to edges it's interesting
        edges_to_add: dict[Pair[QuotientNode[Node]], Any] = {}
        for e in G.edges:
            n1, n2 = cast(Pair[Node], e)
            n1_nonq = n1 in nonquotient_nodes
            n2_nonq = n2 in nonquotient_nodes

            # old behaviour is just preserved:
            if n1_nonq and n2_nonq:
                edges_to_add[e] = G.edges[e]
            else:
                if n1_nonq and not n2_nonq:
                    new_e = (n1, quotient_entry[n2])
                elif not n1_nonq and n2_nonq:
                    new_e = (quotient_entry[n1], n2)
                elif not n1_nonq and not n2_nonq:
                    new_e = (quotient_entry[n1], quotient_entry[n2])
                # tertium non datur
                if new_e not in edges_to_add:
                    e_attr = self._create_edge_by_vertex_sets(
                        G, quotient_entry[n1], quotient_entry[n2]
                    )
                    edges_to_add[new_e] = e_attr
        G_q.add_edges_from(edges_to_add)
        return G_q, quotient_entry
