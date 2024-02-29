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
from pydoc import classname
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
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
from sponge_networks.sponge_networks import SpongeNetwork, SpongeNode

__all__ = ["QuotientNode", "QuotientNetwork"]

QuotientNode: TypeAlias = Node | frozenset[Node]


class _QuotientProperties(Generic[Node]):
    def __init__(self, G: nx.DiGraph, quotient_nodes: Iterable[Iterable[Node]]) -> None:
        quotient_nodes_frozen: list[frozenset[Node]] = list(
            map(frozenset, quotient_nodes)
        )
        all_quotient_nodes_ = flatten(quotient_nodes)

        self._check_quotient_constraints(G, quotient_nodes_frozen, all_quotient_nodes_)
        self.quotient_nodes_sets: set[frozenset[Node]] = set(quotient_nodes_frozen)

        self.all_quotient_nodes: set[Node] = set(all_quotient_nodes_)
        self.nonquotient_nodes: list[Node] = list(
            filter(lambda v: v not in all_quotient_nodes_, G.nodes)
        )
        self.quotient_entry: dict[Node, frozenset[Node]] = reduce(
            lambda x, y: x | y,
            (
                {node: frozenset(nodeset) for node in nodeset}
                for nodeset in quotient_nodes
            ),
            dict[Node, frozenset[Node]](),
        )
        self.quotient_entry |= {
            node: frozenset({node}) for node in self.nonquotient_nodes
        }

    @classmethod
    def _check_quotient_constraints(
        cls,
        G: nx.DiGraph,
        quotient_nodes_frozen: list[frozenset[Node]],
        all_quotient_nodes_: list[Node],
    ) -> None:
        if not all(v in G.nodes for v in all_quotient_nodes_):
            raise ValueError("some unknown vertices encountered")
        if len(set(all_quotient_nodes_)) != len(all_quotient_nodes_):
            raise ValueError("repeated vertices encountered")


class QuotientNetwork(Generic[Node]):
    def __init__(
        self,
        original_network: ResourceNetwork[Node],
        quotient_nodes: Iterable[Iterable[Node]],
    ) -> None:
        self.original_network: ResourceNetwork[Node] = original_network
        self.quotient_properties = _QuotientProperties(
            original_network._G, quotient_nodes
        )

        self.quotient_network: ResourceNetwork[Node] = type(original_network)(
            self.generate_quotient_graph()
        )

    @classmethod
    def _nodes_merge_policy(cls, key: Any, xs: list[T]) -> T | Empty:
        if key == "pos":
            return Empty()
        if not all_equals(xs):
            raise ValueError(f"cannot merge parameters '{xs=}'")
        return xs[0]

    @classmethod
    def _merge_edges_policy(cls, key: Any, xs: list[T]) -> list[T] | float:
        return float(np.average(xs)) if key == "weight" else xs

    @classmethod
    def _create_edge_by_vertex_sets(
        cls, G: nx.DiGraph, start: frozenset[Node], end: frozenset[Node]
    ) -> dict:
        is_loop = start == end
        edges_to_consider: set[Pair[Node]] = set()
        for start_vertex in start:
            for incident_to_start_vertex in G[start_vertex]:
                if incident_to_start_vertex in end and (
                    not is_loop or start_vertex == incident_to_start_vertex
                ):
                    edges_to_consider.add((start_vertex, incident_to_start_vertex))
        res_edge = merge_dicts_with_policy(
            (G.edges[e] for e in edges_to_consider), cls._merge_edges_policy
        )
        return res_edge

    def generate_quotient_graph(self) -> nx.DiGraph:
        G = self.original_network._G
        quotient_entry = self.quotient_properties.quotient_entry

        nodes_to_add: dict[QuotientNode[Node], Any] = {}
        for nodeset in self.quotient_properties.quotient_nodes_sets:
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
            n1_nonq = n1 in self.quotient_properties.nonquotient_nodes
            n2_nonq = n2 in self.quotient_properties.nonquotient_nodes

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
        G_q.add_edges_from((u, v, d) for (u, v), d in edges_to_add.items())
        return G_q

    # def run_simulation(
    #     self, initial_state: dict[QuotientNode[Node], float] | list[float], n_iters: int = 30
    # ) -> StateArray[QuotientNode[Node]]:


class QuotientSpongeNetwork(Generic[Node]):
    """
    Since the notion of a "Sponge Network" is too vague and the ability of the
    `SpongeNetwork` class to reflect is limited and sometimes faulty, no one can
    guarantee that all methods of this class behave correctly with all possible
    equivalence relations.
    """

    def __init__(
        self,
        original_sponge_network: SpongeNetwork[Node],
        quotient_nodes: Iterable[Iterable[Node]],
    ) -> None:
        self.original_sponge_network = original_sponge_network
        quotient_properties = _QuotientProperties[Node](
            original_sponge_network.resource_network._G, quotient_nodes
        )
        new_sink_nodes = set(
            self._gen_quotient_sink_nodes(original_sponge_network, quotient_properties)
        )
        sink_nodes_classes: set[frozenset[Node]] = set(
            filter(lambda qn: isinstance(qn, frozenset), new_sink_nodes)
        )  # type: ignore
        self.quotient_network = QuotientNetwork(
            original_sponge_network.resource_network,
            quotient_properties.quotient_nodes_sets | sink_nodes_classes,
        )
        quotient_upper_nodes = self._normalize_upper_nodes(
            original_sponge_network, self.quotient_network
        )
        # here we ignore the error saying "`ResourceNetworkGreedy` is
        # incompatible with `ResourceNetwork`" since we *actually have*
        # `ResourceNetworkGreedy`. Alas, since python's type system prohibits
        # generic `TypeVar`s we couldn't make `QuotientNetwork` generic in
        # `ResourceNetworkT[Node]`, so as soon as we make a quotient network
        # we lose all the information about the concrete type of network
        # (during static analysis only, of course)
        self.quotient_sponge_network = SpongeNetwork(
            resource_network=self.quotient_network.quotient_network,  # type: ignore
            upper_nodes=quotient_upper_nodes,
            sink_nodes=new_sink_nodes,
            built_with=original_sponge_network.built_with,
            builder_is_correct=False,
        )

    @classmethod
    def _gen_quotient_sink_nodes(
        cls,
        original_sponge_network: SpongeNetwork[Node],
        quotient_properties: _QuotientProperties[Node],
    ) -> frozenset[QuotientNode[Node]]:
        if len(original_sponge_network.sink_nodes) == 0:
            return frozenset(quotient_properties.quotient_nodes_sets)

        G = original_sponge_network.resource_network._G
        sink_nodes = set(original_sponge_network.sink_nodes)
        res: list[frozenset[Node]] = []

        for nodeset in quotient_properties.quotient_nodes_sets:
            new_equivalence_class: list[Node] = []
            for node in nodeset:
                for adjacent_node in G[node]:
                    if adjacent_node in sink_nodes:
                        new_equivalence_class.append(adjacent_node)
            if not len(new_equivalence_class) == 0:
                res.append(frozenset(new_equivalence_class))
        return frozenset(res)

    @classmethod
    def _normalize_upper_nodes(
        cls,
        original_sponge_network: SpongeNetwork[Node],
        quotient_network: QuotientNetwork[Node],
    ) -> list[QuotientNode[Node]]:
        qp = quotient_network.quotient_properties
        new_upper_nodes: list[QuotientNode[Node]] = []
        visited_nodesets: set[frozenset[Node]] = set()

        for node in original_sponge_network.upper_nodes:
            if node in qp.all_quotient_nodes:
                respective_node_set = qp.quotient_entry[node]
                if respective_node_set not in visited_nodesets:
                    new_upper_nodes.append(respective_node_set)
                    visited_nodesets.add(respective_node_set)
            else:
                new_upper_nodes.append(node)
        return new_upper_nodes
