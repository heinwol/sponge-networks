from itertools import starmap
import pytest

from sponge_networks.resource_networks import *
from sponge_networks.network_manipulation import *
from sponge_networks.utils.utils import *


DescriptorPair = TypedDict(
    "DescriptorPair",
    {"node_descriptor": dict[Node, int], "idx_descriptor": TypedMapping[int, Node]},
)


def resourceDiGraph_from_array(arr: npt.ArrayLike) -> ResourceDiGraph:
    ig = np.asarray(arr)
    rn = ResourceDiGraph(nx.from_numpy_array(ig, create_using=nx.DiGraph))
    return rn


rns_with_simulations = [
    {
        # если заменить 2 на 1 в первой строке, то аттрактором станет другая вершина!
        "rn": resourceDiGraph_from_array(
            [
                [0, 0, 2, 0],
                [0, 0, 0, 4],
                [3, 2, 0, 0],
                [1, 5, 0, 0],
            ]
        ),
        "simulations": [
            {
                "initial_state": [20, 10, 6, 12],
                "n_iters": 200,
                "correct_file": "g1.xlsx",
            },
        ],
    }
]

# @pytest.fixture
# def interesting_graph():
#     ig = np.array(
#         [
#             [0, 0, 2, 0],  # если заменить 2 на 1, то аттрактором станет другая вершина!
#             [0, 0, 0, 4],
#             [3, 2, 0, 0],
#             [1, 5, 0, 0],
#         ]
#     )
#     interesting_graph = ResourceDiGraph(
#         nx.from_numpy_array(ig, create_using=nx.DiGraph)
#     )
#     sim = interesting_graph.run_simulation([20, 10, 6, 12], 200)
#     plot_simulation(interesting_graph, sim)


def test_resource_is_preserved(simulation: StateArray):
    sim_sum = simulation.states_arr.sum(axis=1)
    assert np.allclose(sim_sum[0], sim_sum, atol=1e-7)


def test_descriptors_are_aligned(desc_pair: DescriptorPair):
    assert all(
        desc_pair["idx_descriptor"][v] == k
        for k, v in desc_pair["node_descriptor"].items()
    )


def test_node_descriptor_is_trivial(desc_pair: DescriptorPair):
    assert all(starmap(lambda k, v: k == v, desc_pair["node_descriptor"].items()))


def test_lala():
    print("lala")
    assert 1 == 1


def test_bebe():
    print("bebe")
    assert 2 == 1
