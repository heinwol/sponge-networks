from itertools import starmap
from typing import Generator
import pytest

from sponge_networks.resource_networks import *
from sponge_networks.network_manipulation import *
from sponge_networks.utils.utils import *


DescriptorPair = TypedDict(
    "DescriptorPair",
    {
        "node_descriptor": dict[Node, int],
        "idx_descriptor": TypedMapping[int, Node],
    },
)

ProtocolPair = TypedDict(
    "ProtocolPair",
    {
        "result_protocol": pd.DataFrame,
        "standard_protocol": pd.DataFrame,
    },
)


def resourceDiGraph_from_array(arr: npt.ArrayLike) -> ResourceDiGraph:
    ig = np.asarray(arr)
    rn = ResourceDiGraph(nx.from_numpy_array(ig, create_using=nx.DiGraph))
    return rn


rns_with_simulations_raw = [
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
    },
    {
        "rn": resourceDiGraph_from_array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1],
                [4, 1, 1, 1, 1],
                [5, 1, 1, 1, 1],
            ]
        ),
        "simulations": [
            {
                "initial_state": [0, 40, 0, 0, 0],
                "n_iters": 40,
                "correct_file": "g2.xlsx",
            },
        ],
    },
]


class _TestsProvider:
    def __init__(self, rns_with_simulations_raw) -> None:
        res = []
        for d in rns_with_simulations_raw:
            elt = {}
            elt["rn"] = d["rn"]
            elt["simulations"] = [
                {
                    "result_sim": d["rn"].run_simulation(
                        initial_state=it["initial_state"],
                        n_iters=it["n_iters"],
                    ),
                    "standard_protocol": pd.read_excel(
                        f"tests/data/{it['correct_file']}"
                    ).set_index("t", drop=True),
                }
                for it in d["simulations"]
            ]
            res.append(elt)
        self.rns_with_simulations = res

    def all_simulations(self) -> Generator[StateArray, None, None]:
        for d in self.rns_with_simulations:
            for sims in d["simulations"]:
                yield sims["result_sim"]

    def all_descriptor_pairs(self) -> Generator[DescriptorPair, None, None]:
        for d in self.rns_with_simulations:
            yield {
                "idx_descriptor": d["rn"].idx_descriptor,
                "node_descriptor": d["rn"].node_descriptor,
            }

    def all_protocol_pairs(
        self,
    ) -> Generator[ProtocolPair, None, None]:
        for d in self.rns_with_simulations:
            for sims in d["simulations"]:
                yield {
                    "result_protocol": sims["result_sim"].simple_protocol(),
                    "standard_protocol": sims["standard_protocol"],
                }


tests_provider = _TestsProvider(rns_with_simulations_raw)


@pytest.fixture(params=tests_provider.all_simulations())
def simulation(request):
    return request.param


@pytest.fixture(params=tests_provider.all_descriptor_pairs())
def desc_pair(request):
    return request.param


@pytest.fixture(params=tests_provider.all_protocol_pairs())
def protocol_pair(request):
    return request.param


# @pytest.mark.parametrize(
#     "simulation",
#     [(1, 2), (3, 4)],
# )
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


def test_protocols_are_equal(protocol_pair: ProtocolPair):
    # print(protocol_pair["standard_protocol"].columns)
    assert np.allclose(
        protocol_pair["result_protocol"],
        protocol_pair["standard_protocol"],
    )
