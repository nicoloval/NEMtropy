import numpy as np
import pytest

import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg

DIR_SMALL = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]])


@pytest.mark.integration
class TestSolveToolSmoke:
    def test_dcm(self):
        g = gc.DirectedGraph(DIR_SMALL)
        g.solve_tool(model="dcm", max_steps=200, verbose=False)
        assert g.error < 0.1

    def test_dcm_exp(self):
        g = gc.DirectedGraph(DIR_SMALL)
        g.solve_tool(
            model="dcm_exp",
            method="quasinewton",
            initial_guess="uniform",
            max_steps=200,
            verbose=False,
        )
        assert g.error < 0.1

    def test_decm(self):
        a = mg.random_weighted_matrix_generator_dense(
            4, sup_ext=10, sym=False, seed=25, intweights=True
        )
        g = gc.DirectedGraph(a)
        g.solve_tool(
            model="decm",
            method="quasinewton",
            initial_guess="uniform",
            max_steps=200,
            verbose=False,
        )
        assert g.error < 0.1

    def test_decm_exp(self):
        a = mg.random_weighted_matrix_generator_dense(
            10, sup_ext=10, sym=False, seed=25, intweights=True
        )
        g = gc.DirectedGraph(a)
        g.solve_tool(
            model="decm_exp",
            method="quasinewton",
            initial_guess="uniform",
            max_steps=200,
            verbose=False,
        )
        assert g.error < 0.1


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,method,adjacency_kw",
    [
        ("ecm", "fixed-point", {}),
        ("ecm_exp", "fixed-point", {}),
    ],
)
def test_ecm_weighted_undirected(model, method, adjacency_kw):
    network = mg.random_weighted_matrix_generator_dense(
        n=20, sup_ext=10, sym=True, seed=10, intweights=True
    )
    g = gc.UndirectedGraph(adjacency=network)
    max_steps = 10000 if model == "ecm_exp" else 20000
    g.solve_tool(
        model=model,
        method=method,
        max_steps=max_steps,
        verbose=False,
        initial_guess="random",
        **adjacency_kw,
    )
    g._solution_error()
    assert g.relative_error_strength < 0.1


CREMA_CASES = [
    ("directed", False, False, "dcm"),
    ("directed", False, True, None),
    ("undirected", True, False, "cm_exp"),
    ("undirected", False, True, None),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "graph_kind,sym,use_bin_adj,adjacency",
    CREMA_CASES,
    ids=[
        f"crema-{k}-{'sym' if s else 'asym'}-{'bin' if b else 'model'}"
        for k, s, b, _ in CREMA_CASES
    ],
)
def test_crema_fixedpoint(graph_kind, sym, use_bin_adj, adjacency):
    network = mg.random_weighted_matrix_generator_dense(
        n=20, sup_ext=10, sym=sym, seed=10, intweights=True
    )
    network_bin = (network > 0).astype(int)
    graph_cls = gc.UndirectedGraph if sym else gc.DirectedGraph
    g = graph_cls(adjacency=network)
    adj_kw = (
        {"adjacency": network_bin}
        if use_bin_adj
        else {"adjacency": adjacency}
    )
    g.solve_tool(
        model="crema",
        method="fixed-point",
        initial_guess="random",
        max_steps=1000,
        verbose=False,
        **adj_kw,
    )
    g._solution_error()
    assert g.relative_error_strength < 0.1
