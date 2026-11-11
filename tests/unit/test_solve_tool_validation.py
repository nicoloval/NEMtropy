"""solve_tool must reject unknown models and methods with a clear error."""
import numpy as np
import pytest

import NEMtropy.ensemble_functions as ef
import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg


@pytest.fixture(scope="module")
def weighted_directed():
    return mg.random_weighted_matrix_generator_dense(
        n=10, sup_ext=10, sym=False, seed=3, intweights=True
    ).astype(float)


@pytest.mark.integration
@pytest.mark.parametrize("model", ["DCM-exp", "cm_exp", "unknown"])
def test_directed_wrong_model(weighted_directed, model):
    g = gc.DirectedGraph(adjacency=weighted_directed)
    with pytest.raises(ValueError, match="Model"):
        g.solve_tool(model=model, method="newton", initial_guess="random")


@pytest.mark.integration
@pytest.mark.parametrize("model", ["CM", "dcm_exp", "unknown"])
def test_undirected_wrong_model(weighted_directed, model):
    g = gc.UndirectedGraph(adjacency=weighted_directed + weighted_directed.T)
    with pytest.raises(ValueError, match="Model"):
        g.solve_tool(model=model, method="newton", initial_guess="random")


@pytest.mark.integration
@pytest.mark.parametrize("cls, model", [(gc.DirectedGraph, "dcm_exp"),
                                        (gc.UndirectedGraph, "cm_exp")])
def test_wrong_method(weighted_directed, cls, model):
    g = cls(adjacency=weighted_directed + weighted_directed.T)
    with pytest.raises(ValueError, match="Method"):
        g.solve_tool(model=model, method="Newton", initial_guess="random")


@pytest.mark.math
@pytest.mark.parametrize("func", [ef.std_dcm_3motif_11, ef.std_dcm_3motif_12])
def test_jitted_3motif_std_matches_python(func):
    rng = np.random.default_rng(0)
    sol = rng.uniform(0.1, 1.0, size=12)
    np.testing.assert_allclose(func(sol), func.py_func(sol), rtol=1e-12)
