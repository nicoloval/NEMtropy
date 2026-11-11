"""All scipy sparse formats (matrices and arrays) must give the same results as dense input."""
import numpy as np
import pytest
from scipy import sparse

import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg
import NEMtropy.network_functions as nef

SPARSE_FORMATS = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.coo_matrix,
    sparse.csr_array,
    sparse.csc_array,
    sparse.coo_array,
]


@pytest.fixture(scope="module")
def weighted_undirected():
    return mg.random_weighted_matrix_generator_dense(
        n=20, sup_ext=10, sym=True, seed=10, intweights=True
    ).astype(float)


@pytest.fixture(scope="module")
def weighted_directed():
    return mg.random_weighted_matrix_generator_dense(
        n=20, sup_ext=10, sym=False, seed=10, intweights=True
    ).astype(float)


@pytest.mark.math
@pytest.mark.parametrize("fmt", SPARSE_FORMATS, ids=lambda f: f.__name__)
def test_degree_and_strength_functions(weighted_directed, fmt):
    a = weighted_directed
    s = fmt(a)
    for func in (nef.degree, nef.strength, nef.out_degree, nef.in_degree,
                 nef.out_strength, nef.in_strength):
        np.testing.assert_allclose(func(s), func(a))


@pytest.mark.integration
@pytest.mark.parametrize("fmt", SPARSE_FORMATS, ids=lambda f: f.__name__)
def test_undirected_cm(weighted_undirected, fmt):
    dense = gc.UndirectedGraph(adjacency=weighted_undirected)
    dense.solve_tool(model="cm_exp", method="newton", initial_guess="degrees_minor")
    sp = gc.UndirectedGraph(adjacency=fmt(weighted_undirected))
    sp.solve_tool(model="cm_exp", method="newton", initial_guess="degrees_minor")
    np.testing.assert_allclose(sp.x, dense.x, rtol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("fmt", SPARSE_FORMATS, ids=lambda f: f.__name__)
def test_directed_dcm(weighted_directed, fmt):
    dense = gc.DirectedGraph(adjacency=weighted_directed)
    dense.solve_tool(model="dcm_exp", method="newton", initial_guess="degrees_minor")
    sp = gc.DirectedGraph(adjacency=fmt(weighted_directed))
    sp.solve_tool(model="dcm_exp", method="newton", initial_guess="degrees_minor")
    np.testing.assert_allclose(np.r_[sp.x, sp.y], np.r_[dense.x, dense.y], rtol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("fmt", SPARSE_FORMATS, ids=lambda f: f.__name__)
def test_crema_with_sparse_binary_adjacency(weighted_undirected, fmt):
    binary = (weighted_undirected > 0).astype(float)
    kwargs = dict(model="crema", method="fixed-point", initial_guess="strengths_minor",
                  max_steps=1000)
    dense = gc.UndirectedGraph(adjacency=weighted_undirected)
    dense.solve_tool(adjacency=binary, **kwargs)
    sp = gc.UndirectedGraph(adjacency=weighted_undirected)
    sp.solve_tool(adjacency=fmt(binary), **kwargs)
    np.testing.assert_allclose(np.asarray(sp.beta), np.asarray(dense.beta), rtol=1e-6)
