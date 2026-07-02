import hashlib
import os

import numpy as np
import pytest

ENSEMBLE_N = int(os.environ.get("NEMTROPY_ENSEMBLE_N", "10"))

import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg
import networkx as nx


def _ensemble_tolerance():
    return max(5.0, 40.0 / ENSEMBLE_N)


def _ensemble_degree_error(g, output_dir, n_samples):
    n = g.n_nodes
    d = {str(i): float(g.dseq[i]) for i in range(n)}
    d_emp = {str(i): 0.0 for i in range(n)}
    counted = 0
    for idx in range(n_samples):
        path = output_dir + f"{idx}.txt"
        if not os.path.isfile(path) or os.stat(path).st_size == 0:
            continue
        g_tmp = nx.read_edgelist(path, nodetype=int)
        d_tmp = dict(g_tmp.degree)
        for item in d_tmp:
            key = str(item)
            if key in d_emp:
                d_emp[key] += d_tmp[item]
        counted += 1
    if counted == 0:
        return float("inf")
    for item in d_emp:
        d_emp[item] /= counted
    diffs = np.array([abs(d[item] - d_emp[item]) for item in d])
    return float(np.linalg.norm(diffs, np.inf))


@pytest.mark.slow
def test_ensemble_cm_binary(ensemble_output_dir):
    """Binary UBCM ensemble: mean degree sequence should be close to observed."""
    n_nodes, seed = 50, 42
    a = mg.random_binary_matrix_generator_dense(n_nodes, sym=True, seed=seed)
    g = gc.UndirectedGraph(a)
    g._solve_problem(
        model="cm",
        method="fixed-point",
        max_steps=100,
        verbose=False,
        linsearch=True,
        initial_guess="uniform",
    )
    g.ensemble_sampler(n=ENSEMBLE_N, output_dir=ensemble_output_dir, seed=42)
    assert _ensemble_degree_error(g, ensemble_output_dir, ENSEMBLE_N) < _ensemble_tolerance()


@pytest.mark.slow
@pytest.mark.skip(reason="Directed/weighted ensembles need n>=100; run legacy suite manually.")
def test_ensemble_dcm_binary(ensemble_output_dir):
    pass


@pytest.mark.slow
@pytest.mark.skip(reason="Directed/weighted ensembles need n>=100; run legacy suite manually.")
def test_ensemble_ecm_weighted(ensemble_output_dir):
    pass


@pytest.mark.slow
@pytest.mark.skip(reason="Directed/weighted ensembles need n>=100; run legacy suite manually.")
def test_ensemble_decm_weighted(ensemble_output_dir):
    pass


@pytest.mark.slow
@pytest.mark.skip(reason="Directed/weighted ensembles need n>=100; run legacy suite manually.")
def test_ensemble_crema_ecm(ensemble_output_dir):
    pass


@pytest.mark.slow
def test_ensemble_seed_not_identical(ensemble_output_dir):
    """Default ensemble seed should vary across calls when not fixed."""
    n_nodes, seed = 10, 42
    a = mg.random_binary_matrix_generator_dense(n_nodes, sym=True, seed=seed)
    g = gc.UndirectedGraph(a)
    g._solve_problem(
        model="cm",
        method="fixed-point",
        max_steps=100,
        verbose=False,
        linsearch=True,
        initial_guess="uniform",
    )
    digests = []
    for _ in range(3):
        g.ensemble_sampler(n=1, output_dir=ensemble_output_dir)
        path = ensemble_output_dir + "0.txt"
        if os.path.isfile(path) and os.stat(path).st_size > 0:
            with open(path, "rb") as handle:
                digests.append(hashlib.md5(handle.read()).hexdigest())
    assert len(digests) >= 2
    assert len(set(digests)) > 1
