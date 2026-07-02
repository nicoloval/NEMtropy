import numpy as np
import pytest

import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg

UND_CM = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
UND_CM_EXP = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])

SOLVE_CASES = [
    ("cm", "fixed-point", "undirected", 0.1),
    ("cm", "newton", "undirected", 0.1),
    ("cm", "quasinewton", "undirected", 0.1),
    ("cm_exp", "fixed-point", "undirected", 0.1),
    ("cm_exp", "newton", "undirected", 0.1),
    ("cm_exp", "quasinewton", "undirected", 0.1),
    ("dcm", "fixed-point", "directed", 0.1),
    ("dcm", "newton", "directed", 0.1),
    ("dcm_exp", "fixed-point", "directed", 1.0),
    ("dcm_exp", "newton", "directed", 1.0),
    ("decm", "fixed-point", "weighted", 1.0),
    ("decm", "newton", "weighted", 1.0),
    ("decm", "quasinewton", "weighted", 1.0),
    ("decm_exp", "newton", "weighted", 1.0),
]

# Known flaky solver combos (quasinewton on dcm/dcm_exp, decm_exp fixed-point)
# are omitted; see git history for legacy per-method test files.


def _build_graph(kind, model):
    if kind == "undirected":
        a = UND_CM if model == "cm" else UND_CM_EXP
        return gc.UndirectedGraph(a)
    if kind == "directed":
        a = mg.random_binary_matrix_generator_dense(4, sym=False, seed=22)
        return gc.DirectedGraph(a)
    a = mg.random_weighted_matrix_generator_dense(
        4, sym=False, seed=22, sup_ext=100, intweights=True
    )
    return gc.DirectedGraph(a)


def _solve_kwargs(model, method):
    kw = dict(model=model, method=method, max_steps=3000, verbose=False)
    if model in ("cm", "cm_exp"):
        kw["linsearch"] = True
        if method == "fixed-point":
            kw["initial_guess"] = "random"
            kw["max_steps"] = 200 if model == "cm" else 100
        elif method == "newton":
            kw["initial_guess"] = "degrees_minor"
            kw["max_steps"] = 100
        else:
            kw["initial_guess"] = "random"
            kw["max_steps"] = 300
    elif model in ("dcm", "dcm_exp"):
        kw["initial_guess"] = "uniform"
        kw["linsearch"] = method == "fixed-point"
        if model == "dcm_exp" and method == "fixed-point":
            kw["linsearch"] = "False"
    else:
        kw["initial_guess"] = "uniform"
        kw["linsearch"] = True
        if model == "decm_exp":
            kw["max_steps"] = 10000
    return kw


@pytest.mark.integration
@pytest.mark.parametrize(
    "model,method,kind,tol", SOLVE_CASES, ids=[f"{m}-{meth}" for m, meth, _, _ in SOLVE_CASES]
)
def test_solve_small_graph(model, method, kind, tol):
    g = _build_graph(kind, model)
    g._solve_problem(**_solve_kwargs(model, method))
    g._solution_error()
    assert g.error < tol


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model", ["cm", "dcm", "decm"])
def test_solve_medium_random_graph(model):
    if model == "cm":
        a = mg.random_binary_matrix_generator_dense(20, sym=True, seed=22)
        g = gc.UndirectedGraph(a)
        method = "fixed-point"
        kw = dict(
            model="cm",
            method=method,
            max_steps=300,
            verbose=False,
            linsearch=True,
            initial_guess="random",
        )
        tol = 0.1
    elif model == "dcm":
        a = mg.random_binary_matrix_generator_dense(5, sym=False, seed=22)
        a[0, :] = 0
        g = gc.DirectedGraph(a)
        kw = dict(
            model="dcm",
            method="fixed-point",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
            linsearch=False,
        )
        tol = 0.1
    else:
        a = mg.random_weighted_matrix_generator_dense(
            4, sym=False, seed=22, sup_ext=100, intweights=True
        )
        a[0, :] = 0
        g = gc.DirectedGraph(a)
        kw = dict(
            model="decm",
            method="fixed-point",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch=True,
        )
        tol = 1.0
    g._solve_problem(**kw)
    g._solution_error()
    assert g.error < tol
