import numpy as np
import pytest

import NEMtropy.graph_classes as gc


@pytest.mark.math
def test_degree_reduction_dcm():
    a = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
    g = gc.DirectedGraph()
    g._initialize_graph(a)
    g.degree_reduction()
    g.initial_guess = "degrees"
    g._set_initial_guess("dcm")
    sol = np.concatenate((g.r_x, g.r_y))
    g.last_model = "dcm"
    g._set_solved_problem(sol)
    assert (g.dseq_out == g.x).all()
    assert (g.dseq_in == g.y).all()
