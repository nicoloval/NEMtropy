import numpy as np
import pytest

import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg


@pytest.mark.slow
def test_zscore_3motifs_returns_dict():
    """Smoke test: motif z-scores are computed after DCM solve."""
    n_nodes, seed = 10, 100
    a = mg.random_binary_matrix_generator_dense(n_nodes, sym=False, seed=seed)
    g = gc.DirectedGraph(a)
    g.solve_tool(model="dcm", max_steps=200, verbose=False)
    z = g.zscore_3motifs()
    assert isinstance(z, dict)
    assert len(z) > 0
    assert all(isinstance(v, (float, np.floating)) for v in z.values())
