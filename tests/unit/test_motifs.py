import numpy as np
import pytest
import scipy.sparse

import NEMtropy.graph_classes as gc
import NEMtropy.network_functions as mf


A_DIR = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]])


@pytest.mark.math
class TestTwoMotifs:
    @pytest.mark.parametrize(
        "counter,expected",
        [
            (mf.count_2motif_2, 2),
            (mf.count_2motif_1, 2),
            (mf.count_2motif_0, 0),
        ],
    )
    def test_dense(self, counter, expected):
        assert counter(A_DIR) == expected

    def test_zscore_dict(self):
        g = gc.DirectedGraph(A_DIR)
        g.solve_tool(model="dcm", max_steps=200, verbose=False)
        assert isinstance(g.zscore_2motifs(), dict)


@pytest.mark.math
class TestThreeMotifs:
    @pytest.mark.parametrize(
        "matrix,counter,expected",
        [
            (
                np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
                mf.count_3motif_13,
                6,
            ),
            (
                np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
                mf.count_3motif_2,
                2,
            ),
            (
                np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
                mf.count_3motif_5,
                1,
            ),
            (
                np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
                mf.count_3motif_10,
                1,
            ),
        ],
    )
    def test_dense_counts(self, matrix, counter, expected):
        assert counter(matrix) == expected

    @pytest.mark.parametrize(
        "matrix,counter,expected",
        [
            (
                np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
                mf.count_3motif_13,
                6,
            ),
            (
                np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
                mf.count_3motif_2,
                2,
            ),
        ],
    )
    def test_sparse_counts(self, matrix, counter, expected):
        assert counter(scipy.sparse.lil_matrix(matrix)) == expected
