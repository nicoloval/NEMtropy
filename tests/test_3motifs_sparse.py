import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import numpy as np
import unittest  # test tool
import NEMtropy.network_functions as mf
import os
import networkx as nx
import NEMtropy.matrix_generator as mg
import NEMtropy.ensemble_functions as en
import scipy.sparse


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_sparse_13(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        lilA = scipy.sparse.lil_matrix(A)

        n = mf.count_3motif_13(lilA)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug

        # test result
        self.assertTrue(n == 6)

    def test_sparse_10(self):
        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        lilA = scipy.sparse.lil_matrix(A)

        n = mf.count_3motif_10_sparse(lilA)
        print(n)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug

        # test result
        self.assertTrue(n == 1)


if __name__ == "__main__":
    unittest.main()
