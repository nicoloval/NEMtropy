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


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_13(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        n = mf.count_3motif_13(A)

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

    def test_2(self):
        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

        n = mf.count_3motif_2(A)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug

        # test result
        self.assertTrue(n == 2)

    def test_5(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

        n = mf.count_3motif_5(A)

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

    def test_10(self):
        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        n = mf.count_3motif_10(A)

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
