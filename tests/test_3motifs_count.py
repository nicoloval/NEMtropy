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

    def test_count_13(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        n = mf.motif13_count(A)

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

    def test_count_2(self):
        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

        n = mf.motif2_count(A)

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

    def test_count_5(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

        n = mf.motif5_count(A)

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

    def test_count_10(self):
        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        n = mf.motif10_count(A)

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
