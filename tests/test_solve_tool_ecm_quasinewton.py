import sys
import os

sys.path.append("../")
import NEMtropy.graph_classes as sample
import NEMtropy.graph_classes as sample_und
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_ECM_quasinewton_random_dense_20_undir(self):

        network = mg.random_weighted_matrix_generator_dense(
            n=20, sup_ext=10, sym=True, seed=10, intweights=True
        )

        g = sample_und.UndirectedGraph(adjacency=network)

        g.solve_tool(
            model="ecm",
            method="quasinewton",
            max_steps=1000,
            verbose=False,
            initial_guess="random",
        )

        g._solution_error()

        # test result
        self.assertTrue(g.error < 1e-1)


if __name__ == "__main__":

    unittest.main()
