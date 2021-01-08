import sys
import os

sys.path.append("../")
import Directed_graph_Class as sample
import Undirected_graph_Class as sample_und
import Matrix_Generator as mg
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

        g.solution_error()

        # test result
        self.assertTrue(g.error < 1e-1)


if __name__ == "__main__":

    unittest.main()
