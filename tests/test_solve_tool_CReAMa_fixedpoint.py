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

    def test_crema_dcm_Dianati_random_dense_20(self):

        network = mg.random_weighted_matrix_generator_dense(
            n=20, sup_ext=10, sym=False, seed=None
        )
        network_bin = (network > 0).astype(int)

        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(
            model="crema",
            method="fixed-point",
            initial_guess="random",
            adjacency="dcm",
            max_steps=1000,
            verbose=False,
        )

        g._solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)

    def test_crema_original_Dianati_random_dense_20_dir(self):

        network = mg.random_weighted_matrix_generator_dense(
            n=20, sup_ext=10, sym=False, seed=None
        )
        network_bin = (network > 0).astype(int)

        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(
            model="crema",
            method="fixed-point",
            initial_guess = "random",
            adjacency=network_bin,
            max_steps=1000,
            verbose=False,
        )

        g._solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)

    def test_crema_cm_Dianati_random_dense_20(self):

        network = mg.random_weighted_matrix_generator_dense(
            n=20, sup_ext=10, sym=True, seed=None
        )
        network_bin = (network > 0).astype(int)

        g = sample_und.UndirectedGraph(adjacency=network)

        g.solve_tool(
            model="crema",
            method="fixed-point",
            initial_guess = "random",
            adjacency="cm_exp",
            max_steps=1000,
            verbose=False,
        )

        g._solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)

    def test_crema_original_Dianati_random_dense_20_undir(self):

        network = mg.random_weighted_matrix_generator_dense(
            n=20, sup_ext=10, sym=False, seed=None
        )
        network_bin = (network > 0).astype(int)

        g = sample_und.UndirectedGraph(adjacency=network)

        g.solve_tool(
            model="crema",
            method="fixed-point",
            initial_guess = "random",
            adjacency=network_bin,
            max_steps=1000,
            verbose=False,
        )

        g._solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)


if __name__ == "__main__":

    unittest.main()
