import sys
import os

sys.path.append("../")
import netrecon.Directed_graph_Class as sample
import netrecon.Undirected_graph_Class as sample_und
import netrecon.Matrix_Generator as mg
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
            initial_guess = "random",
            adjacency="dcm",
            max_steps=1000,
            verbose=False,
        )

        g.solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g._weighted_realisation() >= 0).all())

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

        g.solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g._weighted_realisation() >= 0).all())

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
            adjacency="cm-new",
            max_steps=1000,
            verbose=False,
        )

        g.solution_error()

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

        g.solution_error()

        # test result

        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)


if __name__ == "__main__":

    unittest.main()
