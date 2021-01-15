"""
# all tests works
"""
import sys

sys.path.append("../")
import netrecon.Directed_graph_Class as sample
from netrecon.Directed_new import *
import netrecon.Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_iterative_0(self):

        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g._solve_problem(
            model="dcm_new",
            method="fixed-point",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch="False",
        )

        g.solution_error()
        err = g.error
        # debug
        # print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1)

    def test_iterative_1(self):

        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g._solve_problem(
            model="dcm_new",
            method="fixed-point",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch="False",
        )

        g.solution_error()
        err = g.error
        # debug
        # print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1)

    @unittest.skip("skip large graph")
    def test_iterative_2(self):

        n, seed = (40, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g._solve_problem(
            model="dcm_new",
            method="fixed-point",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch="False",
        )

        g.solution_error()
        err = g.error
        # debug
        # print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1)

    @unittest.skip("skip large graph")
    def test_iterative_3(self):

        n, seed = (40, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g._solve_problem(
            model="dcm_new",
            method="fixed-point",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch="False",
        )

        g.solution_error()
        err = g.error
        # debug
        # print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1)


if __name__ == "__main__":
    unittest.main()
