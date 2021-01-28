import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_0(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm_exp",
            method="newton",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch=True,
        )

        g._solution_error()
        # debug
        # print("\n test 0, no zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    def test_1(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm_exp",
            method="newton",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch=True,
        )

        g._solution_error()
        # debug
        # print("\n test 1, zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_2(self):
        n, seed = (40, 35)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm_exp",
            method="newton",
            max_steps=25000,
            verbose=False,
            initial_guess="uniform",
            linsearch=True,
        )

        g._solution_error()
        # debug
        # print("\n test 2, no zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_3(self):
        # test Matrix 1
        n, seed = (40, 35)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm_exp",
            method="newton",
            max_steps=20000,
            verbose=False,
            initial_guess="uniform",
            linsearch=True,
        )

        g._solution_error()
        # debug
        # print("\n test 3, zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


if __name__ == "__main__":
    unittest.main()
