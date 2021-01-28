import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_newton_1(self):
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()

        # debug
        # print('\n test 1, no zeros, dimension n = {} error = {}'.format(n, g.error))
        # # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1)

    def test_newton_2(self):
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
            regularise=True,
        )

        g._solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print(g.error)
        # print(g.error_dseq)
        # print('\n test 2, zeros, dimension n = {} error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_newton_3(self):
        n, seed = (40, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print('\n test 3, no zeros, dimension n = {} error = {}'.format(n, g.error))
        # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_newton_decm_4(self):
        n, seed = (40, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
            regularise=True,
        )

        g._solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print('\n test 4, zeros, dimension n = {} error = {}'.format(n, g.error))
        # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1)


if __name__ == "__main__":
    unittest.main()
