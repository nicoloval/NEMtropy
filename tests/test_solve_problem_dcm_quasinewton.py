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
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="quasinewton",
            max_steps=400,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # debug
        # print('\ntest 0: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)

    def test_1(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="quasinewton",
            max_steps=100,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 1: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-2)

    @unittest.skip("skip large graph")
    def test_2(self):
        # test Matrix 1
        n, seed = (40, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        # print(A)

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="quasinewton",
            max_steps=100,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 2: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-2)

    @unittest.skip("skip large graph")
    def test_3(self):
        # test Matrix 1
        n, seed = (40, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="quasinewton",
            max_steps=100,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 3: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-2)

    @unittest.skip("skip large graph")
    def test_emi(self):

        n, seed = (50, 1)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g._solve_problem(
            model="dcm",
            method="quasinewton",
            max_steps=3000,
            verbose=False,
            initial_guess="uniform",
            linsearch="False",
        )

        g._solution_error()
        err = g.error
        # debug
        print("\ntest emi: error = {}".format(err))

        # test result
        self.assertTrue(err < 1)


if __name__ == "__main__":
    unittest.main()
