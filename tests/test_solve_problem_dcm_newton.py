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
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        # print('\ntest 0  error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1)

    def test_1(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        A[0, :] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        # print('\ntest 1, n={}: error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_2(self):
        n, seed = (40, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        print("\ntest 2, n={}: error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_3(self):
        # test Matrix 1
        n, seed = (40, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="newton",
            max_steps=300,
            verbose=False,
            initial_guess="uniform",
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        print("\ntest 3, n={}: error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)

    @unittest.skip("skip large graph")
    def test_iterative_emi(self):

        n, seed = (50, 1)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g._solve_problem(
            model="dcm",
            method="newton",
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
