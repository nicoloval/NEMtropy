import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_fixedpoint_cm_5(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        A = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [1, 1, 0],
            ]
        )

        g = sample.UndirectedGraph(A)

        g._solve_problem(
            model="cm_exp",
            method="fixed-point",
            initial_guess = "random",
            max_steps=100,
            verbose=False,
            linsearch=True,
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 5: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


    def test_fixedpoint_cm_6(self):
        """classes with cardinality > 1, no zero degree"""
        n, seed = (20, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=True, seed=seed)

        g = sample.UndirectedGraph(A)

        g._solve_problem(
            model="cm_exp",
            method="fixed-point",
            initial_guess = "random",
            max_steps=300,
            verbose=False,
            linsearch="True",
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 6: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


    @unittest.skip("skip large graph")
    def test_fixedpoint_cm_9(self):
        """classes with cardinality more than 1 and zero degrees"""
        # test Matrix 1
        n, seed = (100, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=True, seed=seed)

        g = sample.UndirectedGraph(A)

        g._solve_problem(
            model="cm_exp",
            method="fixed-point",
            initial_guess = "random",
            max_steps=300,
            verbose=False,
            linsearch="True",
        )

        g._solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 9: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


if __name__ == "__main__":
    unittest.main()
