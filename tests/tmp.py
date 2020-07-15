import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_0(self):
        """u been tricked
        """
        # test Matrix 1
        # n, seed = (5, 22)  # caso divergente senza regolarizazione
        n, seed = (50,22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, dtype=np.int64)
        A[0,:] = 0

        print(A)

        g = sample.DirectedGraph(A)

        print(g.dseq_out)
        print(g.dseq_in)
        print(g.out_strength)
        print(g.in_strength)

        g._solve_problem(model='decm', method='fixed-point', max_steps=1500, tol=1e-3, verbose=True, initial_guess='uniform',regularise=False, linsearch=True)

        g.solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print(g.error)
        print('\n test 2 error = {}'.format(g.error))
        print('\n dseq error = {}'.format(g.error_dseq))
        print('\n sseq error = {}'.format(g.error_sseq))
        

        # test result
        self.assertTrue(g.error < 1)


if __name__ == '__main__':
    unittest.main()

