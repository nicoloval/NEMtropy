import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_newton_decm_2(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, dtype=np.int64)
        # A[0,:] = 0

        print(A)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=True, initial_guess='uniform')

        g.solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print(g.error)
        # print(g.error_dseq)
        print('\n test 2 error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


if __name__ == '__main__':
    unittest.main()

