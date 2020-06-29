import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_qn_2(self):
        """classes with cardinality 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, dtype=np.int64)
        # print(A)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        print('\n test 1, no zeros, n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_qn_3(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, dtype=np.int64)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        print('\n test 2, zeros, dimension n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_qn_dcm_3(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (50, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, dtype=np.int64)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        print('\n test 3, no zeros, dimension n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_qn_dcm_4(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (50, 23)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, dtype=np.int64)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        print('\n test 4, zeros, dimension n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)



if __name__ == '__main__':
    unittest.main()

