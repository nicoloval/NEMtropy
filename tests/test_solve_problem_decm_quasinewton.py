import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_qn_2(self):
        """
        * no zeros
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, intweights=True)
        # print(A)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print('\n test 1, no zeros, n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_qn_3(self):
        """
        * zeros
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, intweights=True)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print('\n test 2, zeros, dimension n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_qn_dcm_3(self):
        """
        * no zeros
        """
        # test Matrix 1
        n, seed = (50, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, intweights=True)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print('\n test 3, no zeros, dimension n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_qn_dcm_4(self):
        """
        * zeros
        """
        # test Matrix 1
        n, seed = (50, 23)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext=100, intweights=True)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print('\n test 4, zeros, dimension n = {}, error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)



if __name__ == '__main__':
    unittest.main()

