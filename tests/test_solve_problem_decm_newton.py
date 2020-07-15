import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_newton_decm_1(self):
        """ 
        * no zeros
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()

        # debug
        # print('\n test 1, no zeros, dimension n = {} error = {}'.format(n, g.error))
        # # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1)


    def test_newton_decm_2(self):
        """
        * zeros
        * works only with regularise=False
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform', regularise=False)

        g.solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print(g.error)
        # print(g.error_dseq)
        # print('\n test 2, zeros, dimension n = {} error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_newton_decm_3(self):
        """
        * no zeros
        """
        # test Matrix 1
        n, seed = (50, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print('\n test 3, no zeros, dimension n = {} error = {}'.format(n, g.error))
        # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1)




    def test_newton_decm_4(self):
        """
        * zeros
        * works only with regularise=False
        """
        # test Matrix 1
        n, seed = (50, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform', regularise=False)

        g.solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        # print('\n test 4, zeros, dimension n = {} error = {}'.format(n, g.error))
        # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1)



if __name__ == '__main__':
    unittest.main()

