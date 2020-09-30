import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_qn_0(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)


        g._solve_problem(model='dcm', method='quasinewton', max_steps=400, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print('\ntest 0: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


    def test_qn_1(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        """
        print('test 1:')
        print(A)
        print(g.dseq)
        print(g.r_dseq)
        print(g.r_x, g.r_y)
        print(g.x, g.y)
        """
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 1: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-2)


    @unittest.skip("skip large graph")
    def test_qn_2(self):
        """classes with cardinality 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (40, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        # print(A)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        """
        print('test 1:')
        print(A)
        print(g.dseq)
        print(g.r_dseq)
        print(g.r_x, g.r_y)
        print(g.x, g.y)
        """
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 2: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-2)


    @unittest.skip("skip large graph")
    def test_qn_3(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (40, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 3: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-2)


if __name__ == '__main__':
    unittest.main()

