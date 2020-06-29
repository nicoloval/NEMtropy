import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_newton_decm_0(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        A = np.array([[0, 2, 3, 0],
                      [1, 0, 1, 0],
                      [0, 3, 0, 1],
                      [1, 0, 2, 0]])


        g = sample.DirectedGraph(A)


        g._solve_problem(model='decm', method='newton', max_steps=200, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        print('\n test 0 error = {}'.format(g.error))
        ##  print(g.error_dseq)

        # test result
        self.assertTrue(g.relative_error_strength < 1e-1)


    def test_newton_decm_1(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, dtype=np.int64)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()

        # debug
        print('\n test 1 error = {}'.format(g.error))
        # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1e-1)


    def test_newton_decm_2(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, dtype=np.int64)
        A[0,:] = 0

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


    def test_newton_decm_3(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (50, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, dtype=np.int64)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()

        # print(g.expected_dseq)
        # print(g.dseq_out,g.dseq_in)
        print('\n test 3 error = {}'.format(g.error))
        # print(g.error_dseq)

        # test result
        self.assertTrue(g.error < 1e-1)



if __name__ == '__main__':
    unittest.main()

