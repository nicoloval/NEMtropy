import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_0(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        A = np.array([[0, 10, 1],
                      [3, 0, 0],
                      [0, 1, 0],
			])

        g = sample.DirectedGraph(A)


        g._solve_problem(model='decm', method='quasinewton', max_steps=200, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug

        # test result
        self.assertTrue(g.relative_error_strength < 1e-2)


    def test_qn_1(self):
        """classes with cardinality > 1, no zero degree
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug

        # test result
        self.assertTrue(g.relative_error_strength < 1e-2)


    def test_qn_2(self):
        """classes with cardinality 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (5, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0
        # print(A)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_qn_3(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_qn_dcm_4(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (50, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_qn_dcm_4(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (50, 23)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='quasinewton', max_steps=100, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug

        # test result
        self.assertTrue(g.relative_error_strength < 1)



if __name__ == '__main__':
    unittest.main()

