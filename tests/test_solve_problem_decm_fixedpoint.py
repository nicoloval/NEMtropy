import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_fixedpoint_5(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        A = np.array([[0, 2, 3, 0],
                      [1, 0, 1, 0],
                      [0, 3, 0, 1],
                      [1, 0, 2, 0]], dtype=np.float64)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='fixed-point', max_steps=300, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()
        # debug
        # print(g.relative_error_strength)

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_fixedpoint_6(self):
        """classes with cardinality > 1, no zero degree
        """
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='fixed-point', max_steps=300, verbose=False, initial_guess='uniform', linsearch = True) 

        g.solution_error()

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_fixedpoint_dcm_7(self):
        """classes with cardinality 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (5, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='fixed-point', max_steps=300, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_fixedpoint_dcm_8(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='fixed-point', max_steps=300, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()

        # test result
        self.assertTrue(g.relative_error_strength < 1)


    def test_fixedpoint_dcm_9(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (100, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='fixed-point', max_steps=300, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()

        # test result
        self.assertTrue(g.relative_error_strength < 1)


if __name__ == '__main__':
    unittest.main()

