import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_newton_dcm_10(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        A = np.array([[0, 3, 2],
                      [6, 0, 0],
                      [0, 4, 0],
			], dtype=np.float64)

        g = sample.DirectedGraph(A)


        g._solve_problem(model='decm', method='newton', max_steps=200, verbose=False, initial_guess='uniform')

        g.solution_error()
        # debug
        # print(g.relative_error_strength)

        # test result
        self.assertTrue(g.relative_error_strength < 1e-1)


    def test_newton_dcm_11(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = sample.random_weighted_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()

        # test result
        self.assertTrue(g.relative_error_strength < 1)


if __name__ == '__main__':
    unittest.main()

