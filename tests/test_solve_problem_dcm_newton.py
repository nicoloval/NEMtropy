import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_newton_dcm_0(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        A = np.array([[0, 1, 1],
              [1, 0, 0],
              [0, 1, 0],
                ])


        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        # print('\ntest 0  error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1)



    def test_newton_dcm_1(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (80, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        # print('\ntest 1, n={}: error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_newton_dcm_2(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (100, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        # print('\ntest 2, n={}: error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_newton_dcm_3(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (150, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='dcm', method='newton', max_steps=300, verbose=False, initial_guess='uniform')

        g.solution_error()
        # print('degseq = ', np.concatenate((g.dseq_out, g.dseq_in)))
        # print('expected degseq = ',g.expected_dseq)
        # print(np.concatenate((g.dseq_out, g.dseq_in)) - g.expected_dseq)
        # debug
        # print('\ntest 3, n={}: error = {}'.format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)



if __name__ == '__main__':
    unittest.main()

