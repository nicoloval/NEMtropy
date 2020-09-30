import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_newton_0(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

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


    def test_newton_1(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        A[0,:] = 0

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


    @unittest.skip("skip large graph")
    def test_newton_2(self):
        """classes with cardinality more than 1 and zero degrees
        """
        # test Matrix 1
        n, seed = (40, 22)
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


    @unittest.skip("skip large graph")
    def test_newton_3(self):
        # test Matrix 1
        n, seed = (40, 22)
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

