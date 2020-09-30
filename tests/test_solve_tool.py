import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool
import Matrix_Generator as mg


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_dcm(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        A = np.array([[0, 1, 1],
                      [1, 0, 0],
                      [0, 1, 0],
			])

        g = sample.DirectedGraph(A)

        g.solve_tool(model='dcm', method='quasinewton', initial_guess = 'uniform', max_steps=200, verbose=False)

        g.solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 0: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


    def test_dcm_new(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        A = np.array([[0, 1, 1],
                      [1, 0, 0],
                      [0, 1, 0],
			])

        g = sample.DirectedGraph(A)

        g.solve_tool(model='dcm_new', method='quasinewton', initial_guess = 'uniform', max_steps=200, verbose=False)

        g.solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 0: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


    def test_decm(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        n,s = (4,25)

        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 10, sym=False, seed=s, intweights = True)

        g = sample.DirectedGraph(A)

        g.solve_tool(model='decm', method='quasinewton', initial_guess = 'uniform', max_steps=200, verbose=False)

        g.solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 0: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


    def test_decm_new(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        n,s = (4,25)

        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 10, sym=False, seed=s, intweights = True)

        g = sample.DirectedGraph(A)

        g.solve_tool(model='decm_new', method='quasinewton', initial_guess = 'uniform', max_steps=200, verbose=False)

        g.solution_error()
        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)
        # print('\ntest 0: error = {}'.format(g.error))

        # test result
        self.assertTrue(g.error < 1e-1)


if __name__ == '__main__':
    unittest.main()

