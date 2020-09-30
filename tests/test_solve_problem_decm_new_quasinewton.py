import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_fixedpoint_6(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)


        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm_new', method='quasinewton', max_steps=3000, verbose=False, initial_guess='uniform', linsearch = True) 

        g.solution_error()
        # debug
        # print("\n test 1, no zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    def test_fixedpoint_dcm_7(self):
        # test Matrix 1
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)
        A[0,:] = 0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm_new', method='quasinewton', max_steps=3000, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()
        # debug
        # print("\n test 2, zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    @unittest.skip("skip large graph")
    def test_fixedpoint_dcm_9(self):
        # test Matrix 1
        n, seed = (40, 35)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm_new', method='quasinewton', max_steps=25000, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()
        # debug
        # print("\n test 3, no zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)


    @unittest.skip("skip large graph")
    def test_fixedpoint_dcm_10(self):
        # test Matrix 1
        n, seed = (40, 35)
        A = mg.random_weighted_matrix_generator_dense(n, sym=False, seed=seed, sup_ext = 100, intweights=True)
        A[0, :]=0

        g = sample.DirectedGraph(A)

        g._solve_problem(model='decm_new', method='quasinewton', max_steps=20000, verbose=False, initial_guess='uniform', linsearch = True)

        g.solution_error()
        # debug
        # print("\n test 4, zeros, dimension n = {}, error = {}".format(n, g.error))

        # test result
        self.assertTrue(g.error < 1)



if __name__ == '__main__':
    unittest.main()

