import sys
sys.path.append('../')
import Directed_graph_Class as sample
from Directed_new import *
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_iterative_0(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        
        a = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 0, 0]])

        k_out = np.array([2,2,3,0])
        k_in = np.array([2,2,2,1])
        """
        nz_index_out = np.array([0,1,2])
        nz_index_in = np.array([0,1,2,3])
        c = np.array([2,1,1])
        """
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='random'
        g.full_return = False
        g._set_initial_guess('dcm')
        g._set_args('dcm')

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args 

        fun = lambda x: sample.iterative_dcm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)
        # print(f, norm)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, linsearch_fun = lin_fun, tol=1e-6, eps=1e-10, max_steps=300, method='fixed-point', verbose=True, regularise=True, full_return = False, linsearch=False)
, linsearch=False
        print('theta_sol', theta_sol)
        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0
        print('full', sol)

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.linalg.norm(ek - k)
        # debug
        print(ek)
        print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1)

    '''
    def test_quasinewton_0(self):
        A = np.array([[0, 2, 3, 0],
                      [1, 0, 1, 0],
                      [0, 3, 0, 1],
                      [1, 0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.9*np.ones(16)
        args = (k_out, k_in, s_out, s_in)

        fun = lambda x: -sample.loglikelihood_prime_decm(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_diag_decm(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm(x, args)

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac, tol=1e-6, eps=1e-10, max_steps=300, method='quasinewton', verbose=False, regularise=True, full_return = False)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.linalg.norm(ek - k)
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-1)


    def test_newton_0(self):
        A = np.array([[0, 2, 3, 0],
                      [1, 0, 1, 0],
                      [0, 3, 0, 1],
                      [1, 0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.9*np.ones(16)
        args = (k_out, k_in, s_out, s_in)

        fun = lambda x: -sample.loglikelihood_prime_decm(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_decm(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm(x, args)

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac, tol=1e-6, eps=1e-3, max_steps=300, method='newton', verbose=False, regularise=True, full_return = False)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.linalg.norm(ek - k)
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-1)
    '''

        
if __name__ == '__main__':
    unittest.main()

