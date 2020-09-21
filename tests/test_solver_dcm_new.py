"""
* test_iterative_0:
    4x4 matrix with 1 zeros row
* test_iterative_1:
    30x30matrix with no zeros row
* test_iterative_2:
    30x30matrix with 1 zeros row
* test_newton_3:
    4x4 matrix with no zeros row
* test_newton_4:
    40x40matrix with no zeros row
* test_newton_5:
    40x40matrix with 1 zeros row
* test_quasinewton_6:
    50x50matrix with no zeros row
* test_quasinewton_7:
    50x50matrix with 1 zeros row

# all tests works
"""
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

        fun = lambda x: iterative_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, linsearch_fun = lin_fun, tol=1e-6, eps=1e-10, max_steps=700, method='fixed-point', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


    def test_iterative_1(self):

        n, seed = (30, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)


        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: iterative_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, linsearch_fun = lin_fun, tol=1e-6, eps=1e-10, max_steps=700, method='fixed-point', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


    def test_iterative_2(self):

        n, seed = (30, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0,:] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: iterative_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun,  linsearch_fun = lin_fun, tol=1e-6, eps=1e-10, max_steps=700, method='fixed-point', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 2: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)



    def test_newton_3(self):

        n, seed = (4, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: -loglikelihood_prime_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac = fun_jac, linsearch_fun = lin_fun, tol=1e-6, eps=1e-10, max_steps=100, method='newton', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 3: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


    def test_newton_4(self):

        n, seed = (40, 56)
        # n, seed = (400, 26)
        # n, seed = (4, 26)
        # n, seed = (5, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: -loglikelihood_prime_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac = fun_jac, linsearch_fun = lin_fun, tol=1e-6, eps=1e-1, max_steps=100, method='newton', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 4: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


    def test_newton_5(self):

        n, seed = (40, 56)
        # n, seed = (400, 26)
        # n, seed = (4, 26)
        # n, seed = (5, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0,:] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: -loglikelihood_prime_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f= fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac = fun_jac, linsearch_fun = lin_fun, tol=1e-6, eps=1e-1, max_steps=100, method='newton', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 5: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


    def test_quasinewton_6(self):

        # n, seed = (40, 56)
        # n, seed = (400, 26)
        n, seed = (50, 26)
        # n, seed = (5, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: -loglikelihood_prime_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_diag_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac = fun_jac, linsearch_fun = lin_fun, tol=1e-6, eps=1e-1, max_steps=100, method='quasinewton', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 6: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


    def test_quasinewton_7(self):

        # n, seed = (40, 56)
        # n, seed = (400, 26)
        n, seed = (50, 26)
        # n, seed = (5, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0,:] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
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

        fun = lambda x: -loglikelihood_prime_dcm_new(x, args)
        step_fun = lambda x: -loglikelihood_dcm_new(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_diag_dcm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DCM_new(x, (step_fun, ))

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac = fun_jac, linsearch_fun = lin_fun, tol=1e-6, eps=1e-1, max_steps=100, method='quasinewton', verbose=False, regularise=True, full_return = False, linsearch=True)

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate((sample.expected_out_degree_dcm(sol), sample.expected_in_degree_dcm(sol)))
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 7: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-2)


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

