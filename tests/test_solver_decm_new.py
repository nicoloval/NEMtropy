"""

# WORKING TESTS

* test_quasinewton_0:
    40x40 matrix with no zeros row
* test_quasinewton_1:
    40x40 matrix with 1 zeros row
* test_newton_2:
    4x4 matrix with no zeros row
* test_newton_3:
    40x40 matrix with no zeros row

# NOT WORKING TESTS

* test_newton_4:
    40x40 matrix with 1 zeros row
* test_iterative_5:
    4x4 matrix with no zeros row


"""

import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool
import Matrix_Generator as mg


class MyTest(unittest.TestCase):


    def setUp(self):
        pass
    @unittest.skip("works")
    def test_quasinewton_0(self):
        n,s = (40,25)

        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 10, sym=False, seed=s, intweights = True)

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.9*np.ones(n*4)
        args = (k_out, k_in, s_out, s_in)

        fun = lambda x: -sample.loglikelihood_prime_decm_new(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_diag_decm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DECM_new(x, (step_fun, ))

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac, linsearch_fun=lin_fun, tol=1e-6, eps=1e-10, max_steps=300, method='quasinewton', verbose=False, regularise=True, full_return = False, linsearch=True)
        sol = np.exp(-sol)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.max(np.abs(ek-k))
        # debug
        # print(ek)
        # print(k)
        print('\ntest 0: error = {}'.format(err))
        print('method = {}, matrix {}x{}'.format('quasinewton', n, n))

        # test result
        self.assertTrue(err< 1e-1)


    @unittest.skip("works")
    def test_quasinewton_1(self):

        n,s = (40,25)

        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 10, sym=False, seed=s, intweights = True)
        A[0,:] = 0

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.9*np.ones(n*4)
        args = (k_out, k_in, s_out, s_in)

        fun = lambda x: -sample.loglikelihood_prime_decm_new(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_diag_decm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DECM_new(x, (step_fun, ))

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac, linsearch_fun=lin_fun, tol=1e-6, eps=1e-10, max_steps=300, method='quasinewton', verbose=False, regularise=True, full_return = False, linsearch=True)
        sol = np.exp(-sol)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.max(np.abs(ek-k))
        # debug
        # print(ek)
        # print(k)
        print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-1)
        print('method = {}, matrix {}x{} with zeros'.format('quasinewton', n, n))


    @unittest.skip("works")
    def test_newton_2(self):
        # x0 relies heavily on x0
        A = np.array([[0, 2, 3, 0],
                      [1, 0, 1, 0],
                      [0, 3, 0, 1],
                      [1, 0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = np.ones(16)
        args = (k_out, k_in, s_out, s_in)

        fun = lambda x: -sample.loglikelihood_prime_decm_new(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_decm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DECM_new(x, (step_fun, ))

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac,linsearch_fun = lin_fun, tol=1e-6, eps=1e-1, max_steps=300, method='newton', verbose=False, regularise=True, full_return = False, linsearch = True)
        sol = np.exp(-sol)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.max(np.abs(ek-k))
        # debug
        # print(ek)
        # print(k)
        print('\ntest 2: error = {}'.format(err))
        print('method = {}, matrix {}x{} '.format('newton', 4, 4))

        # test result
        self.assertTrue(err< 1e-1)


    @unittest.skip("works")
    def test_newton_3(self):
        # convergence relies heavily on x0
        # n, s = (4, 35)
        n, s = (40, 35)
        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 10, sym=False, seed=s, intweights = True)
        # A[0,:] = 0

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        # x0 = 1*np.ones(4*n)
        x0 = np.concatenate((-1*np.ones(2*n), np.ones(2*n)))
        args = (k_out, k_in, s_out, s_in)
        # x0[args == 0] = np.infty

        fun = lambda x: -sample.loglikelihood_prime_decm_new(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_decm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DECM_new(x, (step_fun, ))

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac,linsearch_fun = lin_fun, tol=1e-6, eps=1e-2, max_steps=300, method='newton', verbose=False, regularise=True, full_return = False, linsearch = True)
        sol = np.exp(-sol)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.max(np.abs(ek-k))
        # debug
        # print(ek)
        # print(k)
        print('\ntest 3: error = {}'.format(err))
        print('method: {}, matrix {}x{}'.format('newton', n,n))

        # test result
        self.assertTrue(err< 1e-1)


    # @unittest.skip("doesnt work")
    def test_newton_4(self):
        # convergence relies heavily on x0
        n, s = (4, 35)
        # n, s = (5, 35)
        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 100, sym=False, seed=s, intweights = True)
        A[0,:] = 0

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.2*np.ones(4*n)
        # x0 = np.concatenate((-1*np.ones(2*n), np.ones(2*n)))
        args = (k_out, k_in, s_out, s_in)
        x0[ np.concatenate(args) == 0] = 1e3 

        fun = lambda x: -sample.loglikelihood_prime_decm_new(x, args)
        fun_jac = lambda x: -sample.loglikelihood_hessian_decm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DECM_new(x, (step_fun, ))

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, fun_jac=fun_jac,linsearch_fun = lin_fun, tol=1e-6, eps=1e-5, max_steps=10, method='newton', verbose=True, regularise=True, full_return = False, linsearch = True)
        sol = np.exp(-sol)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.max(np.abs(ek-k))
        # debug
        # print(ek)
        # print(k)
        print('\ntest 4: error = {}'.format(err))
        print('method: {}, matrix {}x{} with zeros'.format('newton', n,n))

        # test result
        self.assertTrue(err< 1e-1)


    @unittest.skip("doesnt work")
    def test_iterative_5(self):
        
        n, s = (4, 35)
        # n, s = (5, 35)
        A = mg.random_weighted_matrix_generator_dense(n, sup_ext = 100, sym=False, seed=s, intweights = True)
        # A[0,:] = 0

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.3*np.ones(16)
        args = (k_out, k_in, s_out, s_in)

        fun = lambda x: sample.iterative_decm_new(x, args)
        step_fun = lambda x: -sample.loglikelihood_decm_new(x, args)
        lin_fun = lambda x: sample.linsearch_fun_DECM_new(x, (step_fun, ))

        sol = sample.solver(x0, fun=fun, step_fun=step_fun, linsearch_fun = lin_fun, tol=1e-6, eps=1e-10, max_steps=3000, method='fixed-point', verbose=True, regularise=True, full_return = False, linsearch=False)

        sol = np.exp(-sol)
        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.max(np.abs(ek-k))
        # debug
        # print(ek)
        # print(k)
        print('\ntest 5: error = {}'.format(err))
        print('method: {}, matrix {}x{} '.format('iterative', n,n))

        # test result
        self.assertTrue(err< 1)

        
if __name__ == '__main__':
    unittest.main()

