import sys
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_iterative_0(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        
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

        fun = lambda x: sample.iterative_decm(x, args)
        stop_fun = lambda x: -sample.loglikelihood_decm(x, args)

        sol = sample.solver(x0, fun=fun, g=stop_fun, tol=1e-6, eps=1e-10, max_steps=300, method='fixed-point', verbose=False, regularise=True, full_return = False)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.linalg.norm(ek - k)
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1)


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
        stop_fun = lambda x: -sample.loglikelihood_decm(x, args)

        sol = sample.solver(x0, fun=fun, g=stop_fun, fun_jac=fun_jac, tol=1e-6, eps=1e-10, max_steps=300, method='quasinewton', verbose=False, regularise=True, full_return = False)

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
        stop_fun = lambda x: -sample.loglikelihood_decm(x, args)

        sol = sample.solver(x0, fun=fun, g=stop_fun, fun_jac=fun_jac, tol=1e-6, eps=1e-3, max_steps=300, method='newton', verbose=False, regularise=True, full_return = False)

        ek = sample.expected_decm(sol)
        k = np.concatenate((k_out, k_in, s_out, s_in))
        err = np.linalg.norm(ek - k)
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err< 1e-1)

        
if __name__ == '__main__':
    unittest.main()

