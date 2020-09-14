import sys
sys.path.append('../')
import Directed_graph_Class as sample
from Directed_new import *
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool
from scipy.optimize import approx_fprime

class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_iterative_decm_new(self):

        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)


        # rd
        g = sample.DirectedGraph(A)
        g.initial_guess='uniform'
        g._initialize_problem('decm', 'fixed-point')
        # theta = np.random.rand(6) 
        theta = 0.5*np.ones(12) 
        x0 = np.exp(-theta)

        f_sample = g.fun(x0)
        f_full = -np.log(f_sample)
        f_new = iterative_decm_new(theta, g.args)


        # f_new_bis = iterative_dcm_new_bis(theta, g.args)
        # print('normale ',f_new)
        # print('bis',f_new_bis)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_full)
        # print(f_new)
        # print(f_full)

        # test result
        # first halves are same, second not
        self.assertTrue(np.allclose(f_full[:6], f_new[:6]))


    def test_loglikelihood_dcm_new(self):

        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        g = sample.DirectedGraph(A)
        g.initial_guess='uniform'
        g._initialize_problem('decm', 'newton')
        # theta = np.random.rand(6) 
        theta = 0.5*np.ones(12) 
        x0 = np.exp(-theta)

        f_sample = g.step_fun(x0) 
        g.last_model = 'decm'
        f_new = -loglikelihood_decm_new(theta, g.args)


        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_sample)
        # print(f_new)

        # test result
        self.assertTrue(np.round(f_sample, decimals=5) == np.round(f_new, decimals=5))


    def test_loglikelihood_prime_dcm_new(self):

        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        g = sample.DirectedGraph(A)
        g.initial_guess='uniform'
        g._initialize_problem('decm', 'newton')
        # theta = np.random.rand(6) 
        theta = 0.5*np.ones(12) 
        x0 = np.exp(-theta)

        f = lambda x: loglikelihood_decm_new(x, g.args)
        f_sample = approx_fprime(theta, f, epsilon=1e-6)  
        f_new = loglikelihood_prime_decm_new(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_new)

        # test result
        self.assertTrue(np.allclose(f_sample,f_new))


    def test_loglikelihood_hessian_dcm_new(self):

        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        g = sample.DirectedGraph(A)
        g.initial_guess='uniform'
        g._initialize_problem('decm', 'newton')
        # theta = np.random.rand(6) 
        theta = 0.5*np.ones(12) 
        x0 = np.exp(-theta)

        f_sample = np.zeros((12, 12))
        for i in range(12):
            f = lambda x: loglikelihood_prime_decm_new(x, g.args)[i]
            f_sample[i, :] = approx_fprime(theta, f, epsilon=1e-6)  

        f_new = loglikelihood_hessian_decm_new(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print('approx',f_sample)
        # print('my',f_new)
        # print('diff',f_sample - f_new)
        # print('max',np.max(np.abs(f_sample - f_new)))

        # test result
        self.assertTrue(np.allclose(f_sample,f_new))


    def test_loglikelihood_hessian_diag_dcm_new(self):

        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        g = sample.DirectedGraph(A)
        g.initial_guess='uniform'
        g._initialize_problem('decm', 'newton')
        # theta = np.random.rand(6)
        theta = 0.5*np.ones(12)
        x0 = np.exp(-theta)

        f_sample = np.zeros(12)
        for i in range(12):
            f = lambda x: loglikelihood_prime_decm_new(x, g.args)[i]
            f_sample[i] = approx_fprime(theta, f, epsilon=1e-6)[i]

        f_new = loglikelihood_hessian_diag_decm_new(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print('approx',f_sample)
        # print('my',f_new)
        # print('diff',f_sample - f_new)
        # print('max',np.max(np.abs(f_sample - f_new)))

        # test result
        self.assertTrue(np.allclose(f_sample,f_new))

    """
    def test_loglikelihood_hessian_diag_dcm_new(self):

        n, seed = (3, 42)
        # a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0]])

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'newton')
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        f_sample = np.zeros(6)
        for i in range(6):
            f = lambda x: loglikelihood_prime_dcm_new(x, g.args)[i]
            f_sample[i] = approx_fprime(theta, f, epsilon=1e-6)[i]  
        g.last_model = 'dcm'
        f_new = loglikelihood_hessian_diag_dcm_new(theta, g.args)


        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_new)

        # test result
        self.assertTrue(np.allclose(f_sample,f_new))


    def test_loglikelihood_hessian_diag_vs_normal_dcm_new(self):

        n, seed = (3, 42)
        # a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0]])

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'newton')
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        g.last_model = 'dcm'
        f_diag = loglikelihood_hessian_diag_dcm_new(theta, g.args)
        f_full = loglikelihood_hessian_dcm_new(theta, g.args)
        f_full_d = np.diag(f_full)


        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_new)

        # test result
        self.assertTrue(np.allclose(f_diag, f_full_d))
    """



if __name__ == '__main__':
    unittest.main()

