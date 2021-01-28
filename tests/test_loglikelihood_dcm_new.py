import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
from NEMtropy.models_functions import *
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest  # test tool
from scipy.optimize import approx_fprime


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_iterative_dcm_exp(self):

        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        # rd
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "fixed-point")
        # theta = np.random.rand(6)
        theta = np.array([np.infty, 0.5, 0.5, 0.5, np.infty, 0.5])
        x0 = np.exp(-theta)

        f_sample = g.fun(x0)
        g.last_model = "dcm"
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        f_full = -np.log(f_full)
        f_exp = iterative_dcm_exp(theta, g.args)
        g._set_solved_problem_dcm(f_exp)
        f_exp_full = np.concatenate((g.x, g.y))

        # f_exp_bis = iterative_dcm_exp_bis(theta, g.args)
        # print('normale ',f_exp)
        # print('bis',f_exp_bis)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_full)
        # print(f_exp)
        # print(f_exp_full)

        # test result
        self.assertTrue(np.allclose(f_full, f_exp_full))

    def test_loglikelihood_dcm_exp(self):

        n, seed = (3, 42)
        # a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")
        theta = np.array([np.infty, 0.5, 0.5, 0.5, 0.5, 0.5])
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        f_sample = g.step_fun(x0)
        g.last_model = "dcm"
        f_exp = -loglikelihood_dcm_exp(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(f_sample == f_exp)

    def test_loglikelihood_prime_dcm_exp(self):

        n, seed = (3, 42)
        # a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        f = lambda x: loglikelihood_dcm_exp(x, g.args)
        f_sample = approx_fprime(theta, f, epsilon=1e-6)
        g.last_model = "dcm"
        f_exp = loglikelihood_prime_dcm_exp(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(np.allclose(f_sample, f_exp))

    def test_loglikelihood_hessian_dcm_exp(self):

        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")
        theta = np.random.rand(2 * n)
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        f_sample = np.zeros((n * 2, n * 2))
        for i in range(n * 2):
            f = lambda x: loglikelihood_prime_dcm_exp(x, g.args)[i]
            f_sample[i, :] = approx_fprime(theta, f, epsilon=1e-6)

        f_exp = loglikelihood_hessian_dcm_exp(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(np.allclose(f_sample, f_exp))

    def test_loglikelihood_hessian_dcm_exp_emi(self):

        n, s = (50, 1)
        a = mg.random_binary_matrix_generator_custom_density(
            n=n, p=0.15, sym=False, seed=s
        )

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        n_rd = len(k_out)
        theta = np.random.rand(2 * n_rd)
        f_sample = np.zeros((n_rd * 2, n_rd * 2))
        for i in range(n_rd * 2):
            f = lambda x: loglikelihood_prime_dcm_exp(x, g.args)[i]
            f_sample[i, :] = approx_fprime(theta, f, epsilon=1e-6)

        f_exp = loglikelihood_hessian_dcm_exp(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(np.allclose(f_sample, f_exp))

    def test_loglikelihood_hessian_dcm_exp_emi_simmetry(self):

        n, s = (50, 1)
        # n,s =(2, 1)
        a = mg.random_binary_matrix_generator_custom_density(
            n=n, p=0.15, sym=False, seed=s
        )

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        n_rd = len(k_out)
        # print(n_rd/n)
        theta = np.random.rand(2 * n_rd)

        f_exp = loglikelihood_hessian_dcm_exp(theta, g.args)

        for i in range(2 * n_rd):
            for j in range(2 * n_rd):
                if f_exp[i, j] - f_exp[j, i] != 0:
                    print(i, j)
        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(
            np.allclose(f_exp - f_exp.T, np.zeros((2 * n_rd, 2 * n_rd)))
        )

    def test_loglikelihood_hessian_dcm_exp_simmetry(self):

        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")
        theta = np.random.rand(2 * n)
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        f = loglikelihood_hessian_dcm_exp(theta, g.args)
        f_t = f.T

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f-f_t)

        # test result
        self.assertTrue(np.allclose(f - f_t, np.zeros((2 * n, 2 * n))))

    def test_loglikelihood_hessian_diag_dcm_exp(self):

        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")
        theta = np.random.rand(n * 2)
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        f_sample = np.zeros(2 * n)
        for i in range(2 * n):
            f = lambda x: loglikelihood_prime_dcm_exp(x, g.args)[i]
            f_sample[i] = approx_fprime(theta, f, epsilon=1e-6)[i]
        g.last_model = "dcm"
        f_exp = loglikelihood_hessian_diag_dcm_exp(theta, g.args)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(np.allclose(f_sample, f_exp))

    def test_loglikelihood_hessian_diag_vs_normal_dcm_exp(self):

        n, seed = (3, 42)
        # a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "identity"
        g._initialize_problem("dcm", "newton")
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        x0 = np.exp(-theta)

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        g.last_model = "dcm"
        f_diag = loglikelihood_hessian_diag_dcm_exp(theta, g.args)
        f_full = loglikelihood_hessian_dcm_exp(theta, g.args)
        f_full_d = np.diag(f_full)

        # debug
        # print(a)
        # print(theta, x0)
        # print(g.args)
        # print(f_sample)
        # print(f_exp)

        # test result
        self.assertTrue(np.allclose(f_diag, f_full_d))


if __name__ == "__main__":
    unittest.main()
