import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import NEMtropy.models_functions as mof
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest  # test tool
from scipy.optimize import approx_fprime


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_loglikelihood_dcm(self):
        """
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        """
        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "eigenvalues"
        g._initialize_problem("dcm", "quasinewton")
        x0 = np.concatenate((g.r_x, g.r_y))

        # call loglikelihood function
        f_sample = -g.step_fun(x0)
        f_correct = 4 * np.log(1 / 2) - 3 * np.log(5 / 4)
        # debug
        # print(x0)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(round(f_sample, 3) == round(f_correct, 3))

    def test_loglikelihood_prime_dcm(self):
        """
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        """
        n, seed = (30, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        # rd
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "eigenvalues"
        g._initialize_problem("dcm", "newton")

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        n_rd = len(k_out)
        theta = np.random.rand(2 * n_rd)
        f_sample = np.zeros(n_rd * 2)
        f = lambda x: mof.loglikelihood_dcm(x, g.args)
        f_sample = approx_fprime(theta, f, epsilon=1e-6)
        f_new = mof.loglikelihood_prime_dcm(theta, g.args)

        # debug
        # print(a)
        # print(x0, x)
        # print(f_sample)
        # print(f_new)
        # for i in range(2*n_rd):
        #         if not np.allclose(f_new[i], f_sample[i],atol=1e-1):
        #             print(i)

        # test result
        self.assertTrue(np.allclose(f_sample, f_new, atol=1e-1))

    def test_loglikelihood_hessian_dcm(self):

        # n,s =(3, 1)
        n, s = (30, 1)
        a = mg.random_binary_matrix_generator_custom_density(
            n=n, p=0.15, sym=False, seed=s
        )

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "eigenvalues"
        g._initialize_problem("dcm", "newton")

        k_out = g.args[0]
        k_in = g.args[1]
        nz_index_out = g.args[2]
        nz_index_in = g.args[3]

        n_rd = len(k_out)
        theta = np.random.rand(2 * n_rd)
        f_sample = np.zeros((n_rd * 2, n_rd * 2))
        for i in range(n_rd * 2):
            f = lambda x: mof.loglikelihood_prime_dcm(x, g.args)[i]
            f_sample[i, :] = approx_fprime(theta, f, epsilon=1e-6)

        f_new = mof.loglikelihood_hessian_dcm(theta, g.args)

        """
        for i in range(2*n_rd):
            for j in range(2*n_rd):
                if np.allclose(f_new[i,j], f_sample[i,j], atol=1e-1) == False:
                    print(i,j)
                    print(f_new[i,j])
                    print(f_sample[i,j])
                    print(f_sample[i,j]/f_new[i,j])
        """

        # debug
        # print(theta, x0)
        # print(g.args)
        # print(n_rd/n)
        # print(f_sample)
        # print(f_new)

        # test result
        self.assertTrue(np.allclose(f_sample, f_new, atol=1))

    def test_loglikelihood_hessian_diag_dcm(self):
        a = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
        k_out = np.sum(a > 0, 1)
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        c = np.array([1, 1, 1])
        args = (k_out, k_in, nz_ind_out, nz_ind_in, c)
        x = 0.5 * np.ones(len(k_out) + len(k_in))
        # call loglikelihood function
        f_sample = mof.loglikelihood_hessian_diag_dcm(x, args)
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        # self.assertTrue(np.allclose(f_sample, f_correct))

    def test_iterative_dcm(self):

        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        # rd
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "eigenvalues"
        g._initialize_problem("dcm", "fixed-point")
        x0 = 0.5 * np.ones(4)

        f_sample = -g.fun(x0)
        g.last_model = "dcm"
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        f_correct = -np.array([2.5, 2.5, 0, 0, 1, 1.25])

        # debug
        # print(a)
        # print(x0, x)
        # print(f_full)

        # test result
        self.assertTrue(np.allclose(f_full, f_correct))

    def test_iterative_dcm_1(self):
        degseq = np.array([0, 1, 2, 1, 2, 2, 2, 0, 2, 0])

        # rd
        g = sample.DirectedGraph(degree_sequence=degseq)
        g.degree_reduction()
        g.initial_guess = "uniform"
        g.regularise = "eigenvalues"
        g._initialize_problem("dcm", "fixed-point")
        x0 = np.ones(6)
        # x0[x0 == 0] = 0

        f_sample = -g.fun(x0)
        # g._set_solved_problem(f_sample)
        # f_full = np.concatenate((g.x, g.y))
        f_correct = -np.array([0, 0.5, 1, 1, 1, 0])

        # debug
        # print(g.args)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))

    def test_loglikelihood_hessian_dcm_vs_diag(self):
        a = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
        k_out = np.sum(a > 0, 1)
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        c = np.array([1, 1, 1])
        args = (k_out, k_in, nz_ind_out, nz_ind_in, c)
        x = 0.5 * np.ones(len(k_out) + len(k_in))
        # call loglikelihood function
        f_diag = mof.loglikelihood_hessian_diag_dcm(x, args)
        f_full = mof.loglikelihood_hessian_dcm(x, args)
        f_df = np.diag(f_full)
        # debug
        # print(f_diag, f_full, f_df)

        # test result
        self.assertTrue(np.allclose(f_diag, f_df))


if __name__ == "__main__":
    unittest.main()
