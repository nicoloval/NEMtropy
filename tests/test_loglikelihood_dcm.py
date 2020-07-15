import sys
sys.path.append('../')
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


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
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'quasinewton')
        x0 = np.concatenate((g.r_x, g.r_y))

	# call loglikelihood function 
        f_sample = -g.step_fun(x0)
        f_correct = 4*np.log(1/2) - 3*np.log(5/4)
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
        n, seed = (3, 42)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        # rd
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'quasinewton')
        x0 = g.x0 

        f_sample = -g.fun(x0)
        g.last_model = 'dcm'
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        # f_correct = np.array([3.2, 1.2, 3.2, 1.2])

        # debug
        # print(a)
        # print(x0, x)
        # print('f_sample, f_correct', f_full, f_notrd)

        # test result
        # self.assertTrue(np.allclose(f_full, f_notrd))


    def test_loglikelihood_hessian_diag_dcm(self):
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        c = np.array([1,1,1])
        args = (k_out, k_in, nz_ind_out, nz_ind_in, c)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_sample = sample.loglikelihood_hessian_diag_dcm(x, args)
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
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'fixed-point')
        x0 = 0.5*np.ones(4) 

        f_sample = -g.fun(x0)
        g.last_model = 'dcm'
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        f_correct = - np.array([2.5, 2.5, 0, 0, 1, 1.25])

        # debug
        # print(a)
        # print(x0, x)
        # print(f_full)

        # test result
        self.assertTrue(np.allclose(f_full, f_correct))


    def test_iterative_dcm_1(self):
        degseq = np.array([0, 1, 2, 1, 2, 2, 2, 0, 2, 0])

        # rd
        g = sample.DirectedGraph(degree_sequence = degseq)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'fixed-point')
        x0 = np.ones(6) 
        # x0[x0 == 0] = 0

        f_sample = -g.fun(x0)
        # g._set_solved_problem(f_sample)
        # f_full = np.concatenate((g.x, g.y))
        f_correct = - np.array([0, 0.5, 1, 1, 1, 0])

        # debug
        # print(g.args)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_hessian_dcm_vs_diag(self):
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        c = np.array([1,1,1])
        args = (k_out, k_in, nz_ind_out, nz_ind_in, c)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_diag = sample.loglikelihood_hessian_diag_dcm(x, args)
        f_full = sample.loglikelihood_hessian_dcm(x, args)
        f_df = np.diag(f_full)
        # debug
        # print(f_diag, f_full, f_df)
        

        # test result
        self.assertTrue(np.allclose(f_diag, f_df))


if __name__ == '__main__':
    unittest.main()

