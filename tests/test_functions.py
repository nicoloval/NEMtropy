import sys
sys.path.append('../')
import Directed_graph_Class as sample
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
        a = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'quasinewton')
        x0 = np.concatenate((g.r_x, g.r_y))

	# call loglikelihood function 
        f_sample = -g.stop_fun(x0)
        f_correct = 8*np.log(1/2) - 6*np.log(5/4)
        # debug
        # print(x0)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(round(f_sample, 3) == round(f_correct, 3))


    def test_loglikelihood_dcm_notrd(self):
        n, seed = (3, 42)
        a = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)
        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        args = (k_out, k_in, nz_ind_out, nz_ind_in)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_sample = sample.loglikelihood_dcm_notrd(x, args )
        f_correct = 8*np.log(1/2) - 6*np.log(5/4)
        # debug
        # print(args)
        # print(f_sample)
        # print(f_correct)

        self.assertTrue(round(f_sample, 3) == round(f_correct, 3))


    def test_loglikelihood_prime_dcm(self):
        """
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        """
        n, seed = (3, 42)
        a = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)

        # rd
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'quasinewton')
        x0 = np.concatenate((g.r_x, g.r_y))

        f_sample = -g.fun(x0)
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        # f_correct = np.array([3.2, 1.2, 3.2, 1.2])

        # not rd
        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        args_notrd = (k_out, k_in, nz_ind_out, nz_ind_in)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_notrd = sample.loglikelihood_prime_dcm_notrd(x, args_notrd)
 
        # debug
        # print(a)
        # print(x0, x)
        # print('f_sample, f_correct', f_full, f_notrd)

        # test result
        self.assertTrue(np.allclose(f_full, f_notrd))


    def test_loglikelihood_prime_dcm_notrd(self):
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        args = (k_out, k_in, nz_ind_out, nz_ind_in)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_sample = sample.loglikelihood_prime_dcm_notrd(x, args)
        f_correct = np.array([-4/5+4,4-4/5, 2-4/5, -4/5+2, -4/5+4, -4/5+4])  
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_prime_dcm_0(self):
        """
        a = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        """
        n, seed = (3, 42)
        a = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)

        # rd
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess='uniform'
        g._initialize_problem('dcm', 'quasinewton')
        x0 = np.concatenate((g.r_x, g.r_y))

        f_sample = -g.fun_jac(x0)
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        # f_correct = np.array([3.2, 1.2, 3.2, 1.2])

        # not rd
        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        args_notrd = (k_out, k_in, nz_ind_out, nz_ind_in)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_notrd = sample.loglikelihood_hessian_diag_dcm_notrd(x, args_notrd)
 
        # debug
        # print(a)
        # print(x0, x)
        # print('f_sample, f_correct', f_full, f_notrd)

        # test result
        self.assertTrue(np.allclose(f_full, f_notrd))



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
        args_notrd = (k_out, k_in, nz_ind_out, nz_ind_in)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_sample = sample.loglikelihood_hessian_diag_dcm(x, args)
        f_correct = sample.loglikelihood_hessian_diag_dcm_notrd(x, args_notrd)
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_hessian_diag_dcm_notrd(self):
        a = np.array([[0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 0]])

        k_out = np.sum(a > 0, 1) 
        k_in = np.sum(a > 0, 0)
        nz_ind_out = np.nonzero(k_out)[0]
        nz_ind_in = np.nonzero(k_in)[0]
        args = (k_out, k_in, nz_ind_out, nz_ind_in)
        x = 0.5*np.ones(len(k_out)+len(k_in))
	# call loglikelihood function 
        f_sample = sample.loglikelihood_hessian_diag_dcm_notrd(x, args)
        f_correct = np.array([8/25-4, 8/25-8, 8/25-4,8/25-4, 8/25-4, 8/25-8])  
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    """
    def test_loglikelihood_dcm_rd(self):
        n, seed = (3, 42)
        A = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)

        # rd pars
        d = sample.scalability_classes(A, 'dcm_rd')
        x0, args_rd = sample.solver_setting(A, 'dcm_rd')
        x_rd = np.random.random(len(args_rd[0])) 

        # standard pars
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        par = (k_out, k_in)
        args = (par, )
        x = sample.rd2full(x_rd, d, 'dcm_rd')

	# call loglikelihood function 
        f_rd = sample.loglikelihood_dcm_rd(x_rd, args_rd)
        f = sample.loglikelihood_dcm(x, args)

        # debug
        print(A)
        print(args_rd[0], args[0])
        print(x_rd, x)
        print(f_rd, f)

        # test result
        self.assertTrue(round(f_rd, 3) == round(f, 3))


    def test_loglikelihood_prime_dcm(self):
        A = np.array([[0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        par = (k_out, k_in)
        args = (par, )
        x = 0.5*np.ones(2*len(k_out)) 
	# call loglikelihood function 
        f_correct = np.array([-4/5+2,4-4/5, 2-4/5, -4/5+2, -4/5+2, -4/5+4])  
        f_sample = sample.loglikelihood_prime_dcm(x, args)

        # debug
        # print(f_sample, f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_prime_dcm_rd(self):
        n, seed = (3, 42)
        A = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)

        # rd pars
        d = sample.scalability_classes(A, 'dcm_rd')
        x0, args_rd = sample.solver_setting(A, 'dcm_rd')
        x_rd = np.random.random(2*len(args_rd[0][0])) 

        # standard pars
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        par = (k_out, k_in)
        args = (par, )
        x = sample.rd2full(x_rd, d, 'dcm_rd')

	# call loglikelihood function 
        f_rd = sample.loglikelihood_prime_dcm_rd(x_rd, args_rd)
        f_rd2full = sample.rd2full(f_rd, d, 'dcm_rd')
        f = sample.loglikelihood_prime_dcm(x, args)

        # debug
        # print(A)
        # print(args_rd)
        # print(args)
        # print(x_rd, x)
        # print(f_rd2full, f)

        # test result
        self.assertTrue(np.allclose(f_rd2full, f))


    def test_loglikelihood_hessian_diag_dcm(self):
        A = np.array([[0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 0]])
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        par = (k_out, k_in)
        args = (par, )
        x = 0.5*np.ones(2+len(k_out))
	# call loglikelihood function 
        f_correct = np.array([8/25-4, 8/25-8, 8/25-4, 8/25-4, 8/25-4, 8/25-8])  
        f_sample = sample.loglikelihood_hessian_diag_dcm(x, args)

        # debug
        # print(f_sample, f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_hessian_diag_dcm_rd(self):
        n, seed = (3, 42)
        A = sample.random_binary_matrix_generator_nozeros(n, sym=False, seed=seed)

        # rd pars
        d = sample.scalability_classes(A, 'dcm_rd')
        x0, args_rd = sample.solver_setting(A, 'dcm_rd')
        x_rd = np.random.random(2*len(args_rd[0][0])) 

        # standard pars
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        par = (k_out, k_in)
        args = (par, )
        x = sample.rd2full(x_rd, d, 'dcm_rd')

	# call loglikelihood function 
        f_rd = sample.loglikelihood_hessian_diag_dcm_rd(x_rd, args_rd)
        f_rd2full = sample.rd2full(f_rd, d, 'dcm_rd')
        f = sample.loglikelihood_hessian_diag_dcm(x, args)

        # debug
        # print(A)
        # print(args_rd)
        # print(args)
        # print(x_rd, x)
        # print(f_rd2full, f)

        # test result
        self.assertTrue(np.allclose(f_rd2full, f))


    def test_loglikelihood_decm(self):
        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])
	# call loglikelihood function 
        # problem fixed parameters
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        s_out = sample.out_strength(A)
        s_in = sample.in_strength(A)
        par = np.concatenate((k_out, k_in, s_out, s_in))
        x = 0.5*np.ones(len(par))

        f_sample = sample.loglikelihood_decm(x, par)
        f_correct = 30*np.log(0.5) + 6*np.log(1 - 0.5*0.5) - 6*np.log(1 - 0.5*0.5 + 0.5*0.5*0.5*0.5)  
        # test result
        self.assertTrue(round(f_sample, 3) == round(f_correct, 3))


    def test_loglikelihood_prime_decm(self):
        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])
	# call loglikelihood function 
        # problem fixed parameters
        k_out = sample.out_degree(A)
        k_in = sample.in_degree(A)
        s_out = sample.out_strength(A)
        s_in = sample.in_strength(A)
        par = np.concatenate((k_out, k_in, s_out, s_in))
        x = 0.5*np.ones(len(par))

        f_sample = sample.loglikelihood_prime_decm(x, par)
        f_correct = np.concatenate((k_out, k_in, s_out, s_in))/x - 2*np.array([2/13, 2/13, 2/13, 2/13, 2/13, 2/13, 8/39, 8/39, 8/39, 8/39, 8/39, 8/39]) 
        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))
"""


if __name__ == '__main__':
    unittest.main()

