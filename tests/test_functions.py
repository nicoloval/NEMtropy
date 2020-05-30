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


    def test_iterative_dcm(self):
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
        g._initialize_problem('dcm', 'fixed-point')
        x0 = 0.5*np.ones(4) 

        f_sample = -g.fun(x0)
        g._set_solved_problem(f_sample)
        f_full = np.concatenate((g.x, g.y))
        f_correct = np.array([2.5, 1.25, 1.25, 2.5, 1.25, 1.25])

        # debug
        # print(a)
        # print(x0, x)

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
        f_correct = np.array([0, 0.5, 1, 1, 1, 0])

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


    def test_loglikelihood__decm(self):
        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.5*np.ones(12)
        args = (k_out, k_in, s_out, s_in)

	# call loglikelihood function 
        f_sample = sample.loglikelihood_decm(x0, args)
        f_correct = 30*np.log(0.5) + 6*np.log(1 - 0.5*0.5) - 6*np.log(1 - 0.5*0.5 + 0.5*0.5*0.5*0.5)
 
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_prime_decm(self):
        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.5*np.ones(12)
        args = (k_out, k_in, s_out, s_in)

	# call loglikelihood function 
        f_sample = sample.loglikelihood_prime_decm(x0, args)
        f_correct = np.array([3.69231, 3.69231, 1.69231, 1.69231, 3.69231, 3.69231, 7.58974, 7.58974, 3.58974, 3.58974, 7.58974, 7.58974])
 
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_hessian_diag_decm(self):
        A = np.array([[0, 2, 2],
                      [2, 0, 2],
                      [0, 2, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        x0 = 0.5*np.ones(12)
        args = (k_out, k_in, s_out, s_in)

	# call loglikelihood function 
        f_sample = sample.loglikelihood_hessian_diag_decm(x0, args)
        f_correct = np.array([ -7.95266272,  -7.95266272,  -3.95266272,  -3.95266272,  -7.95266272, -7.95266272, -16.46285339, -16.46285339,  -8.46285339,  -8.46285339, -16.46285339, -16.46285339])
 
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_decm_all(self):
        A = np.array([[0, 1, 3],
                      [2, 0, 6],
                      [0, 1, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        np.random.seed(seed=30)
        x0 = np.random.rand(12)
        args = (k_out, k_in, s_out, s_in)

	# call loglikelihood function 

        f = sample.loglikelihood_decm(x0, args)
        f_p = sample.loglikelihood_prime_decm(x0, args)
        f_h = sample.loglikelihood_hessian_diag_decm(x0, args)
 
        # print(f)
        # print(f_p)
        # print(f_h)
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        # self.assertTrue(np.allclose(f_sample, f_correct))


    def test_loglikelihood_hessian_decm(self):
        A = np.array([[0, 1, 3],
                      [2, 0, 6],
                      [0, 1, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        np.random.seed(seed=30)
        x0 = np.random.rand(12)
        args = (k_out, k_in, s_out, s_in)

	# call loglikelihood function 

        f_h = sample.loglikelihood_hessian_decm(x0, args)
        f_h_round = np.round(x0, 5)

        f_correct = np.array([[-4.69767, 00, 00, 00, -0.129844, -0.736316, -0.704801, 00, 00, 00, -1.06078, -1.019], 
        [00, -13.7932, 00, -0.104322, 00, -0.141145, 00, -0.318995, 00, -0.0464158, 00, -0.103109], 
        [00, 00, -2.26592, -0.29249, -0.0778545, 00, 00, 00, -0.246331, -0.154498, -0.597806, 00], 
        [00, -0.104322, -0.29249, -37.2973, 00, 00, 00, -0.186842, -0.434652, -0.733954, 00, 00], 
        [-0.129844, 00, -0.0778545, 00, -2.14701, 00, -0.0975086, 00, -0.0957793, 00, -1.12161, 00], 
        [-0.736316, -0.141145, 00, 00, 00, -16.2813, -1.03885, -0.262158, 00, 00, 00, -2.00668], 
        [-0.704801, 00, 00, 00, -0.0975086, -1.03885, -4.95637, 00, 00, 00, -0.914664, -2.41661], 
        [00, -0.318995, 00, -0.186842, 00, -0.262158, 00, -144.924, 00, -0.091131, 00, -0.216482], 
        [00, 00, -0.246331, -0.434652, -0.0957793, 00, 00, 00, -3.00759, -0.286133, -0.797371, 00], 
        [00, -0.0464158, -0.154498, -0.733954, 00, 00, 00, -0.091131, -0.286133, -12.2527, 00, 00], 
        [-1.06078, 00, -0.597806, 00, -1.12161, 00, -0.914664, 00, -0.797371, 00, -109.274, 00], 
        [-1.019, -0.103109, 00, 00, 00, -2.00668, -2.41661, -0.216482, 00, 00, 00, -33.2992]])

        # diff = abs(f_h - f_correct)
        # ind = np.where(diff >1e-1)
        # print(list(zip(ind[0], ind[1])))

        # print(f)
        # print(f_p)
        # print(f_h)
        # print('\n fh ', f_h)
        # print('\n fcorrect ', f_correct)
        # print('diff', f_h - f_correct)
        err = np.linalg.norm(f_h - f_correct, np.inf)
        # debug

        # test result
        # print(np.linalg.norm(f_h-  f_correct, np.inf))
        self.assertTrue(err < 1e-3)
        # self.assertTrue((f_h - f_correct).all() < 1e-1)


    def test_iterative_decm(self):
        A = np.array([[0, 1, 3],
                      [2, 0, 6],
                      [0, 1, 0]])

        bA = np.array([ [1 if aa != 0 else 0 for aa in a] for a in A])

        k_out = np.sum(bA, axis = 1)
        k_in = np.sum(bA, axis = 0)
        s_out = np.sum(A, axis = 1)
        s_in = np.sum(A, axis = 0)

        np.random.seed(seed=30)
        x0 = np.random.rand(12)
        args = (k_out, k_in, s_out, s_in)

	# call loglikelihood function 

        f = x0 - sample.iterative_decm(x0, args)
        f_correct = np.array([ -3.7115928, -29.44267416, -7.11414071, -3.99441366, -12.71558613, -2.71506243, -6.1850179, -64.58546601, -5.26756932, -15.76092613, -1.57639912,  -9.82196987])

        # print(f_p)
        # print(f_h)
        # debug
        # print(par)
        # print(f_sample)
        # print(f_correct)

        # test result
        self.assertTrue(np.allclose(f, f_correct))



if __name__ == '__main__':
    unittest.main()

