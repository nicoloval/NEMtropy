"""
# all tests works
"""
import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
from NEMtropy.models_functions import *
import NEMtropy.matrix_generator as mg
import NEMtropy.solver_functions as sof
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_iterative_0(self):

        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = 100
        args = g.args

        fun = lambda x: iterative_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-10,
            max_steps=700,
            method="fixed-point",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=False,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    def test_iterative_1(self):

        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)

        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = 100
        args = g.args

        fun = lambda x: iterative_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-10,
            max_steps=700,
            method="fixed-point",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=False,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 0: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    @unittest.skip("skip large graph")
    def test_iterative_2(self):

        n, seed = (40, 22)
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
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = 1e3
        args = g.args

        fun = lambda x: iterative_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-10,
            max_steps=700,
            method="fixed-point",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 1: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    @unittest.skip("skip large graph")
    def test_iterative_3(self):

        n, seed = (40, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
        """
        nz_index_out = np.array([0,1,2])
        nz_index_in = np.array([0,1,2,3])
        c = np.array([2,1,1])
        """
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = 1e3
        args = g.args

        fun = lambda x: iterative_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function
        
        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-10,
            max_steps=700,
            method="fixed-point",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 2: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    def test_newton_0(self):

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
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (loglikelihood_dcm_exp, args))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-10,
            max_steps=100,
            method="newton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser=hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 3: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    def test_newton_1(self):

        n, seed = (4, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
        """
        nz_index_out = np.array([0,1,2])
        nz_index_in = np.array([0,1,2,3])
        c = np.array([2,1,1])
        """
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-10,
            max_steps=100,
            method="newton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 3: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    @unittest.skip("skip large graph")
    def test_newton_1(self):

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
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-1,
            max_steps=100,
            method="newton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 4: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    @unittest.skip("skip large graph")
    def test_newton_2(self):

        n, seed = (40, 56)
        # n, seed = (400, 26)
        # n, seed = (4, 26)
        # n, seed = (5, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
        """
        nz_index_out = np.array([0,1,2])
        nz_index_in = np.array([0,1,2,3])
        c = np.array([2,1,1])
        """
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-1,
            max_steps=100,
            method="newton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 5: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    def test_quasinewton_0(self):

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
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_diag_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (loglikelihood_dcm_exp, args))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-1,
            max_steps=100,
            method="quasinewton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 6: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

        # debug
        # print(ek)
        # print(k)
        # print('\ntest 7: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    def test_quasinewton_1(self):

        n, seed = (4, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
        """
        nz_index_out = np.array([0,1,2])
        nz_index_in = np.array([0,1,2,3])
        c = np.array([2,1,1])
        """
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_diag_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (loglikelihood_dcm_exp, args))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-1,
            max_steps=100,
            method="quasinewton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 6: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

        # debug
        # print(ek)
        # print(k)
        # print('\ntest 7: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    @unittest.skip("skip large graph")
    def test_quasinewton_2(self):

        n, seed = (40, 26)
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
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_diag_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-1,
            max_steps=100,
            method="quasinewton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 6: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

        # debug
        # print(ek)
        # print(k)
        # print('\ntest 7: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

    @unittest.skip("skip large graph")
    def test_quasinewton_3(self):

        n, seed = (40, 26)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        a[0, :] = 0

        k_out = np.sum(a, 1)
        k_in = np.sum(a, 0)
        """
        nz_index_out = np.array([0,1,2])
        nz_index_in = np.array([0,1,2,3])
        c = np.array([2,1,1])
        """
        g = sample.DirectedGraph(a)
        g.degree_reduction()
        g.initial_guess = "random"
        g.full_return = False
        g._set_initial_guess("dcm")
        g._set_args("dcm")

        x0 = g.x0
        x0[x0 == 0] = np.infty
        args = g.args

        fun = lambda x: -loglikelihood_prime_dcm_exp(x, args)
        step_fun = lambda x: -loglikelihood_dcm_exp(x, args)
        fun_jac = lambda x: -loglikelihood_hessian_diag_dcm_exp(x, args)
        lin_fun = lambda x: linsearch_fun_DCM_exp(x, (step_fun,))
        hes_reg = sof.matrix_regulariser_function

        f = fun(x0)
        norm = np.linalg.norm(f)

        theta_sol = sof.solver(
            x0,
            fun=fun,
            step_fun=step_fun,
            fun_jac=fun_jac,
            linsearch_fun=lin_fun,
            tol=1e-6,
            eps=1e-1,
            max_steps=100,
            method="quasinewton",
            verbose=False,
            regularise=True,
            full_return=False,
            linsearch=True,
            hessian_regulariser = hes_reg,
        )

        g._set_solved_problem_dcm(theta_sol)
        theta_sol_full = np.concatenate((g.x, g.y))

        sol = np.exp(-theta_sol_full)
        sol[np.isnan(sol)] = 0

        ek = np.concatenate(
            (
                expected_out_degree_dcm(sol),
                expected_in_degree_dcm(sol),
            )
        )
        k = np.concatenate((k_out, k_in))
        err = np.max(abs(ek - k))
        # debug
        # print(ek)
        # print(k)
        # print('\ntest 6: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)

        # debug
        # print(ek)
        # print(k)
        # print('\ntest 7: error = {}'.format(err))

        # test result
        self.assertTrue(err < 1e-2)


if __name__ == "__main__":
    unittest.main()
