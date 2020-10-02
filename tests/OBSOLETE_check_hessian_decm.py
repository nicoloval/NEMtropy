import numpy as np
import unittest
import sys
from casadi import *

sys.path.append("../")
import Directed_graph_Class as sample
import Matrix_Generator as mg


def casadi_loglikelihood_decm(A):
    """loglikelihood function for decm as a casadi MX object"""
    # problem fixed parameters
    k_out = np.sum(A > 0, 1)
    k_in = np.sum(A > 0, 0)
    s_out = np.sum(A, 1)
    s_in = np.sum(A, 0)
    # casadi MX function calculation
    n = len(k_in)
    x = MX.sym("x", 4 * n)
    f = 0
    for i in range(n):
        if k_out[i]:
            f += k_out[i] * np.log(x[i])
        if k_in[i]:
            f += k_in[i] * np.log(x[i + n])
        if s_out[i]:
            f += s_out[i] * np.log(x[i + 2 * n])
        if s_in[i]:
            f += s_in[i] * np.log(x[i + 3 * n])
        for j in range(n):
            if i != j:
                f += np.log(1 - x[i + 2 * n] * x[3 * n + j])
                f -= np.log(
                    1
                    - x[i + 2 * n] * x[3 * n + j]
                    + x[i + 2 * n] * x[3 * n + j] * x[i] * x[j + n]
                )
    # fun = Function('f', [x], [f)
    return x, f


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_loglikelihood(self):
        # problem initialisation
        A = mg.random_weighted_matrix_generator_dense(n=10, sym=False)
        prova = sample.DirectedGraph(A)
        prova.initial_guess = "random"
        prova._initialize_problem("decm", "quasinewton")
        sol = np.concatenate((prova.x, prova.y, prova.b_out, prova.b_in))

        # casadi functions initialization
        x, f = casadi_loglikelihood_decm(A)
        casadi_fun = Function("f", [x], [f])
        f_og = sample.loglikelihood_decm(sol, prova.args)
        f_casadi = casadi_fun(sol)

        err_loglikelihood = abs(f_og - f_casadi)
        # print('loglikelihood og-casadi = {}'.format(err_loglikelihood))
        self.assertTrue(err_loglikelihood < 1e-10)

    def test_loglikelihood_prime(self):
        # problem initialisation
        A = mg.random_weighted_matrix_generator_dense(n=10, sym=False)
        prova = sample.DirectedGraph(A)
        prova.initial_guess = "random"
        prova._initialize_problem("decm", "quasinewton")
        sol = np.concatenate((prova.x, prova.y, prova.b_out, prova.b_in))

        # casadi functions initialization
        x, f = casadi_loglikelihood_decm(A)
        casadi_fun = Function("f", [x], [f])

        fj = jacobian(f, x)
        casadi_fun_gradient = Function("j", [x], [fj])
        fj_og = sample.loglikelihood_prime_decm(sol, prova.args)
        fj_casadi = casadi_fun_gradient(sol)

        err_ll_prime = max(abs(fj_og - fj_casadi.__array__()[0]))

        # print('loglikelihood prime og-casadi = {}'.format(err_ll_prime ))

        self.assertTrue(err_ll_prime < 1e-10)

    def test_loglikelihood_second(self):
        # problem initialisation
        A = mg.random_weighted_matrix_generator_dense(n=2, sym=False)
        prova = sample.DirectedGraph(A)
        prova.initial_guess = "random"
        prova._initialize_problem("decm", "quasinewton")
        sol = np.concatenate((prova.x, prova.y, prova.b_out, prova.b_in))

        # casadi functions initialization
        x, f = casadi_loglikelihood_decm(A)
        casadi_fun = Function("f", [x], [f])

        fj = jacobian(f, x)
        casadi_fun_gradient = Function("j", [x], [fj])

        fh = jacobian(fj, x)
        casadi_fun_hessian = Function("h", [x], [fh])
        fh_og = sample.loglikelihood_hessian_decm(sol, prova.args)
        fh_casadi = casadi_fun_hessian(sol)

        err_ll_second = np.max(np.abs(fh_og - fh_casadi.__array__()))
        # print(err_ll_second)

        self.assertTrue(err_ll_second < 1e-10)


if __name__ == "__main__":
    unittest.main()
