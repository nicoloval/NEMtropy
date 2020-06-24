import numpy as np
import unittest
import sys
from casadi import *
sys.path.append('../')
import Directed_graph_Class as sample

def wrong_loglikelihood_dcm(args):
    """loglikelihood function for dcm as a casadi MX object
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = np.sum(c) 

    # casadi MX function calculation
    x = MX.sym('x', 2*n)
    f = 0
    for i in nz_index_out:
        f += c[i]*k_out[i]*np.log(x[i])
        for j in nz_index_in:
            if i != j:
                f -= c[i]*c[j]*np.log(1 + x[i]*x[n+j])
            else:
                f -= c[i]*(c[i] - 1)*np.log(1 + x[i]*x[n+j])

    for j in nz_index_in:
        f += c[j]*k_in[j]*np.log(x[j+n])

    print(f)
    return f


def casadi_loglikelihood_dcm(A):
    """loglikelihood function for dcm as a casadi MX object
    """
    # problem fixed parameters
    k_out = np.sum(A>0, 1)
    k_in = np.sum(A>0, 0)
    # casadi MX function calculation
    n = len(k_in)
    x = MX.sym('x', 2*n)
    f = 0
    for i in range(n):
        f += k_out[i]*np.log(x[i]) \
            + k_in[i]*np.log(x[i+n])
        for j in range(n):
            if i != j:
                f -= np.log(1 + x[i]*x[n+j])
    # fun = Function('f', [x], [f])
    return x, f


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_loglikelihood(self):
        # problem initialisation 
        A = sample.random_binary_matrix_generator_dense(n=10, sym=False)
        prova = sample.DirectedGraph(A)
        prova.degree_reduction()
        prova.initial_guess='random'
        prova._initialize_problem('dcm', 'quasinewton')
        sol = np.concatenate((prova.r_x,prova.r_y))
        prova._set_solved_problem_dcm(sol)
        sol_full = np.concatenate((prova.x, prova.y))

        # casadi functions initialization
        x, f = casadi_loglikelihood_dcm(A)
        casadi_fun = Function('f', [x], [f])
        f_og = sample.loglikelihood_dcm(sol, prova.args)
        f_casadi = casadi_fun(sol_full)

        err_loglikelihood = abs(f_og - f_casadi)
        # print('loglikelihood og-casadi = {}'.format(err_loglikelihood))
        self.assertTrue(err_loglikelihood < 1e-10)


    def test_loglikelihood_prime(self):
        # problem initialisation 
        A = sample.random_binary_matrix_generator_dense(n=10, sym=False)
        prova = sample.DirectedGraph(A)
        prova.degree_reduction()
        prova.initial_guess='random'
        prova._initialize_problem('dcm', 'quasinewton')
        sol = np.concatenate((prova.r_x,prova.r_y))
        prova._set_solved_problem_dcm(sol)
        sol_full = np.concatenate((prova.x, prova.y))

        # casadi functions initialization
        x, f = casadi_loglikelihood_dcm(A)
        casadi_fun = Function('f', [x], [f])

        fj = jacobian(f, x)
        casadi_fun_gradient = Function('j', [x], [fj])
        fj_og = sample.loglikelihood_prime_dcm(sol, prova.args)
        prova._set_solved_problem_dcm(fj_og)
        fj_og_full = np.concatenate((prova.x, prova.y))
        fj_casadi = casadi_fun_gradient(sol_full)

        err_ll_prime = max(abs(fj_og_full - fj_casadi.__array__()[0]))

        # print('loglikelihood prime og-casadi = {}'.format(err_ll_prime ))

        self.assertTrue(err_ll_prime < 1e-10)


    def test_loglikelihood_second(self):
        # problem initialisation 
        A = sample.random_binary_matrix_generator_dense(n=2, sym=False)
        prova = sample.DirectedGraph(A)
        prova.degree_reduction()
        prova.initial_guess='random'
        prova._initialize_problem('dcm', 'quasinewton')
        sol = np.concatenate((prova.r_x,prova.r_y))
        prova._set_solved_problem_dcm(sol)
        sol_full = np.concatenate((prova.x, prova.y))


        # casadi functions initialization
        x, f = casadi_loglikelihood_dcm(A)
        casadi_fun = Function('f', [x], [f])

        fj = jacobian(f, x)
        casadi_fun_gradient = Function('j', [x], [fj])

        fh = jacobian(fj, x)
        casadi_fun_hessian = Function('h', [x], [fh])
        fh_og = sample.loglikelihood_hessian_dcm(sol, prova.args)
        fh_casadi = casadi_fun_hessian(sol_full)

        print(sol_full)
        print('\t interludio \t')
        print(fh_og)
        print('\t interludio \t')
        print(fh_casadi.__array__())

        err_ll_second = np.max(np.abs(fh_og - fh_casadi.__array__()))
        ##  print(err_ll_second)

        self.assertTrue(err_ll_second < 1e-10)


if __name__ == '__main__':
    unittest.main()

