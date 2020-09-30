import numpy as np
import unittest
import sys
from casadi import *
sys.path.append('../')
import Directed_graph_Class as sample


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_loglikelihood_second(self):
        # problem initialisation 
        A = sample.random_weighted_matrix_generator_dense(n=2, sym=False)
        prova = sample.DirectedGraph(A)
        prova.initial_guess='random'
        prova._initialize_problem('decm', 'quasinewton')
        x = np.concatenate((prova.x, prova.y, prova.b_out, prova.b_in))

        # casadi functions initialization
        fh = sample.loglikelihood_hessian_decm(x, prova.args)
        fh_diag = sample.loglikelihood_hessian_diag_decm(x, prova.args)

        err_ll_second = np.max(np.abs(np.diag(fh)- fh_diag))
        print(np.abs(np.diag(fh)-fh_diag))
        ##  print(err_ll_second)

        self.assertTrue(err_ll_second < 1e-10)


if __name__ == '__main__':
    unittest.main()

