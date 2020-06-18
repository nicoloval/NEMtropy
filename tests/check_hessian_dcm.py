import numpy as np
import sys
sys.path.append('../')

import Directed_graph_Class

blax = Directed_graph_Class.random_binary_matrix_generator_dense(n=10, sym=False)
prova = Directed_graph_Class.DirectedGraph(blax)
prova.degree_reduction()
prova.initial_guess='uniform'
prova._initialize_problem('dcm', 'quasinewton')

solutions = np.concatenate((prova.r_x,prova.r_y))
hessian = Directed_graph_Class.loglikelihood_hessian_dcm(solutions,prova.args)
diag_hessian = Directed_graph_Class.loglikelihood_hessian_diag_dcm(solutions,prova.args)
y = np.diag(hessian) - diag_hessian
for i in range(len(y)):
    if abs(y[i]) > 1e-3:
        print(i)
        print(y[i])

