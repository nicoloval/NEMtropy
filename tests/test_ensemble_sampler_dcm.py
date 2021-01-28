import sys
import os
sys.path.append("../")
import NEMtropy.graph_classes as sample
import NEMtropy.matrix_generator as mg
import numpy as np
import unittest  # test tool
import random
import networkx as nx


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_0(self):
        N, seed = (5, 42)
        A = mg.random_binary_matrix_generator_dense(N, sym=False, seed=seed)
        # number of copies to generate

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="dcm",
            method="fixed-point",
            max_steps=100,
            verbose=False,
            linsearch=True,
            initial_guess="uniform",
        )

        # g._solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n = 100
        output_dir = "sample_dcm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=seed)

        d_out = {'{}'.format(i):g.dseq_out[i] for i in range(N)}
        d_in = {'{}'.format(i):g.dseq_in[i] for i in range(N)}


        # read all sampled graphs and check the average degree distribution is close enough
        d_out_emp = {'{}'.format(i):0 for i in range(N)}
        d_in_emp = {'{}'.format(i):0 for i in range(N)}

        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, create_using=nx.DiGraph())
                d_out_tmp = dict(g_tmp.out_degree)
                d_in_tmp = dict(g_tmp.in_degree)
                for item in d_out_tmp.keys(): 
                    d_out_emp[item] += d_out_tmp[item]
                    d_in_emp[item] += d_in_tmp[item]


        for item in d_out_emp.keys(): 
            d_out_emp[item] = d_out_emp[item]/n
            d_in_emp[item] = d_in_emp[item]/n

        a_out_diff = np.array([abs(d_out[item] - d_out_emp[item]) for item in d_out.keys()])
        a_in_diff = np.array([abs(d_in[item] - d_in_emp[item]) for item in d_in.keys()])
        d_out_diff = {item:d_out[item] - d_out_emp[item] for item in d_out.keys()}
        d_in_diff = {item:d_in[item] - d_in_emp[item] for item in d_in.keys()}

        ensemble_error = np.linalg.norm(np.concatenate((a_out_diff, a_in_diff)), np.inf)

        #debug
        """
        for i in range(N):
            for j in range(N):
                if i!=j:
                    aux = x[i]*x[j]
                    # print("({},{}) p = {}".format(i,j,aux/(1+aux)))
        """


        # debug
        # print('original dseq',d_out,d_in)
        # print('original dseq out sum ',g.dseq_out.sum())
        # print('original dseq in sum ',g.dseq_in.sum())
        # print('ensemble average dseq out', d_out_emp)
        # print('ensemble average dseq in', d_in_emp)
        # print('ensemble dseq out sum ',np.array([d_out_emp[key] for key in d_out_emp.keys()]).sum())
        # print('ensemble dseq in sum ',np.array([d_in_emp[key] for key in d_in_emp.keys()]).sum())
        # print(d_out_diff)
        # print(d_in_diff)
        # print('empirical error', ensemble_error)
        # print('theoretical error', err)


        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(ensemble_error < 3)


if __name__ == "__main__":
    unittest.main()
