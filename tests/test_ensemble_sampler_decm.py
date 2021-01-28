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
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        """
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        e = [(0,1), (0,2), (1,3)]
        d = [1,1,2,2]
        print(e)
        print(d)
        """
        N, seed = (50, 42)
        A = mg.random_weighted_matrix_generator_dense(
            n=N, sup_ext=10, sym=False, seed=seed, intweights=True
        )

        g = sample.DirectedGraph(A)

        g._solve_problem(
            model="decm_exp",
            method="newton",
            max_steps=100,
            verbose=False,
            linsearch=True,
            initial_guess="uniform",
        )

        x = g.x
        y = g.y
        b_out = g.b_out
        b_in = g.b_in

        # g._solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n = 100
        output_dir = "sample_decm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=42)

        #

        dk_out = {'{}'.format(i):g.dseq_out[i] for i in range(N)}
        dk_in = {'{}'.format(i):g.dseq_in[i] for i in range(N)}
        ds_out = {'{}'.format(i):g.out_strength[i] for i in range(N)}
        ds_in = {'{}'.format(i):g.in_strength[i] for i in range(N)}

        # read all sampled graphs and check the average degree distribution is close enough
        dk_out_emp = {'{}'.format(i):0 for i in range(N)}
        dk_in_emp = {'{}'.format(i):0 for i in range(N)}
        ds_out_emp = {'{}'.format(i):0 for i in range(N)}
        ds_in_emp = {'{}'.format(i):0 for i in range(N)}

        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, data=(("weight", float),), create_using=nx.DiGraph())
                dk_out_tmp = dict(g_tmp.out_degree)
                dk_in_tmp = dict(g_tmp.in_degree)
                ds_out_tmp = dict(g_tmp.out_degree(weight='weight'))
                ds_in_tmp = dict(g_tmp.in_degree(weight='weight'))
                for item in dk_out_tmp.keys(): 
                    dk_out_emp[item] += dk_out_tmp[item]
                    dk_in_emp[item] += dk_in_tmp[item]
                    ds_out_emp[item] += ds_out_tmp[item]
                    ds_in_emp[item] += ds_in_tmp[item]

        for item in dk_out_emp.keys(): 
            dk_out_emp[item] = dk_out_emp[item]/n
            dk_in_emp[item] = dk_in_emp[item]/n
            ds_out_emp[item] = ds_out_emp[item]/n
            ds_in_emp[item] = ds_in_emp[item]/n

        adk_out_diff = np.array([abs(dk_out[item] - dk_out_emp[item]) for item in dk_out.keys()])
        adk_in_diff = np.array([abs(dk_in[item] - dk_in_emp[item]) for item in dk_in.keys()])
        ads_out_diff = np.array([abs(ds_out[item] - ds_out_emp[item]) for item in ds_out.keys()])
        ads_in_diff = np.array([abs(ds_in[item] - ds_in_emp[item]) for item in ds_in.keys()])
        a_diff = np.concatenate((adk_out_diff, adk_in_diff, ads_out_diff, ads_in_diff)) 
        # d_diff = {item:d[item] - d_emp[item] for item in d.keys()}
        # s_diff = {item:s[item] - s_emp[item] for item in s.keys()}

        ensemble_error = np.linalg.norm(a_diff, np.inf)

        #debug
        """
        for i in range(N):
            for j in range(N):
                if i!=j:
                    aux = x[i]*x[j]
                    # print("({},{}) p = {}".format(i,j,aux/(1+aux)))
        """

        # debug
        """
        print('\n original degree sequence ', dk_out, dk_in)
        print('\n original strength sequence ',ds_out, ds_in)
        print('\n ensemble average degree sequence', dk_out_emp, dk_in_emp)
        print('\n ensemble average strength sequence', ds_out_emp, ds_in_emp)
        print('\n empirical error = {}'.format(ensemble_error))
        print('\n theoretical error = {}'.format(err))
        """

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(ensemble_error<10)


if __name__ == "__main__":
    unittest.main()
