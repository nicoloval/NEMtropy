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

    @unittest.skip("tmp")
    def test_0(self):
        N, seed = (10, 42)
        network = mg.random_weighted_matrix_generator_dense(
            n=N, sup_ext=10, sym=False, seed=seed, intweights=False
        )

        g = sample.DirectedGraph(network)
        # network_bin = (network > 0).astype(int)

        g.solve_tool(
            model="crema",
            method="quasinewton",
            initial_guess="random",
            adjacency="dcm",
            max_steps=1000,
            verbose=False,
        )

        # g._solution_error()

        # print('\ntest 5: error = {}'.format(g.error))
        n = 1000
        output_dir = "sample_crema_decm_prob/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=42)

        #

        dk_out = {'{}'.format(i): g.dseq_out[i] for i in range(N)}
        dk_in = {'{}'.format(i): g.dseq_in[i] for i in range(N)}
        ds_out = {'{}'.format(i): g.out_strength[i] for i in range(N)}
        ds_in = {'{}'.format(i): g.in_strength[i] for i in range(N)}

        # read all sampled graphs and check the average degree distribution is close enough
        dk_out_emp = {'{}'.format(i): 0 for i in range(N)}
        dk_in_emp = {'{}'.format(i): 0 for i in range(N)}
        ds_out_emp = {'{}'.format(i): 0 for i in range(N)}
        ds_in_emp = {'{}'.format(i): 0 for i in range(N)}

        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, data=(
                    ("weight", float),), create_using=nx.DiGraph())
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

        adk_out_diff = np.array(
            [abs(dk_out[item] - dk_out_emp[item]) for item in dk_out.keys()])
        adk_in_diff = np.array(
            [abs(dk_in[item] - dk_in_emp[item]) for item in dk_in.keys()])
        ads_out_diff = np.array(
            [abs(ds_out[item] - ds_out_emp[item]) for item in ds_out.keys()])
        ads_in_diff = np.array(
            [abs(ds_in[item] - ds_in_emp[item]) for item in ds_in.keys()])
        a_diff = np.concatenate(
            (adk_out_diff, adk_in_diff, ads_out_diff, ads_in_diff))
        # d_diff = {item:d[item] - d_emp[item] for item in d.keys()}
        # s_diff = {item:s[item] - s_emp[item] for item in s.keys()}

        ensemble_error = np.linalg.norm(a_diff, np.inf)

        # debug
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
        print('\n original degree sequence ', dk_out, dk_in)
        """

        l = os.listdir(output_dir)

        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)


        # test result
        self.assertTrue(ensemble_error < 4)

    def test_1(self):
        N, seed = (5, 42)
        """
        network = mg.random_weighted_matrix_generator_dense(
            n=N, sup_ext=10, sym=False, seed=seed, intweights=False
        )
        """
        network = mg.random_weighted_matrix_generator_uniform_custom_density(
            n=N,
            p=0.2,
            sym=False,
            sup_ext=30,
            intweights=True,
            seed=seed
        )


        g = sample.DirectedGraph(network)
        # network_bin = (network > 0).astype(int)

        g.solve_tool(
            model="crema",
            method="quasinewton",
            initial_guess="random",
            adjacency="dcm",
            max_steps=1000,
            verbose=False,
        )

        # g._solution_error()

        # print('\ntest 5: error = {}'.format(g.error))
        n = 1000
        output_dir = "sample_crema_decm_prob/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=42)

        #

        dk_out = {'{}'.format(i): g.dseq_out[i] for i in range(N)}
        dk_in = {'{}'.format(i): g.dseq_in[i] for i in range(N)}
        ds_out = {'{}'.format(i): g.out_strength[i] for i in range(N)}
        ds_in = {'{}'.format(i): g.in_strength[i] for i in range(N)}

        # read all sampled graphs and check the average degree distribution is close enough
        dk_out_emp = {'{}'.format(i): 0 for i in range(N)}
        dk_in_emp = {'{}'.format(i): 0 for i in range(N)}
        ds_out_emp = {'{}'.format(i): 0 for i in range(N)}
        ds_in_emp = {'{}'.format(i): 0 for i in range(N)}

        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, data=(
                    ("weight", float),), create_using=nx.DiGraph())
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

        adk_out_diff = np.array(
            [abs(dk_out[item] - dk_out_emp[item]) for item in dk_out.keys()])
        adk_in_diff = np.array(
            [abs(dk_in[item] - dk_in_emp[item]) for item in dk_in.keys()])
        ads_out_diff = np.array(
            [abs(ds_out[item] - ds_out_emp[item]) for item in ds_out.keys()])
        ads_in_diff = np.array(
            [abs(ds_in[item] - ds_in_emp[item]) for item in ds_in.keys()])
        a_diff = np.concatenate(
            (adk_out_diff, adk_in_diff, ads_out_diff, ads_in_diff))
        # d_diff = {item:d[item] - d_emp[item] for item in d.keys()}
        # s_diff = {item:s[item] - s_emp[item] for item in s.keys()}

        ensemble_error = np.linalg.norm(a_diff, np.inf)

        # debug
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
        print('\n original degree sequence ', dk_out, dk_in)
        """

        l = os.listdir(output_dir)

        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)


        # test result
        self.assertTrue(ensemble_error < 4)



if __name__ == "__main__":
    unittest.main()
