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
        n, seed = (5, 42)
        network = mg.random_weighted_matrix_generator_dense(
            n=n, sup_ext=10, sym=True, seed=seed, intweights=True
        )
        # number of copies to generate

        g = sample.UndirectedGraph(adjacency=network)

        g.solve_tool(
            model="crema-sparse",
            method="quasinewton",
            initial_guess="random",
            adjacency="cm",
            max_steps=1000,
            verbose=False,
        )

        # g._solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n_sample = 50
        output_dir = "sample_crema_ecm_prob/"
        # random.seed(100)
        g.ensemble_sampler(n=n_sample, output_dir=output_dir, seed=42)


        d = {'{}'.format(i): g.dseq[i] for i in range(n)}
        s = {'{}'.format(i): g.strength_sequence[i] for i in range(n)}

        # read all sampled graphs and check the average degree distribution
        d_emp = {'{}'.format(i): 0 for i in range(n)}
        s_emp = {'{}'.format(i): 0 for i in range(n)}

        for l in range(n_sample):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, data=(("weight", float),))
                d_tmp = dict(g_tmp.degree)
                s_tmp = dict(g_tmp.degree(weight='weight'))
                for item in d_tmp.keys():
                    d_emp[item] += d_tmp[item]
                    s_emp[item] += s_tmp[item]

        for item in d_emp.keys():
            d_emp[item] = d_emp[item]/n_sample
            s_emp[item] = s_emp[item]/n_sample

        ad_diff = np.array([abs(d[item] - d_emp[item]) for item in d.keys()])
        as_diff = np.array([abs(s[item] - s_emp[item]) for item in s.keys()])
        a_diff = np.concatenate((ad_diff, as_diff))
        d_diff = {item: abs(d[item] - d_emp[item]) for item in d.keys()}
        s_diff = {item: abs(s[item] - s_emp[item]) for item in s.keys()}

        ensemble_error = np.linalg.norm(a_diff, np.inf)

        # debug
        # print('\n original degree sequence ', d)
        # print('\n original strength sequence ', s)
        # print('\n ensemble average strength sequence', s_emp)
        # print('\n degree by degree difference vector ', d_diff)
        # print('\n strength by strength difference vector ', s_diff)
        # print('\n empirical error = {}'.format(ensemble_error))
        # print('\n theoretical error = {}'.format(err))

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(ensemble_error < 3)

    def test_1(self):
        n, seed = (10, 42)
        """
        network = mg.random_weighted_matrix_generator_dense(
            n=n, sup_ext=10, sym=True, seed=seed, intweights=True
        )
        """
        network = mg.random_weighted_matrix_generator_uniform_custom_density(
            n=n,
            p=0.2,
            sym=True,
            sup_ext=10,
            intweights=True,
            seed=seed
        )

        # number of copies to generate

        g = sample.UndirectedGraph(adjacency=network)

        g.solve_tool(
            model="crema-sparse",
            method="quasinewton",
            initial_guess="random",
            adjacency="cm",
            max_steps=1000,
            verbose=False,
        )

        # g._solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n_sample = 500
        output_dir = "sample_crema_ecm_prob/"
        # random.seed(100)
        g.ensemble_sampler(n=n_sample, output_dir=output_dir, seed=42)


        d = {'{}'.format(i): g.dseq[i] for i in range(n)}
        s = {'{}'.format(i): g.strength_sequence[i] for i in range(n)}

        # read all sampled graphs and check the average degree distribution
        d_emp = {'{}'.format(i): 0 for i in range(n)}
        s_emp = {'{}'.format(i): 0 for i in range(n)}

        for l in range(n_sample):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, data=(("weight", float),))
                d_tmp = dict(g_tmp.degree)
                s_tmp = dict(g_tmp.degree(weight='weight'))
                for item in d_tmp.keys():
                    d_emp[item] += d_tmp[item]
                    s_emp[item] += s_tmp[item]

        for item in d_emp.keys():
            d_emp[item] = d_emp[item]/n_sample
            s_emp[item] = s_emp[item]/n_sample

        ad_diff = np.array([abs(d[item] - d_emp[item]) for item in d.keys()])
        as_diff = np.array([abs(s[item] - s_emp[item]) for item in s.keys()])
        a_diff = np.concatenate((ad_diff, as_diff))
        d_diff = {item: abs(d[item] - d_emp[item]) for item in d.keys()}
        s_diff = {item: abs(s[item] - s_emp[item]) for item in s.keys()}

        ensemble_error = np.linalg.norm(a_diff, np.inf)

        # debug
        # print('\n original degree sequence ', d)
        # print('\n original strength sequence ', s)
        # print('\n ensemble average strength sequence', s_emp)
        # print('\n degree by degree difference vector ', d_diff)
        # print('\n strength by strength difference vector ', s_diff)
        # print('\n empirical error = {}'.format(ensemble_error))
        # print('\n theoretical error = {}'.format(err))

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(ensemble_error < 13)



if __name__ == "__main__":
    unittest.main()
