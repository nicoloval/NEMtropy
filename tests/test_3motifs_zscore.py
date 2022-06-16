import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import numpy as np
import unittest  # test tool
import NEMtropy.network_functions as mf
import os
import networkx as nx
import NEMtropy.matrix_generator as mg
import NEMtropy.ensemble_functions as en


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skip("it works")
    def test_zscore_2(self):
        N, seed = (20, 100)
        A = mg.random_binary_matrix_generator_dense(N, sym=False, seed=seed)
        g = sample.DirectedGraph(A)

        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )

        d = g.motifs_3_zscore()
        sol = g.solution_array

        n = 100
        output_dir = "sample_dcm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=seed)
        n_emp = np.zeros(n)
        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, create_using=nx.DiGraph())
                a_tmp = nx.adjacency_matrix(g_tmp).toarray()
                n_emp[l] = mf.motif2_count(a_tmp)
				
        n = mf.motif2_count(A)
        n_emp_mu = np.mean(n_emp)
        n_emp_std = np.std(n_emp)
        n_mu = en.expected_motif2_dcm(sol)
        n_std = en.std_motif2_dcm(sol)
        # zz is d['13']
        zz = (n - n_mu)/n_std
        z = (n - n_emp_mu)/n_emp_std
        # debug
        print(f'm empirical mu = {n_emp_mu}')
        print(f'm empirical std = {n_emp_std}')
        print(f'm analytical mu = {n_mu}')
        print(f'm analytical std = {n_std}')
        print(f"analytical z score = {d['2']}")
        print(f'empirical z-score = {z}')
        print(f"diff = d['2'] - z = {abs(d['2'] - z})")

        # test result
        #TODO: write a better motif testing
        self.assertTrue(abs(d['2'] - z)< 1)

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

    @unittest.skip("it works")
    def test_zscore_13(self):
        N, seed = (20, 100)
        A = mg.random_binary_matrix_generator_dense(N, sym=False, seed=seed)
        g = sample.DirectedGraph(A)

        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )

        d = g.motifs_3_zscore()
        sol = g.solution_array

        n = 100
        output_dir = "sample_dcm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=seed)
        n13_emp = np.zeros(n)
        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, create_using=nx.DiGraph())
                a_tmp = nx.adjacency_matrix(g_tmp).toarray()
                n13_emp[l] = mf.motif13_count(a_tmp)
				
        n13 = mf.motif13_count(A)
        n13_emp_mu = np.mean(n13_emp)
        n13_emp_std = np.std(n13_emp)
        n13_mu = en.expected_motif13_dcm(sol)
        n13_std = en.std_motif13_dcm(sol)
        # zz is d['13']
        zz = (n13 - n13_mu)/n13_std
        z = (n13 - n13_emp_mu)/n13_emp_std
        # debug
        """
        print(f'm13 empirical mu = {n13_emp_mu}')
        print(f'm13 empirical std = {n13_emp_std}')
        print(f'm13 analytical mu = {n13_mu}')
        print(f'm13 analytical std = {n13_std}')
        print(f"analytical z score = {d['13']}")
        print(f'empirical z-score = {z}')
        print(f"diff = d['13'] - z")
        """

        # test result
        #TODO: write a better motif testing
        self.assertTrue(d['13'] - z < 1)

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
