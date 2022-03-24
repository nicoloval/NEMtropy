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

    def test_count_13(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        n = mf.motif13_count(A)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug

        # test result
        self.assertTrue(n == 6)

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
        zz = (n13 - n13_mu)/n13_std
        # debug
        """
        print(f'm13 observed = {n13}')
        print(f'm13 empirical mu = {n13_emp_mu}')
        print(f'm13 empirical std = {n13_emp_std}')
        print(f'm13 analytical mu = {n13_mu}')
        print(f'm13 analytical std = {n13_std}')
        print(f"analytical z score = {d['13']}")
        z = (n13 - n13_emp_mu)/n13_emp_std
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
