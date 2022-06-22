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

    @unittest.skip("works, need external data")
    def test_zscore_all_wtw(self):
        """This test use the wtw 2000 network, which is not usually loaded in the git repository and is used only for internal testing by the devs.
        """
        N, seed = (20, 100)
        # A = mg.random_binary_matrix_generator_dense(N, sym=False, seed=seed)
        A = np.loadtxt("A.txt")
        g = sample.DirectedGraph(A)

        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )

        d = g.zscore_3motifs()
        sol = np.concatenate((g.x, g.y))

        n = 100
        output_dir = "sample_dcm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=seed)
        n_sam = {str(i): np.zeros(n) for i in range(1,14)}
        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, create_using=nx.DiGraph())
                a_tmp = nx.adjacency_matrix(g_tmp).toarray()
                for i in range(1, 14):
                    n_sam[str(i)][l] = eval(f'mf.count_3motif_{i}(a_tmp)')
				
        """
        z = {}
        for i in range(1, 14):
            n_emp = n_sam[str(i)]
            n = eval(f'mf.count_3motif_{i}(A)')
            n_emp_mu = np.mean(n_emp)
            n_emp_std = np.std(n_emp)
            n_mu = eval(f"en.expected_dcm_3motif_{i}(sol)")
            n_std = eval(f"en.std_dcm_3motif_{i}(sol)")
            # zz is d['13']
            zz = (n - n_mu)/n_std
            z[str(i)] = (n - n_emp_mu)/n_emp_std
            # debug
            print(f"motif {i}")
            print(f'n = {n}')
            print(f'm empirical mu = {n_emp_mu}')
            print(f'm analytical mu = {n_mu}')
            print(f'm empirical std = {n_emp_std}')
            print(f'm analytical std = {n_std}')
            print(f'empirical z-score = {z[str(i)]}')
            print(f"analytical z score = {d[str(i)]}")
            print(f"diff = d[{i}] - z = {abs(d[str(i)] - z[str(i)])}")
        """

        goal = {
            "1": -33.8,
            "2": -62.9,
            "3": -30.8,
            "4": -32.9,
            "5": -35.9,
            "6": -30.8,
            "7": -25.6,
            "8": 79.3,
            "9": -36,
            "10": -53.6,
            "11": -29.9,
            "12": -116.3,
            "13": 43.5
        }
        # debug
        # for i in range(1, 14):
        #     print(f"diff = d[{i}] - z = {abs(d[str(i)] - goal[str(i)])}")
        # test result
        for i in range(1, 14):
            print(i)
            self.assertTrue(abs(d[str(i)] - goal[str(i)])< 0.1)

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

    # @unittest.skip("debug")
    def test_zscore_all(self):
        N, seed = (10, 100)
        A = mg.random_binary_matrix_generator_dense(N, sym=False, seed=seed)
        g = sample.DirectedGraph(A)

        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )

        d = g.zscore_3motifs()
        sol = np.concatenate((g.x, g.y))

        n = 100
        output_dir = "sample_dcm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=seed)
        n_sam = {str(i): np.zeros(n) for i in range(1,14)}
        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, create_using=nx.DiGraph())
                a_tmp = nx.adjacency_matrix(g_tmp).toarray()
                for i in range(1, 14):
                    n_sam[str(i)][l] = eval(f'mf.count_3motif_{i}(a_tmp)')
				
        z = {}
        for i in range(1, 14):
            n_emp = n_sam[str(i)]
            n = eval(f'mf.count_3motif_{i}(A)')
            n_emp_mu = np.mean(n_emp)
            n_emp_std = np.std(n_emp)
            n_mu = eval(f"en.expected_dcm_3motif_{i}(sol)")
            n_std = eval(f"en.std_dcm_3motif_{i}(sol)")
            # zz is d['13']
            zz = (n - n_mu)/n_std
            z[str(i)] = (n - n_emp_mu)/n_emp_std
            """
            # debug
            print(f"motif {i}")
            # print(f'n = {n}')
            # print(f'm empirical mu = {n_emp_mu}')
            # print(f'm analytical mu = {n_mu}')
            # print(f'm empirical std = {n_emp_std}')
            # print(f'm analytical std = {n_std}')
            # print(f'empirical z-score = {z[str(i)]}')
            # print(f"analytical z score = {d[str(i)]}")
            print(f"diff = d[{i}] - z = {abs(d[str(i)] - z[str(i)])}")
            """

        # test result
        #TODO: 5 is a bit too much of an interval
        for i in range(1, 14):
            self.assertTrue(abs(d[str(i)] - z[str(i)])< 5)

        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
