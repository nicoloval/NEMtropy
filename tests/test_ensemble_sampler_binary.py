import sys
import os
sys.path.append("../")
import Undirected_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


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
        N, seed = (20, 22)
        A = mg.random_binary_matrix_generator_dense(N, sym=True, seed=seed)
        # number of copies to generate

        g = sample.UndirectedGraph(A)

        g._solve_problem(
            model="cm",
            method="fixed-point",
            max_steps=100,
            verbose=False,
            linsearch=True,
            initial_guess="uniform",
        )

        g.solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n = 10000
        output_dir = "sample_cm/"
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=100)

        gdseq = g.dseq
        gdseq_sort = np.sort(gdseq) 

        # read all sampled graphs and check the average degree distribution is close enough
        gdseq_sort_av = np.zeros(N) 

        for g in range(n):
            f = output_dir + "{}.txt".format(g)
            if not os.stat(f).st_size == 0:
                edges_list = np.loadtxt(fname=f, dtype=int, delimiter=" ")
                if type(edges_list[0]) == np.int64:
                    # if True, there s only one link and it's not in the right format
                    edges_list = [(edges_list[0], edges_list[1])]
                else:
                    edges_list = [tuple(item) for item in edges_list]

                G = sample.UndirectedGraph()
                G._initialize_graph(edgelist=edges_list)
                tmp = np.zeros(N)
                tmp[-len(G.dseq):] = np.sort(G.dseq) 
                gdseq_sort_av = gdseq_sort_av + tmp

        gdseq_sort_av = gdseq_sort_av/n
        ensemble_error = np.linalg.norm(gdseq_sort - gdseq_sort_av, np.inf)

        # debug
        print('original dseq',gdseq_sort)
        print('original dseq sum ',gdseq_sort.sum())
        print('ensemble dseq',gdseq_sort_av)
        print('ensemble dseq sum ',gdseq_sort_av.sum())
        print('error', ensemble_error)
        print('solution error', err)


        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(err < 1e-1)


if __name__ == "__main__":
    unittest.main()
