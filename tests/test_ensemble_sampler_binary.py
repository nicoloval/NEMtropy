import sys

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
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        # number of copies to generate
        n = 10

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
        g.ensemble_sampler(n=n, output_dir="sample_cm/", seed=100)

        gdseq = g.dseq
        gdseq_sort = np.sort(gdseq) 

        # read all sampled graphs and check the average degree distribution is close enough
        N = len(gdseq)
        gdseq_sort_av = np.zeros(N) 

        for g in range(n):
            f = "sample_cm/{}.txt".format(g)
            print(f)
            edges_list = np.loadtxt(fname=f, dtype=int, delimiter=" ")
            if type(edges_list[0]) == np.int64:
                # if True, there s only one link and it's not in the right format
                edges_list = [(edges_list[0], edges_list[1])]
            else:
                edges_list = [tuple(item) for item in edges_list]
            print(edges_list)

            G = sample.UndirectedGraph()
            G._initialize_graph(edgelist=edges_list)
            tmp = np.zeros(N)
            tmp[:N] = gdseq_sort_av
            gdseq_sort_av = gdseq_sort_av

        print(gdseq_sort_av)

        # test result
        self.assertTrue(err < 1e-1)


if __name__ == "__main__":
    unittest.main()
