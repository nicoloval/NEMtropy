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

        # print('\ntest 5: error = {}'.format(g.error))
        g.ensemble_sampler(n=n, output_dir="sample_cm/", seed=100)

        gdseq = g.dseq
        gdseq_sort = np.sort(gdseq) 

        # read all sampled graphs and check the average degree distribution is close enough
        N = len(gdseq)
        gdseq_av = np.zeros(N) 
        gdseq_sort_av = np.zeros(N) 
        for g in range(n):
            f = "sample_cm/{}.txt".format(g)
            edges_list = np.loadtxt(fname=f, dtype=int, delimiter=" ")
            print(edges_list)
            edgelist = list(zip(*edges_list))
            print(edgelist)
            sys.exit()

            G = sample.UndirectedGraph()
            G._initialize_graph(edgelist=edges_list)
            print(G.dseq)
            sys.exit()




        # test result
        self.assertTrue(g.error < 1e-1)


if __name__ == "__main__":
    unittest.main()
