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

        x = g.x
        # g._solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n = 100
        output_dir = "sample_cm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=42)

        d = {'{}'.format(i):g.dseq[i] for i in range(N)}


        # read all sampled graphs and check the average degree distribution is close enough
        d_emp = {'{}'.format(i):0 for i in range(N)}

        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f)
                d_tmp = dict(g_tmp.degree)
                for item in d_tmp.keys(): 
                    d_emp[item] += d_tmp[item]


        for item in d_emp.keys(): 
            d_emp[item] = d_emp[item]/n

        a_diff = np.array([abs(d[item] - d_emp[item]) for item in d.keys()])
        d_diff = {item:d[item] - d_emp[item] for item in d.keys()}

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
        print('original dseq',d)
        print('original dseq sum ',g.dseq.sum())
        print('ensemble average dseq', d_emp)
        print('ensemble dseq sum ',np.array([d_emp[key] for key in d_emp.keys()]).sum())
        print(d_diff)
        print('empirical error', ensemble_error)
        print('theoretical error', err)
        """


        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(ensemble_error < 3)


if __name__ == "__main__":
    unittest.main()
