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
        A  = mg.random_weighted_matrix_generator_dense(
            n=N, sup_ext=10, sym=True, seed=seed, intweights=True
        )
        # number of copies to generate

        g = sample.UndirectedGraph(A)

        g._solve_problem(
            model="ecm",
            method="newton",
            max_steps=100,
            verbose=False,
            linsearch=True,
            initial_guess="uniform",
        )

        x = g.x
        # g._solution_error()
        err = g.error

        # print('\ntest 5: error = {}'.format(g.error))
        n = 1000
        output_dir = "sample_ecm/"
        # random.seed(100)
        g.ensemble_sampler(n=n, output_dir=output_dir, seed=42)

        d = {'{}'.format(i):g.dseq[i] for i in range(N)}
        s = {'{}'.format(i):g.strength_sequence[i] for i in range(N)}


        # read all sampled graphs and check the average degree distribution is close enough
        d_emp = {'{}'.format(i):0 for i in range(N)}
        s_emp = {'{}'.format(i):0 for i in range(N)}

        for l in range(n):
            f = output_dir + "{}.txt".format(l)
            if not os.stat(f).st_size == 0:
                g_tmp = nx.read_edgelist(f, data=(("weight", float),))
                d_tmp = dict(g_tmp.degree)
                s_tmp = dict(g_tmp.degree(weight='weight'))
                for item in d_tmp.keys(): 
                    d_emp[item] += d_tmp[item]
                    s_emp[item] += s_tmp[item]

        for item in d_emp.keys(): 
            d_emp[item] = d_emp[item]/n
            s_emp[item] = s_emp[item]/n

        ad_diff = np.array([abs(d[item] - d_emp[item]) for item in d.keys()])
        as_diff = np.array([abs(s[item] - s_emp[item]) for item in s.keys()])
        a_diff = np.concatenate((ad_diff, as_diff)) 
        d_diff = {item:d[item] - d_emp[item] for item in d.keys()}
        s_diff = {item:s[item] - s_emp[item] for item in s.keys()}

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
        print('\n original degree sequence ', d)
        print('\n original strength sequence ', s)
        print('\n ensemble average strength sequence', s_emp)
        print('\n degree by degree difference vector ', d_diff)
        print('\n strength by strength difference vector ', s_diff)
        print('\n empirical error = {}'.format(ensemble_error))
        print('\n theoretical error = {}'.format(err))


        l = os.listdir(output_dir)
        for f in l:
            os.remove(output_dir + f)
        os.rmdir(output_dir)

        # test result
        self.assertTrue(ensemble_error<3)


if __name__ == "__main__":
    unittest.main()
