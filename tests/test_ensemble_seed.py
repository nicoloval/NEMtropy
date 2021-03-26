"""This test checks the default setting for the seed is working.
When the seed is not manually set, it should be ranodmly chosen.
"""

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
        N, seed = (10, 42)
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
        n = 10
        output_dir = "sample/"
        # random.seed(100)
        g_list = []
        for i in range(n):
            g.ensemble_sampler(n=1, output_dir=output_dir)
            g_list.append(np.loadtxt("sample/0.txt"))

        appo = True
        old = g_list[0]
        for i in range(1, n):
            appo = appo*np.all(old == g_list[i])

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
        self.assertTrue(not appo)


if __name__ == "__main__":
    unittest.main()
