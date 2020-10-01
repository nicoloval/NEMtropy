import sys

sys.path.append("../")
import Directed_graph_Class as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_adjacency_init(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
        k_out = np.array([2, 2, 1])
        k_in = np.array([1, 2, 2])

        g0 = sample.DirectedGraph()
        g0._initialize_graph(A)
        # debug
        # test result
        self.assertTrue(k_out.all() == g0.dseq_out.all())
        self.assertTrue(k_in.all() == g0.dseq_in.all())

    def test_edgelist_init(self):
        E = np.array([(0, 1), (0, 2), (1, 2), (1, 0), (2, 1)])
        k_out = np.array([2, 2, 1])
        k_in = np.array([1, 2, 2])

        g0 = sample.DirectedGraph()
        g0._initialize_graph(E)
        # debug
        # test result
        self.assertTrue(k_out.all() == g0.dseq_out.all())
        self.assertTrue(k_in.all() == g0.dseq_in.all())


if __name__ == "__main__":
    unittest.main()
