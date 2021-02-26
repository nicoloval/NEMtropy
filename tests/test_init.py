import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_adjacency_init(self):
        a = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
        k_out = np.array([2, 2, 1])
        k_in = np.array([1, 2, 2])

        g0 = sample.DirectedGraph()
        g0._initialize_graph(a)
        # debug
        # test result
        self.assertTrue((k_out == g0.dseq_out).all())
        self.assertTrue((k_in == g0.dseq_in).all())

    def test_edgelist_init(self):
        e = np.array([(0, 1), (0, 2), (1, 2), (1, 0), (2, 1)])
        k_out = np.array([2, 2, 1])
        k_in = np.array([1, 2, 2])

        g0 = sample.DirectedGraph()
        g0._initialize_graph(edgelist=e)
        # debug
        # test result

        self.assertTrue((k_out == g0.dseq_out).all())
        self.assertTrue((k_in == g0.dseq_in).all())

    def test_edgelist_init_string_undirected(self):
        e = np.array([("1", "a"), ("2", "b"), ("2", "a")])
        k = np.array([1, 2, 2, 1])

        g0 = sample.UndirectedGraph()
        g0._initialize_graph(edgelist=e)
        # debug
        # test result
        self.assertTrue(k.all() == g0.dseq.all())

    def test_edgelist_init_string_undirected_weighted(self):
        e = np.array([("1", "a", 3), ("2", "b", 4), ("2", "a", 3)])
        k = np.array([1, 2, 2, 1])
        s = np.array([3., 7., 6., 4.])

        g0 = sample.UndirectedGraph()
        g0._initialize_graph(edgelist=e)
        # debug
        # test result
        self.assertTrue((k == g0.dseq).all())
        self.assertTrue((s == g0.strength_sequence).all())

    def test_edgelist_init_string_directed(self):
        e = np.array([("1", "a"), ("2", "b"), ("2", "a")])
        k_out = np.array([1, 2, 0, 0])
        k_in = np.array([0, 0, 2, 1])

        g0 = sample.DirectedGraph()
        g0._initialize_graph(edgelist=e)
        # debug
        # test result
        self.assertTrue((k_out == g0.dseq_out).all())
        self.assertTrue((k_in == g0.dseq_in).all())

    def test_edgelist_init_string_directed_weighted(self):
        e = np.array([("1", "a", 3), ("2", "b", 4), ("2", "a", 3)])
        k_out = np.array([1, 2, 0, 0])
        k_in = np.array([0, 0, 2, 1])

        s_out = np.array([3., 7., 0., 0.])
        s_in = np.array([0., 0., 6., 4.])

        g0 = sample.DirectedGraph()
        g0._initialize_graph(edgelist=e)
        # debug
        # test result
        self.assertTrue((k_out == g0.dseq_out).all())
        self.assertTrue((k_in == g0.dseq_in).all())
        self.assertTrue((s_out == g0.out_strength).all())
        self.assertTrue((s_in == g0.in_strength).all())


if __name__ == "__main__":
    unittest.main()
