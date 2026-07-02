import numpy as np
import networkx as nx
import pytest
from scipy import sparse

import NEMtropy.graph_classes as gc


@pytest.mark.math
class TestDirectedGraphInit:
    def test_adjacency_init(self):
        a = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
        g = gc.DirectedGraph()
        g._initialize_graph(a)
        assert (np.array([2, 2, 1]) == g.dseq_out).all()
        assert (np.array([1, 2, 2]) == g.dseq_in).all()

    def test_edgelist_init(self):
        e = np.array([(0, 1), (0, 2), (1, 2), (1, 0), (2, 1)])
        g = gc.DirectedGraph()
        g._initialize_graph(edgelist=e)
        assert (np.array([2, 2, 1]) == g.dseq_out).all()
        assert (np.array([1, 2, 2]) == g.dseq_in).all()

    def test_edgelist_string_directed(self):
        e = np.array([("1", "a"), ("2", "b"), ("2", "a")])
        g = gc.DirectedGraph()
        g._initialize_graph(edgelist=e)
        assert (np.array([1, 2, 0, 0]) == g.dseq_out).all()
        assert (np.array([0, 0, 2, 1]) == g.dseq_in).all()

    def test_edgelist_string_directed_weighted(self):
        e = np.array([("1", "a", 3), ("2", "b", 4), ("2", "a", 3)])
        g = gc.DirectedGraph()
        g._initialize_graph(edgelist=e)
        assert (np.array([1, 2, 0, 0]) == g.dseq_out).all()
        assert (np.array([0, 0, 2, 1]) == g.dseq_in).all()
        assert (np.array([3.0, 7.0, 0.0, 0.0]) == g.out_strength).all()
        assert (np.array([0.0, 0.0, 6.0, 4.0]) == g.in_strength).all()


class TestUndirectedGraphInit:
    def test_edgelist_string_undirected(self):
        e = np.array([("1", "a"), ("2", "b"), ("2", "a")])
        k = np.array([1, 2, 2, 1])
        g = gc.UndirectedGraph()
        g._initialize_graph(edgelist=e)
        assert (k == g.dseq).all()

    def test_edgelist_string_undirected_weighted(self):
        e = np.array([("1", "a", 3), ("2", "b", 4), ("2", "a", 3)])
        k = np.array([1, 2, 2, 1])
        s = np.array([3.0, 7.0, 6.0, 4.0])
        g = gc.UndirectedGraph()
        g._initialize_graph(edgelist=e)
        assert (k == g.dseq).all()
        assert (s == g.strength_sequence).all()
        assert g.is_weighted is True

    def test_weighted_numeric_edgelist_sets_is_weighted(self):
        g_nx = nx.karate_club_graph()
        a = nx.adjacency_matrix(g_nx)
        src, trg, weights = sparse.find(a)
        edgelist = np.vstack([src, trg, weights]).T
        graph = gc.UndirectedGraph(edgelist=edgelist)
        assert graph.is_weighted is True
        assert graph.strength_sequence is not None

    def test_unweighted_numeric_edgelist_not_weighted(self):
        g_nx = nx.karate_club_graph()
        a = nx.adjacency_matrix(g_nx)
        src, trg, _weights = sparse.find(a)
        edgelist = np.vstack([src, trg]).T
        graph = gc.UndirectedGraph(edgelist=edgelist)
        assert graph.is_weighted is False

    def test_weighted_adjacency_sets_is_weighted(self):
        g = gc.UndirectedGraph(adjacency=np.array([[0, 2], [2, 0]], dtype=float))
        assert g.is_weighted is True
