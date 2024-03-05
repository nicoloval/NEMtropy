import numpy as np
import scipy
import scipy.sparse
from numba import jit
from scipy.sparse import csr_matrix
from bicm.network_functions import *


def build_adjacency_from_edgelist(
        edgelist,
        is_directed,
        is_sparse=False,
        is_weighted=False):
    """Generates adjacency matrix given edgelist.

    :param edgelist: Edgelist.
    :type edgelist: numpy.ndarray
    :param is_directed: True if edge direction is informative.
    :type is_directed: bool
    :param is_sparse: If true the output is a sparse matrix.
    :type is_sparse: bool
    :param is_weighted: True if the edgelist is weighted
    :type is_weighted: bool
    :return: Adjacency matrix.
    :rtype: numpy.ndarray or scipy.sparse.csr_matrix
    """
    if not isinstance(edgelist, np.ndarray):
        raise TypeError("edgelist must be an numpy.ndarray.")
    if is_sparse:
        if is_weighted:
            adjacency = build_adjacency_sparse_weighted(edgelist, is_directed)
        else:
            adjacency = build_adjacency_sparse(edgelist, is_directed)
    else:
        if is_weighted:
            adjacency = build_adjacency_fast_weighted(edgelist, is_directed)
        else:
            adjacency = build_adjacency_fast(edgelist, is_directed)
    return adjacency


@jit(nopython=True)
def build_adjacency_fast(edgelist, is_directed):
    """Generates adjacency matrix given edgelist, numpy array format
    is used.

    :param edgelist: Edgelist.
    :type edgelist: numpy.ndarray
    :param is_directed: True if edge direction is informative.
    :type is_directed: bool
    :return: Adjacency matrix.
    :rtype: numpy.ndarray
    """
    if is_directed:
        n_nodes = len(set(edgelist[:, 0]) | set(edgelist[:, 1]))
        adj = np.zeros((n_nodes, n_nodes))
        for ii in np.arange(edgelist.shape[0]):
            edges = edgelist[ii]
            i = int(edges[0])
            j = int(edges[1])
            adj[i, j] = 1
    else:
        n_nodes = len(set(edgelist[:, 0]) | set(edgelist[:, 1]))
        adj = np.zeros((n_nodes, n_nodes))
        for ii in np.arange(edgelist.shape[0]):
            edges = edgelist[ii]
            i = int(edges[0])
            j = int(edges[1])
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


def build_adjacency_sparse(edgelist, is_directed):
    """Generates adjacency matrix given edgelist, scipy sparse format
    is used.

    :param edgelist: Edgelist.
    :type edgelist: numpy.ndarray
    :param is_sparse: True if edge direction is informative.
    :type is_sparse: bool
    :return: Adjacency matrix.
    :rtype: scipy.sparse.csr_matrix
    """
    n_nodes = set(edgelist[:, 0]) | set(edgelist[:, 1])
    if is_directed:
        row = edgelist[:, 0].astype(int)
        columns = edgelist[:, 1].astype(int)
        data = np.ones(n_nodes, dtype=int)
        adj = csr_matrix((data, (row, columns)), shape=(n_nodes, n_nodes))
    else:
        row = np.concatenate([edgelist[:, 0], edgelist[:, 1]]).astype(int)
        columns = np.concatenate([edgelist[:, 1], edgelist[:, 0]]).astype(int)
        data = np.ones(2 * n_nodes, dtype=int)
        adj = csr_matrix((data, (row, columns)), shape=(n_nodes, n_nodes))
    return adj


@jit(nopython=True)
def build_adjacency_fast_weighted(edgelist, is_directed):
    """Generates weighted adjacency matrix given edgelist,
    numpy array format is used.

    :param edgelist: Edgelist.
    :type edgelist: numpy.ndarray
    :param is_directed: True if edge direction is informative.
    :type is_directed: bool
    :return: Adjacency matrix.
    :rtype: numpy.ndarray
    """
    if is_directed:
        n_nodes = len(set(edgelist[:, 0]) | set(edgelist[:, 1]))
        adj = np.zeros((n_nodes, n_nodes))
        for ii in np.arange(edgelist.shape[0]):
            edges = edgelist[ii]
            i = int(edges[0])
            j = int(edges[1])
            w = edges[2]
            adj[i, j] = w
    else:
        n_nodes = len(set(edgelist[:, 0]) | set(edgelist[:, 1]))
        adj = np.zeros((n_nodes, n_nodes))
        for ii in np.arange(edgelist.shape[0]):
            edges = edgelist[ii]
            i = int(edges[0])
            j = int(edges[1])
            w = edges[2]
            adj[i, j] = w
            adj[j, i] = w
    return adj


def build_adjacency_sparse_weighted(edgelist, is_directed):
    """Generates weighted adjacency matrix given edgelist,
    scipy sparse format is used.

    :param edgelist: Edgelist.
    :type edgelist: numpy.ndarray
    :param is_sparse: True if edge direction is informative.
    :type is_sparse: bool
    :return: Adjacency matrix.
    :rtype: scipy.sparse.csr_matrix
    """
    n_nodes = set(edgelist[:, 0]) | set(edgelist[:, 1])
    if is_directed:
        row = edgelist[:, 0].astype(int)
        columns = edgelist[:, 1].astype(int)
        data = edgelist[:, 2]
        adj = csr_matrix((data, (row, columns)), shape=(n_nodes, n_nodes))
    else:
        row = np.concatenate([edgelist[:, 0], edgelist[:, 1]]).astype(int)
        columns = np.concatenate([edgelist[:, 1], edgelist[:, 0]]).astype(int)
        data = np.concatenate([edgelist[:, 2], edgelist[:, 2]])
        adj = csr_matrix((data, (row, columns)), shape=(n_nodes, n_nodes))
    return adj


def out_degree(a):
    """Returns matrix *a* out degrees sequence.

    :param a: Adjacency matrix
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: Out degree sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1  # noqa


def in_degree(a):
    """Returns matrix *a* in degrees sequence.

    :param a: Adjacency matrix.
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: In degree sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 0).A1


def out_strength(a):
    """Returns matrix *a* out strengths sequence.

    :param a: Adjacency matrix.
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: Out strengths sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 1).A1


def in_strength(a):
    """Returns matrix *a* in strengths sequence.

    :param a: Adjacency matrix.
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: In strengths sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 0).A1


def degree(a):
    """Returns matrix *a* degrees sequence.

    :param a: Adjacency matrix.
    :type a: numpy.ndarray, scipy.sparse
    :return: Degree sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1


def strength(a):
    """Returns matrix *a* strengths sequence.

    :param a: Adjacency matrix.
    :type a: numpy.ndarray, scipy.sparse
    :return: Strengths sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 1).A1


def edgelist_from_edgelist_undirected(edgelist):
    """Creates a new edgelist with the indexes of the nodes instead of the
     names. Returns also a dictionary that keep track of the nodes and,
      depending on the type of graph, degree and strengths sequences.

    :param edgelist: edgelist.
    :type edgelist: numpy.ndarray or list
    :return: edgelist, degrees sequence, strengths sequence and
     new labels to old labels dictionary.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, dict)
    """
    edgelist = [tuple(item) for item in edgelist]
    if len(edgelist[0]) == 2:
        # nodetype = type(edgelist[0][0])
        edgelist = np.array(
            edgelist,
            dtype=np.dtype([("source", object), ("target", object)]),
        )
    else:
        # nodetype = type(edgelist[0][0])
        # weigthtype = type(edgelist[0][2])
        # Vorrei mettere una condizione sul weighttype che deve essere numerico
        edgelist = np.array(
            edgelist,
            dtype=np.dtype(
                [
                    ("source", object),
                    ("target", object),
                    ("weigth", object),
                ]
            ),
        )
    # If there is a loop we count it twice in the degree of the node.
    unique_nodes, degree_seq = np.unique(
        np.concatenate((edgelist["source"], edgelist["target"])),
        return_counts=True,
    )
    nodes_dict = dict(enumerate(unique_nodes))
    inv_nodes_dict = {v: k for k, v in nodes_dict.items()}
    if len(edgelist[0]) == 2:
        edgelist_new = [
            (inv_nodes_dict[edge[0]], inv_nodes_dict[edge[1]])
            for edge in edgelist
        ]
        edgelist_new = np.array(
            edgelist_new, dtype=np.dtype([("source", int), ("target", int)])
        )
    else:
        edgelist_new = [
            (inv_nodes_dict[edge[0]], inv_nodes_dict[edge[1]], edge[2])
            for edge in edgelist
        ]
        edgelist_new = np.array(
            edgelist_new,
            dtype=np.dtype(
                [("source", int), ("target", int), ("weight", np.float64)]
            ),
        )
    if len(edgelist[0]) == 3:
        aux_edgelist = np.concatenate(
            (edgelist_new["source"], edgelist_new["target"])
        )
        aux_weights = np.concatenate(
            (edgelist_new["weight"], edgelist_new["weight"])
        )
        strength_seq = np.array(
            [aux_weights[aux_edgelist == i].sum() for i in nodes_dict]
        )
        return edgelist_new, degree_seq, strength_seq, nodes_dict
    return edgelist_new, degree_seq, nodes_dict


def edgelist_from_edgelist_directed(edgelist):
    """Creates a new edgelists replacing nodes labels with indexes.
    Returns also two dictionaries that keep track of the
    nodes index-label relation.
    Works also on weighted graphs.

    :param edgelist: List of edges.
    :type edgelist: list
    :return: Re-indexed list of edges, out-degrees, in-degrees,
        index to label dictionary
    :rtype: (dict, numpy.ndarray, numpy.ndarray, dict)
    """
    # TODO: inserire esempio edgelist pesata edgelist binaria
    # nel docstring
    # edgelist = list(zip(*edgelist))
    edgelist = [tuple(item) for item in edgelist]
    if len(edgelist[0]) == 2:
        # nodetype = type(edgelist[0][0])
        edgelist = np.array(
            edgelist,
            dtype=np.dtype(
                [
                    ("source", object),
                    ("target", object)
                ]
            ),
        )
    else:
        # nodetype = type(edgelist[0][0])
        # weigthtype = type(edgelist[0][2])
        # Vorrei mettere una condizione sul weighttype che deve essere numerico
        edgelist = np.array(
            edgelist,
            dtype=np.dtype(
                [
                    ("source", object),
                    ("target", object),
                    ("weigth", object),
                ]
            ),
        )
    # If there is a loop we count it twice in the degree of the node.
    unique_nodes = np.unique(
        np.concatenate((edgelist["source"], edgelist["target"])),
        return_counts=False,
    )
    out_degree = np.zeros_like(unique_nodes)
    in_degree = np.zeros_like(unique_nodes)
    nodes_dict = dict(enumerate(unique_nodes))
    inv_nodes_dict = {v: k for k, v in nodes_dict.items()}
    if len(edgelist[0]) == 2:
        edgelist_new = [
            (inv_nodes_dict[edge[0]], inv_nodes_dict[edge[1]])
            for edge in edgelist
        ]
        edgelist_new = np.array(
            edgelist_new, dtype=np.dtype([("source", int), ("target", int)])
        )
    else:
        edgelist_new = [
            (inv_nodes_dict[edge[0]], inv_nodes_dict[edge[1]], edge[2])
            for edge in edgelist
        ]
        edgelist_new = np.array(
            edgelist_new,
            dtype=np.dtype(
                [("source", int), ("target", int), ("weigth", np.float64)]
            ),
        )
    out_indices, out_counts = np.unique(
        edgelist_new["source"], return_counts=True
    )
    in_indices, in_counts = np.unique(
        edgelist_new["target"], return_counts=True
    )
    out_degree[out_indices] = out_counts
    in_degree[in_indices] = in_counts
    if len(edgelist[0]) == 3:
        out_strength = np.zeros_like(unique_nodes, dtype=np.float64)
        in_strength = np.zeros_like(unique_nodes, dtype=np.float64)
        out_counts_strength = np.array(
            [
                edgelist_new[edgelist_new["source"] == i]["weigth"].sum()
                for i in out_indices
            ]
        )
        in_counts_strength = np.array(
            [
                edgelist_new[edgelist_new["target"] == i]["weigth"].sum()
                for i in in_indices
            ]
        )
        out_strength[out_indices] = out_counts_strength
        in_strength[in_indices] = in_counts_strength
        return (
            edgelist_new,
            out_degree,
            in_degree,
            out_strength,
            in_strength,
            nodes_dict,
        )
    return edgelist_new, out_degree, in_degree, nodes_dict


def count_2motif_2(a):
    """Counts number of dyads.
    :param a np.ndarray: adjacency matrix
    :return: dyads count
    :rtype: int
    """

    tmp = a + a.transpose()
    return int(len(tmp[tmp == 2]))


def count_2motif_1(a):
    """Counts number of singles.
    :param a np.ndarray: adjacency matrix
    :return: singles count
    :rtype: int
    """

    tmp = a + a.transpose()
    return int(len(tmp[tmp == 1])/2)


def count_2motif_0(a):
    """Counts number of zeros.
    :param a np.ndarray: adjacency matrix
    :return: zeros count
    :rtype: int
    """

    n = a.shape[0]
    tmp = a + a.transpose()
    if isinstance(a, np.ndarray):
        return int((n*(n-1) - np.count_nonzero(tmp)))
    if isinstance(
        a,
        (
            scipy.sparse.csr_array,
            # scipy.sparse.coo_array,
            # scipy.sparse.bsr_array,
        )
    ):
        return int((n*(n-1) - tmp.count_nonzero()))

# 3-nodes motifs


@jit(nopython=True)
def count_3motif_1_ndarray(a):
    """Counts abundance of 3-nodes motif 1.
    :param a np.ndarray: adjacency matrix
    :return: motif 1 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s
 

@jit(nopython=True)
def count_3motif_2_ndarray(a):
    """Counts abundance of 3-nodes motif 2.
    :param a np.ndarray: adjacency matrix
    :return: motif 2 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*(1 - a[j, i])*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s
    

@jit(nopython=True)
def count_3motif_3_ndarray(a):
    """Counts abundance of 3-nodes motif 3.
    :param a np.ndarray: adjacency matrix
    :return: motif 3 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s
 

@jit(nopython=True)
def count_3motif_4_ndarray(a):
    """Counts abundance of 3-nodes motif 4.
    :param a np.ndarray: adjacency matrix
    :return: motif 4 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*(1 - a[j, i])*a[i, k]*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s


@jit(nopython=True)
def count_3motif_5_ndarray(a):
    """Counts abundance of 3-nodes motif 5.
    :param a np.ndarray: adjacency matrix
    :return: motif 5 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s


@jit(nopython=True)
def count_3motif_6_ndarray(a):
    """Counts abundance of 3-nodes motif 6.
    :param a np.ndarray: adjacency matrix
    :return: motif 6 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s


@jit(nopython=True)
def count_3motif_7_ndarray(a):
    """Counts abundance of 3-nodes motif 7.
    :param a np.ndarray: adjacency matrix
    :return: motif 7 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*(1 - a[j, k])*a[k, j]
    return s


@jit(nopython=True)
def count_3motif_8_ndarray(a):
    """Counts abundance of 3-nodes motif 8.
    :param a np.ndarray: adjacency matrix
    :return: motif 8 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*a[k, j]
    return s


@jit(nopython=True)
def count_3motif_9_ndarray(a):
    """Counts abundance of 3-nodes motif 9.
    :param a np.ndarray: adjacency matrix
    :return: motif 9 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*a[i, k]*(1 - a[k, i])*(1 - a[j, k])*a[k, j]
    return s


@jit(nopython=True)
def count_3motif_10_ndarray(a):
    """Counts abundance of 3-nodes motif 10.
    :param a np.ndarray: adjacency matrix
    :return: motif 10 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*a[k, j]
    return s


@jit(nopython=True)
def count_3motif_11_ndarray(a):
    """Counts abundance of 3-nodes motif 11.
    :param a np.ndarray: adjacency matrix
    :return: motif 11 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*(1 - a[j, i])*a[i, k]*(1 - a[k, i])*a[j, k]*a[k, j]
    return s


@jit(nopython=True)
def count_3motif_12_ndarray(a):
    """Counts abundance of 3-nodes motif_ 12.
    :param a np.ndarray: adjacency matrix
    :return: motif 12 count
    :rtype: int
    """

    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*a[k, j]
    return s


@jit(nopython=True)
def count_3motif_13_ndarray(a):
    """Counts abundance of 3-nodes motif_ 13.
    :param a np.ndarray: adjacency matrix
    :return: motif 13 count
    :rtype: int
    """
    n = a.shape[0]
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*a[i, k]*a[k, i]*a[j, k]*a[k, j]
    return s



def count_3motif_1_sparse(a):
    """Counts abundance of 3-nodes motif 1.
    :param a np.ndarray: adjacency matrix
    :return: motif 1 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s
 


def count_3motif_2_sparse(a):
    """Counts abundance of 3-nodes motif 2.
    :param a np.ndarray: adjacency matrix
    :return: motif 2 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*(1 - a[j, i])*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s
    


def count_3motif_3_sparse(a):
    """Counts abundance of 3-nodes motif 3.
    :param a np.ndarray: adjacency matrix
    :return: motif 3 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s
 


def count_3motif_4_sparse(a):
    """Counts abundance of 3-nodes motif 4.
    :param a np.ndarray: adjacency matrix
    :return: motif 4 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*(1 - a[j, i])*a[i, k]*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s



def count_3motif_5_sparse(a):
    """Counts abundance of 3-nodes motif 5.
    :param a np.ndarray: adjacency matrix
    :return: motif 5 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s



def count_3motif_6_sparse(a):
    """Counts abundance of 3-nodes motif 6.
    :param a np.ndarray: adjacency matrix
    :return: motif 6 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*(1 - a[k, j])
    return s



def count_3motif_7_sparse(a):
    """Counts abundance of 3-nodes motif 7.
    :param a np.ndarray: adjacency matrix
    :return: motif 7 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*(1 - a[j, k])*a[k, j]
    return s



def count_3motif_8_sparse(a):
    """Counts abundance of 3-nodes motif 8.
    :param a np.ndarray: adjacency matrix
    :return: motif 8 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*(1 - a[i, k])*(1 - a[k, i])*a[j, k]*a[k, j]
    return s



def count_3motif_9_sparse(a):
    """Counts abundance of 3-nodes motif 9.
    :param a np.ndarray: adjacency matrix
    :return: motif 9 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*a[i, k]*(1 - a[k, i])*(1 - a[j, k])*a[k, j]
    return s



def count_3motif_10_sparse(a):
    """Counts abundance of 3-nodes motif 10.
    :param a np.ndarray: adjacency matrix
    :return: motif 10 count
    :rtype: int
    """

    n = a.shape[0]
    
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += (1 - a[i, j])*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*a[k, j]
    return s



def count_3motif_11_sparse(a):
    """Counts abundance of 3-nodes motif 11.
    :param a np.ndarray: adjacency matrix
    :return: motif 11 count
    :rtype: int
    """

    n = a.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*(1 - a[j, i])*a[i, k]*(1 - a[k, i])*a[j, k]*a[k, j]
    return s



def count_3motif_12_sparse(a):
    """Counts abundance of 3-nodes motif_ 12.
    :param a np.ndarray: adjacency matrix
    :return: motif 12 count
    :rtype: int
    """

    n = a.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*a[i, k]*(1 - a[k, i])*a[j, k]*a[k, j]
    return s



def count_3motif_13_sparse(a):
    """Counts abundance of 3-nodes motif_ 13.
    :param a np.ndarray: adjacency matrix
    :return: motif 13 count
    :rtype: int
    """
    n = a.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                for k in range(n):
                    if k is not j and k is not i and i is not j:
                        s += a[i, j]*a[j, i]*a[i, k]*a[k, i]*a[j, k]*a[k, j]
    return s


def count_3motif_1(a):
    """Counts abundance of 3-nodes motif 1.
    :param a np.ndarray: adjacency matrix
    :return: motif 1 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_1_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_1_sparse(a)
    else:
        raise TypeError
 

def count_3motif_2(a):
    """Counts abundance of 3-nodes motif 2.
    :param a np.ndarray: adjacency matrix
    :return: motif 2 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_2_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_2_sparse(a)
    else:
        raise TypeError
    

def count_3motif_3(a):
    """Counts abundance of 3-nodes motif 3.
    :param a np.ndarray: adjacency matrix
    :return: motif 3 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_3_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_3_sparse(a)
    else:
        raise TypeError
 

def count_3motif_4(a):
    """Counts abundance of 3-nodes motif 4.
    :param a np.ndarray: adjacency matrix
    :return: motif 4 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_4_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_4_sparse(a)
    else:
        raise TypeError


def count_3motif_5(a):
    """Counts abundance of 3-nodes motif 5.
    :param a np.ndarray: adjacency matrix
    :return: motif 5 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_5_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_5_sparse(a)
    else:
        raise TypeError


def count_3motif_6(a):
    """Counts abundance of 3-nodes motif 6.
    :param a np.ndarray: adjacency matrix
    :return: motif 6 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_6_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_6_sparse(a)
    else:
        raise TypeError


def count_3motif_7(a):
    """Counts abundance of 3-nodes motif 7.
    :param a np.ndarray: adjacency matrix
    :return: motif 7 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_7_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_7_sparse(a)
    else:
        raise TypeError


def count_3motif_8(a):
    """Counts abundance of 3-nodes motif 8.
    :param a np.ndarray: adjacency matrix
    :return: motif 8 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_8_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_8_sparse(a)
    else:
        raise TypeError


def count_3motif_9(a):
    """Counts abundance of 3-nodes motif 9.
    :param a np.ndarray: adjacency matrix
    :return: motif 9 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_9_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_9_sparse(a)
    else:
        raise TypeError


def count_3motif_10(a):
    """Counts abundance of 3-nodes motif 10.
    :param a np.ndarray: adjacency matrix
    :return: motif 10 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_10_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_10_sparse(a)
    else:
        raise TypeError


def count_3motif_11(a):
    """Counts abundance of 3-nodes motif 11.
    :param a np.ndarray: adjacency matrix
    :return: motif 11 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_11_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_11_sparse(a)
    else:
        raise TypeError


def count_3motif_12(a):
    """Counts abundance of 3-nodes motif_ 12.
    :param a np.ndarray: adjacency matrix
    :return: motif 12 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_12_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_12_sparse(a)
    else:
        raise TypeError


def count_3motif_13(a):
    """Counts abundance of 3-nodes motif_ 13.
    :param a np.ndarray: adjacency matrix
    :return: motif 13 count
    :rtype: int
    """
    if isinstance(a, np.ndarray):
        return count_3motif_13_ndarray(a)
    elif isinstance(a, scipy.sparse.spmatrix):
        return count_3motif_13_sparse(a)
    else:
        raise TypeError
