import numpy as np
import scipy
import scipy.sparse
from numba import jit
from scipy.sparse import csr_matrix


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


# Bipartite networks functions


@jit(nopython=True)
def bicm_from_fitnesses(x, y):
    """
    Rebuilds the average probability matrix of the bicm2 from the fitnesses

    :param x: the fitness vector of the rows layer
    :type x: numpy.ndarray
    :param y: the fitness vector of the columns layer
    :type y: numpy.ndarray
    """
    avg_mat = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            xy = x[i] * y[j]
            avg_mat[i, j] = xy / (1 + xy)
    return avg_mat


def sample_bicm(avg_mat):
    """
    Build a biadjacency matrix sampling from the probability matrix of a BiCM.
    """
    if not isinstance(avg_mat, np.ndarray):
        avg_mat = np.array(avg_mat)
    dim1, dim2 = avg_mat.shape
    return np.array(
        avg_mat > np.reshape(np.random.sample(dim1 * dim2), (dim1, dim2)),
        dtype=int)


def sample_bicm_edgelist(x, y):
    """
    Build an edgelist sampling from the fitnesses of a BiCM.
    """
    edgelist = []
    for i in range(len(x)):
        for j in range(len(y)):
            xy = x[i] * y[j]
            if np.random.uniform() < xy / (1 + xy):
                edgelist.append((i, j))
    return edgelist


def sample_bicm_edgelist_names(dict_x, dict_y):
    """
    Build an edgelist from the BiCM keeping the names of the nodes as contained in the BiCM fitnesses' dictionaries.
    """
    edgelist = []
    for xx in dict_x:
        for yy in dict_y:
            xy = dict_x[xx] * dict_y[yy]
            if np.random.uniform() < xy / (1 + xy):
                edgelist.append((xx, yy))
    return edgelist


@jit(nopython=True)
def edgelist_from_biadjacency_fast(biadjacency):
    """
    Build the edgelist of a bipartite network from its biadjacency matrix.
    """
    edgelist = []
    for i in range(biadjacency.shape[0]):
        for j in range(biadjacency.shape[1]):
            if biadjacency[i, j] == 1:
                edgelist.append((i, j))
    return edgelist


def edgelist_from_biadjacency(biadjacency):
    """
    Build the edgelist of a bipartite network from its biadjacency matrix.
    Accounts for sparse matrices and returns a structured array.
    """
    if scipy.sparse.isspmatrix(biadjacency):
        coords = biadjacency.nonzero()
        if np.sum(biadjacency.data != 1) > 0:
            raise ValueError('Only binary matrices')
        return np.array(list(zip(coords[0], coords[1])),
                        dtype=np.dtype([('rows', int), ('columns', int)])), \
               np.array(biadjacency.sum(1)).flatten(), np.array(
            biadjacency.sum(0)).flatten()
    else:
        if np.sum(biadjacency[biadjacency != 0] != 1) > 0:
            raise ValueError('Only binary matrices')
        return np.array(edgelist_from_biadjacency_fast(biadjacency),
                        dtype=np.dtype([('rows', int), ('columns', int)])), \
               np.sum(biadjacency, axis=1), np.sum(biadjacency, axis=0)


def biadjacency_from_edgelist(edgelist, fmt='array'):
    """
    Build the biadjacency matrix of a bipartite network from its edgelist.
    Returns a matrix of the type specified by ``fmt``, by default a numpy array.
    """
    edgelist, rows_deg, cols_deg, rows_dict, cols_dict = edgelist_from_edgelist_bipartite(
        edgelist)
    if fmt == 'array':
        biadjacency = np.zeros((len(rows_deg), len(cols_deg)), dtype=int)
        for edge in edgelist:
            biadjacency[edge[0], edge[1]] = 1
    elif fmt == 'sparse':
        biadjacency = scipy.sparse.coo_matrix(
            (np.ones(len(edgelist)), (edgelist['rows'], edgelist['columns'])))
    elif not isinstance(format, str):
        raise TypeError('format must be a string (either "array" or "sparse")')
    else:
        raise ValueError('format must be either "array" or "sparse"')
    return biadjacency, rows_deg, cols_deg, rows_dict, cols_dict


def edgelist_from_edgelist_bipartite(edgelist):
    """
    Creates a new edgelist with the indexes of the nodes instead of the names.
    Method for bipartite networks.
    Returns also two dictionaries that keep track of the nodes.
    """
    edgelist = np.array(list(set([tuple(edge) for edge in edgelist])))
    out = np.zeros(np.shape(edgelist)[0],
                   dtype=np.dtype([('source', object), ('target', object)]))
    out['source'] = edgelist[:, 0]
    out['target'] = edgelist[:, 1]
    edgelist = out
    unique_rows, rows_degs = np.unique(edgelist['source'], return_counts=True)
    unique_cols, cols_degs = np.unique(edgelist['target'], return_counts=True)
    rows_dict = dict(enumerate(unique_rows))
    cols_dict = dict(enumerate(unique_cols))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    edgelist_new = [(inv_rows_dict[edge[0]], inv_cols_dict[edge[1]]) for edge
                    in edgelist]
    edgelist_new = np.array(edgelist_new,
                            dtype=np.dtype([('rows', int), ('columns', int)]))
    return edgelist_new, rows_degs, cols_degs, rows_dict, cols_dict


def adjacency_list_from_edgelist_bipartite(edgelist, convert_type=True):
    """
    Creates the adjacency list from the edgelist.
    Method for bipartite networks.
    Returns two dictionaries, each containing an adjacency list with the rows as keys and the columns as keys, respectively.
    If convert_type is True (default), then the nodes are enumerated and the adjacency list is returned as integers.
    Returns also two dictionaries that keep track of the nodes and the two degree sequences.
    """
    if convert_type:
        edgelist, rows_degs, cols_degs, rows_dict, cols_dict = edgelist_from_edgelist_bipartite(
            edgelist)
    adj_list = {}
    inv_adj_list = {}
    for edge in edgelist:
        adj_list.setdefault(edge[0], set()).add(edge[1])
        inv_adj_list.setdefault(edge[1], set()).add(edge[0])
    if not convert_type:
        rows_degs = np.array([len(adj_list[k]) for k in adj_list])
        rows_dict = {k: k for k in adj_list}
        cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
        cols_dict = {k: k for k in inv_adj_list}
    return adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict


def adjacency_list_from_adjacency_list_bipartite(old_adj_list):
    """
    Creates the adjacency list from another adjacency list, convering the data type.
    Method for bipartite networks.
    Returns two dictionaries, each containing an adjacency list with the rows as keys and the columns as keys, respectively.
    Original keys are treated as rows, values as columns.
    The nodes are enumerated and the adjacency list is returned as integers.
    Returns also two dictionaries that keep track of the nodes and the two degree sequences.
    """
    rows_dict = dict(enumerate(np.unique(list(old_adj_list.keys()))))
    cols_dict = dict(enumerate(
        np.unique([el for mylist in old_adj_list.values() for el in mylist])))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    adj_list = {}
    inv_adj_list = {}
    for k in old_adj_list:
        adj_list.setdefault(inv_rows_dict[k], set()).update(
            {inv_cols_dict[val] for val in old_adj_list[k]})
        for val in old_adj_list[k]:
            inv_adj_list.setdefault(inv_cols_dict[val], set()).add(
                inv_rows_dict[k])
    rows_degs = np.array([len(adj_list[k]) for k in adj_list])
    cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
    return adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict


def adjacency_list_from_biadjacency(biadjacency):
    """
    Creates the adjacency list from a biadjacency matrix, given in sparse format or as a list or numpy array.
    Returns two dictionaries, each containing an adjacency list with the rows as keys and the columns as keys, respectively.
    Returns also the two degree sequences.
    """
    if scipy.sparse.isspmatrix(biadjacency):
        if np.sum(biadjacency.data != 1) > 0:
            raise ValueError('Only binary matrices')
        coords = biadjacency.nonzero()
    else:
        biadjacency = np.array(biadjacency)
        if np.sum(biadjacency[biadjacency != 0] != 1) > 0:
            raise ValueError('Only binary matrices')
        coords = np.where(biadjacency != 0)
    adj_list = {}
    inv_adj_list = {}
    for edge_i in range(len(coords[0])):
        adj_list.setdefault(coords[0][edge_i], set()).add(coords[1][edge_i])
        inv_adj_list.setdefault(coords[1][edge_i], set()).add(
            coords[0][edge_i])
    rows_degs = np.array([len(adj_list[k]) for k in adj_list])
    cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
    return adj_list, inv_adj_list, rows_degs, cols_degs
