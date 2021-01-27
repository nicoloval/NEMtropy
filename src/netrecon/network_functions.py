import numpy as np
import scipy.sparse
import scipy
# Stops Numba Warning for experimental feature
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings

warnings.simplefilter(
    action='ignore',
    category=NumbaExperimentalFeatureWarning)


def out_degree(a):
    """Returns matrix *a* out degrees sequence.

    :param a: Adjacency matrix
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: Out degree sequence
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
    """Creates a new edgelist with the indexes of the nodes instead of the names. Returns also a dictionary that keep track of the nodes and, depending on the type of graph, degree and strengths sequences.

    :param edgelist: edgelist.
    :type edgelist: numpy.ndarray or list
    :return: edgelist, degrees sequence, strengths sequence and new labels to old labels dictionary.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, dict)
    """
    edgelist = [tuple(item) for item in edgelist]
    if len(edgelist[0]) == 2:
        nodetype = type(edgelist[0][0])
        edgelist = np.array(
            edgelist,
            dtype=np.dtype([("source", nodetype), ("target", nodetype)]),
        )
    else:
        nodetype = type(edgelist[0][0])
        weigthtype = type(edgelist[0][2])
        # Vorrei mettere una condizione sul weighttype che deve essere numerico
        edgelist = np.array(
            edgelist,
            dtype=np.dtype(
                [
                    ("source", nodetype),
                    ("target", nodetype),
                    ("weigth", weigthtype),
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
                [("source", int), ("target", int), ("weigth", weigthtype)]
            ),
        )
    if len(edgelist[0]) == 3:
        aux_edgelist = np.concatenate(
            (edgelist_new["source"], edgelist_new["target"])
        )
        aux_weights = np.concatenate(
            (edgelist_new["weigth"], edgelist_new["weigth"])
        )
        strength_seq = np.array(
            [aux_weights[aux_edgelist == i].sum() for i in unique_nodes]
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
    if len(edgelist[0]) == 2:
        nodetype = type(edgelist[0][0])
        edgelist = np.array(
            edgelist,
            dtype=np.dtype(
                [
                    ("source", nodetype),
                    ("target", nodetype)
                ]
            ),
        )
    else:
        nodetype = type(edgelist[0][0])
        weigthtype = type(edgelist[0][2])
        # Vorrei mettere una condizione sul weighttype che deve essere numerico
        edgelist = np.array(
            edgelist,
            dtype=np.dtype(
                [
                    ("source", nodetype),
                    ("target", nodetype),
                    ("weigth", weigthtype),
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
                [("source", int), ("target", int), ("weigth", weigthtype)]
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
        out_strength = np.zeros_like(unique_nodes, dtype=weigthtype)
        in_strength = np.zeros_like(unique_nodes, dtype=weigthtype)
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
