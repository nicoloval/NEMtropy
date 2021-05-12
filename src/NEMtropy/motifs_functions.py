import scipy
import numpy as np


def dyads_count(a):
    """Counts number of dyads.
    :param a np.ndarray: adjacency matrix
    :return: dyads count
    :rtype: int
    """

    at = a.transpose()
    tmp = a + at
    if isinstance(a, np.ndarray):
        return int(len(tmp[tmp == 2]))
    if isinstance(
        a,
        (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)
            ):
        return int(tmp[tmp == 2].shape[1])


def singles_count(a):
    """Counts number of singles.
    :param a np.ndarray: adjacency matrix
    :return: singles count
    :rtype: int
    """

    at = a.transpose()
    tmp = a + at
    if isinstance(a, np.ndarray):
        return int(len(tmp[tmp == 1])/2)
    if isinstance(
        a,
        (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)
            ):
        return int(tmp[tmp == 1].shape[1]/2)


def zeros_count(a):
    """Counts number of zeros.
    :param a np.ndarray: adjacency matrix
    :return: zeros count
    :rtype: int
    """

    n = a.shape[0]
    at = a.transpose()
    tmp = a + at
    if isinstance(a, np.ndarray):
        return int((n*(n-1) - np.count_nonzero(tmp)))
    if isinstance(
        a,
        (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)
            ):
        return int((n*(n-1) - tmp.count_nonzero()))
