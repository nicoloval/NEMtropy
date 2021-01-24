import numpy as np
import scipy.sparse
import scipy
from numba import jit, prange
import time
from .Directed_new import *
import os
from . import ensemble_generator as eg
# Stops Numba Warning for experimental feature
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings

warnings.simplefilter(
    action='ignore',
    category=NumbaExperimentalFeatureWarning)


def out_degree(a):
    """Compute matrix out degrees sequence

    :param a: matrix
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: out degree sequence
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1  # noqa


def in_degree(a):
    """Compute matrix in degree sequence

    :param a: matrix
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: in degree sequence
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 0).A1


def out_strength(a):
    """Compute matrix out strengths sequence

    :param a: matrix
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: out strengths sequence
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 1).A1


def in_strength(a):
    """Compute matrix in strengths sequence

    :param a: matrix
    :type a: numpy.ndarray, scipy.sparse.csr.csr_matrix,
        scipy.sparse.coo.coo_matrix
    :return: in strengths sequence
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 0).A1


@jit(nopython=True)
def pmatrix_dcm(x, args):
    """Function evaluating the DBCM probability matrix given the solution of
        the underlying model.

    :param x: DBCM solution
    :type x: numpy.ndarray
    :param args: tuple containing problem dimension,
        out and in indices of non-zero nodes
    :type args: (int, numpy.ndarray, numpy.ndarray)
    :return: DBCM probability matrix
    :rtype: numpy.ndarray
    """
    n = args[0]
    index_out = args[1]
    index_in = args[2]
    p = np.zeros((n, n), dtype=np.float64)
    xout = x[:n]
    yin = x[n:]
    for i in index_out:
        for j in index_in:
            if i != j:
                aux = xout[i] * yin[j]
                p[i, j] = aux / (1 + aux)
    return p


@jit(nopython=True)
def iterative_crema(beta, args):
    """Return the next iterative step for the CReMa Model.

    :param beta: previous solution iterative step
    :type beta: numpy.ndarray
    :param args: tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: next solution iterative step
    :rtype: numpy.ndarray
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    xd = np.zeros(aux_n, dtype=np.float64)
    yd = np.zeros(aux_n, dtype=np.float64)

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        xd[i] -= w / (1 + (beta_in[j] / beta_out[i]))
        yd[j] -= w / (1 + (beta_out[i] / beta_in[j]))
    for i in nz_index_out:
        xd[i] = xd[i] / s_out[i]
    for i in nz_index_in:
        yd[i] = yd[i] / s_in[i]

    return np.concatenate((xd, yd))


@jit(nopython=True)
def iterative_crema_sparse_2(beta, args):
    """Return the next iterative version for the CReMa
        sparse initialisation. Alternative version not in use.

    :param beta: previous solution iterative step
    :type beta: numpy.ndarray
    :param args: tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: next solution iterative step
    :rtype: numpy.ndarray]
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    xd = np.zeros(aux_n, dtype=np.float64)
    yd = np.zeros(aux_n, dtype=np.float64)

    x = adj[0]
    y = adj[1]

    for i in x.nonzero()[0]:
        for j in y.nonzero()[0]:
            if i != j:
                aux = x[i] * y[j]
                aux_entry = aux / (1 + aux)
                if aux_entry > 0:
                    aux = aux_entry / (1 + (beta_in[j] / beta_out[i]))
                    xd[i] -= aux
                    aux = aux_entry / (1 + (beta_out[i] / beta_in[j]))
                    yd[j] -= aux
    for i in nz_index_out:
        xd[i] = xd[i] / s_out[i]
    for i in nz_index_in:
        yd[i] = yd[i] / s_in[i]

    return np.concatenate((xd, yd))


@jit(nopython=True, parallel=True)
def iterative_crema_sparse(beta, args):
    """Return the next iterative version for the CReMa
        sparse initialisation.

    :param beta: previous solution iterative step
    :type beta: numpy.ndarray
    :param args: tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: next solution iterative step
    :rtype: numpy.ndarray]
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    # TODO: following 2 lines to delete?
    # nz_index_out = args[3]
    # nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    xd = np.zeros(aux_n, dtype=np.float64)
    yd = np.zeros(aux_n, dtype=np.float64)

    x = adj[0]
    y = adj[1]

    x_indices = x.nonzero()[0]
    y_indices = y.nonzero()[0]

    for i in prange(x_indices.shape[0]):
        index = x_indices[i]
        aux = x[index] * y
        aux_entry = aux / (1 + aux)
        aux = aux_entry / (1 + (beta_in / (beta_out[index]+np.exp(-100))))
        xd[index] -= (aux.sum() - aux[index])
        xd[index] = xd[index] / (s_out[index]+np.exp(-100))

    for j in prange(y_indices.shape[0]):
        index = y_indices[j]
        aux = x * y[index]
        aux_entry = aux / (1 + aux)
        aux = aux_entry / (1 + (beta_out / (beta_in[index] + np.exp(-100))))
        yd[j] -= (aux.sum() - aux[index])
        yd[j] = yd[index] / (s_in[index]+np.exp(-100))

    return np.concatenate((xd, yd))


@jit(nopython=True)
def loglikelihood_crema(beta, args):
    """Returns CReMa loglikelihood function evaluated in beta.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: float
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = 0.0

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i in nz_index_out:
        f -= s_out[i] * beta_out[i]
    for i in nz_index_in:
        f -= s_in[i] * beta_in[i]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        f += w * np.log(beta_out[i] + beta_in[j])

    return f


@jit(nopython=True)
def loglikelihood_crema_sparse(beta, args):
    """Returns CReMa loglikelihood function evaluated in beta.
        Sparse initialisation version.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: float
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = 0.0

    x = adj[0]
    y = adj[1]

    for i in nz_index_out:
        f -= s_out[i] * beta_out[i]
        for j in nz_index_in:
            if i != j:
                aux = x[i] * y[j]
                aux_entry = aux / (1 + aux)
                if aux_entry > 0:
                    f += aux_entry * np.log(beta_out[i] + beta_in[j])

    for i in nz_index_in:
        f -= s_in[i] * beta_in[i]

    return f


@jit(nopython=True)
def loglikelihood_prime_crema(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[0:aux_n]
    beta_in = beta[aux_n: 2 * aux_n]

    aux_f_out = np.zeros_like(beta_out, dtype=np.float64)
    aux_f_in = np.zeros_like(beta_in, dtype=np.float64)

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]
    for i in nz_index_out:
        aux_f_out[i] -= s_out[i]
    for i in nz_index_in:
        aux_f_in[i] -= s_in[i]
    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux_f_out[i] += w / (beta_out[i] + beta_in[j])
        aux_f_in[j] += w / (beta_out[i] + beta_in[j])

    return np.concatenate((aux_f_out, aux_f_in))


@jit(nopython=True)
def loglikelihood_prime_crema_sparse_2(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
        Sparse initialisation alternative version.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    # TODO: following 2 lines to delete?
    # nz_index_out = args[3]
    # nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[0:aux_n]
    beta_in = beta[aux_n: 2 * aux_n]

    aux_f_out = np.zeros_like(beta_out, dtype=np.float64)
    aux_f_in = np.zeros_like(beta_in, dtype=np.float64)

    x = adj[0]
    y = adj[1]

    for i in x.nonzero()[0]:
        aux_f_out[i] -= s_out[i]
        for j in y.nonzero()[0]:
            if i != j:
                aux = x[i] * y[j]
                aux_value = aux / (1 + aux)
                if aux_value > 0:
                    aux_f_out[i] += aux_value / (beta_out[i] + beta_in[j])
                    aux_f_in[j] += aux_value / (beta_out[i] + beta_in[j])

    for j in y.nonzero()[0]:
        aux_f_in[j] -= s_in[j]
    return np.concatenate((aux_f_out, aux_f_in))


@jit(nopython=True, parallel=True)
def loglikelihood_prime_crema_sparse(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
        Sparse initialisation version.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[0:aux_n]
    beta_in = beta[aux_n: 2 * aux_n]

    aux_F_out = np.zeros_like(beta_out, dtype=np.float64)
    aux_F_in = np.zeros_like(beta_in, dtype=np.float64)

    x = adj[0]
    y = adj[1]

    for i in prange(nz_index_out.shape[0]):
        index = nz_index_out[i]
        aux_F_out[index] -= s_out[index]
        aux = x[index] * y
        aux_value = aux / (1 + aux)
        aux = aux_value / (beta_out[index] + beta_in)
        aux_F_out[index] += aux.sum() - aux[index]

    for j in prange(nz_index_in.shape[0]):
        index = nz_index_in[j]
        aux_F_in[index] -= s_in[index]
        aux = x * y[index]
        aux_value = aux / (1 + aux)
        aux = aux_value / (beta_out + beta_in[index])
        aux_F_in[index] += (aux.sum() - aux[index])

    return np.concatenate((aux_F_out, aux_F_in))


@jit(nopython=True)
def loglikelihood_hessian_crema(beta, args):
    """Returns CReMa loglikelihood hessian function evaluated in beta.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    # TODO: remove commented lines?
    s_out = args[0]
    # s_in = args[1]
    adj = args[2]
    # nz_index_out = args[3]
    # nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(shape=(2 * aux_n, 2 * aux_n), dtype=np.float64)

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux = w / ((beta_out[i] + beta_in[j]) ** 2)
        f[i, i] += -aux
        f[i, j + aux_n] = -aux
        f[j + aux_n, i] = -aux
        f[j + aux_n, j + aux_n] += -aux
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_crema(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian
        function evaluated in beta.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    # TODO: remove commented lines?
    s_out = args[0]
    # s_in = args[1]
    adj = args[2]
    # nz_index_out = args[3]
    # nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(2 * aux_n, dtype=np.float64)

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]
    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        f[i] -= w / ((beta_out[i] + beta_in[j]) ** 2)
        f[j + aux_n] -= w / ((beta_out[i] + beta_in[j]) ** 2)

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_crema_sparse_2(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian
        function evaluated in beta. 
        Sparse initialisation alternative version.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    # TODO: remove commented lines?
    s_out = args[0]
    # s_in = args[1]
    adj = args[2]
    # nz_index_out = args[3]
    # nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(2 * aux_n, dtype=np.float64)

    x = adj[0]
    y = adj[1]

    for i in np.arange(aux_n):
        for j in np.arange(aux_n):
            if i != j:
                aux = x[i] * y[j]
                aux_entry = aux / (1 + aux)
                if aux_entry > 0:
                    f[i] -= aux_entry / ((beta_out[i] + beta_in[j]) ** 2)
                aux = x[j] * y[i]
                aux_entry = aux / (1 + aux)
                if aux_entry > 0:
                    f[i + aux_n] -= aux_entry / (
                        (beta_out[j] + beta_in[i]) ** 2
                    )
    return f


@jit(nopython=True, parallel=True)
def loglikelihood_hessian_diag_crema_sparse(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian
        function evaluated in beta.
        Sparse initialisation version.

    :param beta: Evaluating point
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function
        Tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: numpy.ndarray
    """
    # TODO: remove commented lines?
    s_out = args[0]
    # s_in = args[1]
    adj = args[2]
    # nz_index_out = args[3]
    # nz_index_in = args[4]

    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(2 * aux_n, dtype=np.float64)

    x = adj[0]
    y = adj[1]

    for i in prange(aux_n):
        aux = x[i] * y
        aux_entry = aux / (1 + aux)
        aux = aux_entry / (((beta_out[i] + beta_in) ** 2) + np.exp(-100))
        f[i] -= (aux.sum() - aux[i])

        aux = x * y[i]
        aux_entry = aux / (1 + aux)
        aux = aux_entry / (((beta_out + beta_in[i]) ** 2) + np.exp(-100))
        f[i + aux_n] -= (aux.sum() - aux[i])

    return f


@jit(nopython=True)
def loglikelihood_dcm(x, args):
    """Returns DBCM loglikelihood function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function
        Tuple containing out and in degrees sequences,
        non zero out and in indices, and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value
    :rtype: float
    """
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_out)

    f = 0
    for i in nz_index_out:
        f += c[i] * k_out[i] * np.log(x[i])
        for j in nz_index_in:
            if i != j:
                f -= c[i] * c[j] * np.log(1 + x[i] * x[n + j])
            else:
                f -= c[i] * (c[i] - 1) * np.log(1 + x[i] * x[n + j])

    for j in nz_index_in:
        f += c[j] * k_in[j] * np.log(x[j + n])

    return f


@jit(nopython=True)
def loglikelihood_prime_dcm(x, args):
    """Returns DBCM loglikelihood gradient function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        non zero out and in indices, and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient array
    :rtype: numpy.ndarray
    """
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_in)

    f = np.zeros(2 * n)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i != j:
                const = c[i] * c[j]
                # const = c[j]
            else:
                const = c[i] * (c[j] - 1)
                # const = (c[j] - 1)

            fx += const * x[j + n] / (1 + x[i] * x[j + n])
        # original prime
        f[i] = -fx + c[i] * k_out[i] / x[i]

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i != j:
                const = c[i] * c[j]
                # const = c[i]
            else:
                const = c[i] * (c[j] - 1)
                # const = (c[j] - 1)

            fy += const * x[i] / (1 + x[j + n] * x[i])
        # original prime
        f[j + n] = -fy + c[j] * k_in[j] / x[j + n]

    return f


@jit(nopython=True)
def loglikelihood_hessian_dcm(x, args):
    """Returns DBCM loglikelihood hessian function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        non zero out and in indices, and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix
    :rtype: numpy.ndarray
    """
    k_out = args[0]
    k_in = args[1]
    nz_out_index = args[2]
    nz_in_index = args[3]
    c = args[4]
    n = len(k_out)

    out = np.zeros((2 * n, 2 * n))  # hessian matrix
    # zero elemnts in x have hessian -1

    for h in nz_out_index:
        out[h, h] = -c[h] * k_out[h] / (x[h]) ** 2
        for i in nz_in_index:
            if i == h:
                const = c[h] * (c[h] - 1)
                # const = (c[h] - 1)
            else:
                const = c[h] * c[i]
                # const = c[i]

            out[h, h] += const * (x[i + n] / (1 + x[h] * x[i + n])) ** 2
            out[h, i + n] = -const / (1 + x[i + n] * x[h]) ** 2

    for i in nz_in_index:
        out[i + n, i + n] = -c[i] * k_in[i] / (x[i + n] * x[i + n])
        for h in nz_out_index:
            if i == h:
                const = c[h] * (c[h] - 1)
                # const = (c[i] - 1)
            else:
                const = c[h] * c[i]
                # const = c[h]

            out[i + n, i + n] += (
                const * (x[h] ** 2) / (1 + x[i + n] * x[h]) ** 2
            )
            out[i + n, h] = -const / (1 + x[i + n] * x[h]) ** 2

    return out


@jit(nopython=True)
def iterative_dcm(x, args):
    """Returns the next iterative step for the DBCM.

    :param : previous solution iterative step
    :type : numpy.ndarray
    :param args: tuple containing out and in strengths sequences,
        adjacency matrix, and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: next solution iterative step
    :rtype: numpy.ndarray
    """
    # TODO: remove commented lines?
    k_out = args[0]
    k_in = args[1]
    n = len(k_out)
    # nz_index_out = args[2]
    # nz_index_in = args[3]
    nz_index_out = range(n)
    nz_index_in = range(n)
    c = args[4]

    f = np.zeros(2 * n)

    for i in nz_index_out:
        for j in nz_index_in:
            if j != i:
                f[i] += c[j] * x[j + n] / (1 + x[i] * x[j + n])
            else:
                f[i] += (c[j] - 1) * x[j + n] / (1 + x[i] * x[j + n])

    for j in nz_index_in:
        for i in nz_index_out:
            if j != i:
                f[j + n] += c[i] * x[i] / (1 + x[i] * x[j + n])
            else:
                f[j + n] += (c[i] - 1) * x[i] / (1 + x[i] * x[j + n])

    tmp = np.concatenate((k_out, k_in))
    # ff = np.array([tmp[i]/f[i] if tmp[i] != 0 else 0 for i in range(2*n)])
    ff = tmp / f

    return ff


@jit(nopython=True)
def loglikelihood_hessian_diag_dcm(x, args):
    """Returns the diagonal of DBCM loglikelihood hessian function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        non zero out and in indices, and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix diagonal
    :rtype: numpy.ndarray
    """
    # problem fixed paprameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_in)

    f = np.zeros(2 * n)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i != j:
                # const = c[i]*c[j]
                const = c[j]
            else:
                # const = c[i]*(c[j] - 1)
                const = c[i] - 1

            tmp = 1 + x[i] * x[j + n]
            fx += const * x[j + n] * x[j + n] / (tmp * tmp)
        # original prime
        f[i] = fx - k_out[i] / (x[i] * x[i])

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i != j:
                # const = c[i]*c[j]
                const = c[i]
            else:
                # const = c[i]*(c[j] - 1)
                const = c[j] - 1

            tmp = (1 + x[j + n] * x[i]) * (1 + x[j + n] * x[i])
            fy += const * x[i] * x[i] / (tmp)
        # original prime
        f[j + n] = fy - k_in[j] / (x[j + n] * x[j + n])

    # f[f == 0] = 1

    return f


@jit(nopython=True)
def loglikelihood_decm(x, args):
    """Returns DECM loglikelihood function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood value
    :rtype: float
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    f = 0
    for i in range(n):
        if k_out[i]:
            f += k_out[i] * np.log(x[i])
        if k_in[i]:
            f += k_in[i] * np.log(x[i + n])
        if s_out[i]:
            f += s_out[i] * np.log(x[i + 2 * n])
        if s_in[i]:
            f += s_in[i] * np.log(x[i + 3 * n])
        for j in range(n):
            if i != j:
                tmp = x[i + 2 * n] * x[j + 3 * n]
                f += np.log(1 - tmp)
                f -= np.log(1 - tmp + tmp * x[i] * x[j + n])
    return f


@jit(nopython=True)
def loglikelihood_prime_decm(x, args):
    """Returns DECM loglikelihood gradient function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood gradient array
    :rtype: numpy.array
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    f = np.zeros(4 * n)
    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp = x[i + 2 * n] * x[j + 3 * n]
                fa_out += x[j + n] * tmp / (1 - tmp + x[i] * x[j + n] * tmp)
                tmp = x[j + 2 * n] * x[i + 3 * n]
                fa_in += x[j] * tmp / (1 - tmp + x[j] * x[i + n] * tmp)
                tmp = x[j + 3 * n] * x[i + 2 * n]
                if x[i]:
                    fb_out += x[j + 3 * n] / (1 - tmp) + (
                        x[j + n] * x[i] - 1
                    ) * x[j + 3 * n] / (1 - tmp + x[i] * x[j + n] * tmp)
                else:
                    fb_out += 0
                tmp = x[i + 3 * n] * x[j + 2 * n]
                fb_in += x[j + 2 * n] / (1 - tmp) + (x[i + n] * x[j] - 1) * x[
                    j + 2 * n
                ] / (1 - tmp + x[j] * x[i + n] * tmp)
        if k_out[i]:
            f[i] = k_out[i] / x[i] - fa_out
        else:
            f[i] = -fa_out
        if k_in[i]:
            f[i + n] = k_in[i] / x[i + n] - fa_in
        else:
            f[i + n] = -fa_in
        if s_out[i]:
            f[i + 2 * n] = s_out[i] / x[i + 2 * n] - fb_out
        else:
            f[i + 2 * n] = -fb_out
        if s_in[i]:
            f[i + 3 * n] = s_in[i] / x[i + 3 * n] - fb_in
        else:
            f[i + 3 * n] = -fb_in

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_decm(x, args):
    """Returns the diagonal of DECM loglikelihood hessian function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood hessian matrix
    :rtype: numpy.ndarray
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    f = np.zeros(4 * n)
    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp0 = x[i + 2 * n] * x[j + 3 * n]
                tmp = (x[j + n] * tmp0) / (1 - tmp0 + x[i] * x[j + n] * tmp0)
                fa_out += tmp * tmp

                tmp0 = x[j + 2 * n] * x[i + 3 * n]
                tmp = (x[j] * tmp0) / (1 - tmp0 + x[j] * x[i + n] * tmp0)
                fa_in += tmp * tmp

                tmp0 = x[j + 3 * n] * x[i + 2 * n]
                tmp1 = x[j + 3 * n] / (1 - tmp0)
                tmp2 = ((x[j + n] * x[i] - 1) * x[j + 3 * n]) / (
                    1 - tmp0 + x[i] * x[j + n] * tmp0
                )
                fb_out += tmp1 * tmp1 - tmp2 * tmp2

                # P2
                tmp0 = x[i + 3 * n] * x[j + 2 * n]
                tmp1 = x[j + 2 * n] / (1 - tmp0)
                tmp2 = ((x[i + n] * x[j] - 1) * x[j + 2 * n]) / (
                    1 - tmp0 + x[j] * x[i + n] * tmp0
                )
                fb_in += tmp1 * tmp1 - tmp2 * tmp2
        if k_out[i]:
            f[i] = -k_out[i] / x[i] ** 2 + fa_out
        else:
            f[i] = fa_out
        if k_in[i]:
            f[i + n] = -k_in[i] / x[i + n] ** 2 + fa_in
        else:
            f[i + n] = fa_in
        if s_out[i]:
            f[i + 2 * n] = -s_out[i] / x[i + 2 * n] ** 2 - fb_out
        else:
            f[i + 2 * n] = -fb_out
        if s_in[i]:
            f[i + 3 * n] = -s_in[i] / x[i + 3 * n] ** 2 - fb_in
        else:
            f[i + 3 * n] = fb_in

    return f


@jit(nopython=True)
def loglikelihood_hessian_decm(x, args):
    """Returns DECM loglikelihood hessian function evaluated in x.

    :param x: Evaluating point
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood hessian matrix
    :rtype: numpy.ndarray
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]

    n = len(k_out)
    f = np.zeros((n * 4, n * 4))

    a_out = x[:n]
    a_in = x[n: 2 * n]
    b_out = x[2 * n: 3 * n]
    b_in = x[3 * n:]

    for h in range(n):
        for t in range(n):
            if h == t:
                # dll / da^in da^in
                if k_in[h]:
                    f[h + n, t + n] = -k_in[h] / a_in[h] ** 2
                # dll / da^out da^out
                if k_out[h]:
                    f[h, t] = -k_out[h] / a_out[h] ** 2
                # dll / db^in db^in
                if s_in[h]:
                    f[h + 3 * n, t + 3 * n] = -s_in[h] / b_in[h] ** 2
                # dll / db^out db^out
                if s_out[h]:
                    f[h + 2 * n, t + 2 * n] = -s_out[h] / b_out[h] ** 2

                for j in range(n):
                    if j != h:
                        # dll / da^in da^in
                        f[h + n, t + n] = (
                            f[h + n, t + n]
                            + (
                                a_out[j]
                                * b_in[h]
                                * b_out[j]
                                / (
                                    1
                                    - b_in[h] * b_out[j]
                                    + a_in[h] * a_out[j] * b_in[h] * b_out[j]
                                )
                            )
                            ** 2
                        )
                        # dll / da^in db^in
                        f[h + n, t + 3 * n] = (
                            f[h + n, t + 3 * n]
                            - a_out[j]
                            * b_out[j]
                            / (
                                1
                                - b_in[h] * b_out[j]
                                + a_in[h] * a_out[j] * b_in[h] * b_out[j]
                            )
                            ** 2
                        )
                        # dll / da^out da^out
                        f[h, t] = (
                            f[h, t]
                            + (a_in[j] * b_in[j] * b_out[h]) ** 2
                            / (
                                1
                                - b_in[j] * b_out[h]
                                + a_in[j] * a_out[h] * b_in[j] * b_out[h]
                            )
                            ** 2
                        )
                        # dll / da^out db^out
                        f[h, t + 2 * n] = (
                            f[h, t + 2 * n]
                            - a_in[j]
                            * b_in[j]
                            / (
                                1
                                - b_in[j] * b_out[h]
                                + a_in[j] * a_out[h] * b_in[j] * b_out[h]
                            )
                            ** 2
                        )
                        # dll / db^in da^in
                        f[h + 3 * n, t + n] = (
                            f[h + 3 * n, t + n]
                            - a_out[j]
                            * b_out[j]
                            / (
                                1
                                - b_in[h] * b_out[j]
                                + a_in[h] * a_out[j] * b_in[h] * b_out[j]
                            )
                            ** 2
                        )
                        # dll / db_in db_in
                        f[h + 3 * n, t + 3 * n] = (
                            f[h + 3 * n, t + 3 * n]
                            - (b_out[j] / (1 - b_in[h] * b_out[j])) ** 2
                            + (
                                b_out[j]
                                * (a_in[h] * a_out[j] - 1)
                                / (
                                    1
                                    - b_in[h] * b_out[j]
                                    + a_in[h] * a_out[j] * b_in[h] * b_out[j]
                                )
                            )
                            ** 2
                        )
                        # dll / db_out da_out
                        f[h + 2 * n, t] = (
                            f[h + 2 * n, t]
                            - a_in[j]
                            * b_in[j]
                            / (
                                1
                                - b_in[j] * b_out[h]
                                + a_in[j] * a_out[h] * b_in[j] * b_out[h]
                            )
                            ** 2
                        )
                        # dll / db^out db^out
                        f[h + 2 * n, t + 2 * n] = (
                            f[h + 2 * n, t + 2 * n]
                            - (b_in[j] / (1 - b_in[j] * b_out[h])) ** 2
                            + (
                                (a_in[j] * a_out[h] - 1)
                                * b_in[j]
                                / (
                                    1
                                    - b_in[j] * b_out[h]
                                    + a_in[j] * a_out[h] * b_in[j] * b_out[h]
                                )
                            )
                            ** 2
                        )

            else:
                # dll / da_in da_out
                f[h + n, t] = (
                    -b_in[h]
                    * b_out[t]
                    * (1 - b_in[h] * b_out[t])
                    / (
                        1
                        - b_in[h] * b_out[t]
                        + a_in[h] * a_out[t] * b_in[h] * b_out[t]
                    )
                    ** 2
                )
                # dll / da_in db_out
                f[h + n, t + 2 * n] = (
                    -a_out[t]
                    * b_in[h]
                    / (
                        1
                        - b_in[h] * b_out[t]
                        + a_in[h] * a_out[t] * b_in[h] * b_out[t]
                    )
                    ** 2
                )
                # dll / da_out da_in
                f[h, t + n] = (
                    -b_in[t]
                    * b_out[h]
                    * (1 - b_in[t] * b_out[h])
                    / (
                        1
                        - b_in[t] * b_out[h]
                        + a_in[t] * a_out[h] * b_in[t] * b_out[h]
                    )
                    ** 2
                )
                # dll / da_out db_in
                f[h, t + 3 * n] = (
                    -a_in[t]
                    * b_out[h]
                    / (
                        1
                        - b_in[t] * b_out[h]
                        + a_in[t] * a_out[h] * b_in[t] * b_out[h]
                    )
                    ** 2
                )
                # dll / db_in da_out
                f[h + 3 * n, t] = (
                    -a_in[h]
                    * b_out[t]
                    / (
                        1
                        - b_in[h] * b_out[t]
                        + a_in[h] * a_out[t] * b_in[h] * b_out[t]
                    )
                    ** 2
                )
                # dll / db_in db_out
                f[h + 3 * n, t + 2 * n] = (
                    -1 / (1 - b_in[h] * b_out[t]) ** 2
                    - (a_out[t] * a_in[h] - 1)
                    / (
                        1
                        - b_in[h] * b_out[t]
                        + a_in[h] * a_out[t] * b_in[h] * b_out[t]
                    )
                    ** 2
                )
                # dll / db_out da_in
                f[h + 2 * n, t + n] = (
                    -a_out[h]
                    * b_in[t]
                    / (
                        1
                        - b_in[t] * b_out[h]
                        + a_in[t] * a_out[h] * b_in[t] * b_out[h]
                    )
                    ** 2
                )
                # dll / db_out db_in
                f[h + 2 * n, t + 3 * n] = (
                    -1 / (1 - b_in[t] * b_out[h]) ** 2
                    - (a_in[t] * a_out[h] - 1)
                    / (
                        1
                        - b_in[t] * b_out[h]
                        + a_in[t] * a_out[h] * b_in[t] * b_out[h]
                    )
                    ** 2
                )

    return f


@jit(nopython=True)
def iterative_decm(x, args):
    """Return the next iterative step for the DECM Model.

    :param : Previous solution iterative step
    :type : numpy.ndarray
    :param args: Tuple containing out and in degrees sequences,
        and out and in strengths sequences
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next solution iterative step
    :rtype: numpy.ndarray
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    f = np.zeros(4 * n)

    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp = x[i + 2 * n] * x[j + 3 * n]
                fa_out += x[j + n] * tmp / (1 - tmp + x[i] * x[j + n] * tmp)

                tmp = x[j + 2 * n] * x[i + 3 * n]
                fa_in += x[j] * tmp / (1 - tmp + x[j] * x[i + n] * tmp)

                tmp = x[j + 3 * n] * x[i + 2 * n]
                tmp0 = x[j + n] * x[i]
                fb_out += x[j + 3 * n] / (1 - tmp) + (tmp0 - 1) * x[
                    j + 3 * n
                ] / (1 - tmp + tmp0 * tmp)
                tmp = x[i + 3 * n] * x[j + 2 * n]
                tmp0 = x[i + n] * x[j]
                fb_in += x[j + 2 * n] / (1 - tmp) + (tmp0 - 1) * x[
                    j + 2 * n
                ] / (1 - tmp + tmp0 * tmp)

        """
        f[i] = k_out[i]/fa_out
        f[i+n] = k_in[i]/fa_in
        f[i+2*n] = s_out[i]/fb_out
        f[i+3*n] = s_in[i]/fb_in

        """
        if k_out[i]:
            f[i] = k_out[i] / fa_out
        else:
            f[i] = 0
        if k_in[i]:
            f[i + n] = k_in[i] / fa_in
        else:
            f[i + n] = 0
        if s_out[i]:
            f[i + 2 * n] = s_out[i] / fb_out
        else:
            f[i + 2 * n] = 0
        if s_in[i]:
            f[i + 3 * n] = s_in[i] / fb_in
        else:
            f[i + 3 * n] = 0

    return f


@jit(nopython=True)
def expected_out_degree_dcm(sol):
    """Expected out-degree after the DBCM.

    :param sol: DBCM solution
    :type sol: numpy.ndarray
    :return: Expected out-degree sequence
    :rtype: numpy.ndarray
    """
    n = int(len(sol) / 2)
    a_out = sol[:n]
    a_in = sol[n:]

    k = np.zeros(n)  # allocate k
    for i in a_out.nonzero()[0]:
        for j in a_in.nonzero()[0]:
            if i != j:
                k[i] += a_in[j] * a_out[i] / (1 + a_in[j] * a_out[i])

    return k


@jit(nopython=True)
def expected_in_degree_dcm(sol):
    """Expected in-degree after the DBCM.

    :param sol: DBCM solution
    :type sol: numpy.ndarray
    :return: Expected in-degree sequence
    :rtype: numpy.ndarray
    """
    n = int(len(sol) / 2)
    a_out = sol[:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k
    for i in a_in.nonzero()[0]:
        for j in a_out.nonzero()[0]:
            if i != j:
                k[i] += a_in[i] * a_out[j] / (1 + a_in[i] * a_out[j])

    return k


@jit(nopython=True)
def expected_decm(x):
    """Expected parameters after the DECM.
       It returns a concatenated array of out-degrees, in-degrees,
       out-strengths, in-strengths.

    :param x: DECM solution
    :type x: numpy.ndarray
    :return: DECM expected parameters sequence
    :rtype: numpy.ndarray
    """
    n = int(len(x) / 4)
    f = np.zeros(len(x))

    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                fa_out += (
                    x[j + n]
                    * x[i + 2 * n]
                    * x[j + 3 * n]
                    / (
                        1
                        - x[i + 2 * n] * x[j + 3 * n]
                        + x[i] * x[j + n] * x[i + 2 * n] * x[j + 3 * n]
                    )
                )
                fa_in += (
                    x[j]
                    * x[j + 2 * n]
                    * x[i + 3 * n]
                    / (
                        1
                        - x[j + 2 * n] * x[i + 3 * n]
                        + x[j] * x[i + n] * x[j + 2 * n] * x[i + 3 * n]
                    )
                )
                fb_out += x[j + 3 * n] / (1 - x[j + 3 * n] * x[i + 2 * n]) + (
                    x[j + n] * x[i] - 1
                ) * x[j + 3 * n] / (
                    1
                    - x[i + 2 * n] * x[j + 3 * n]
                    + x[i] * x[j + n] * x[i + 2 * n] * x[j + 3 * n]
                )
                fb_in += x[j + 2 * n] / (1 - x[i + 3 * n] * x[j + 2 * n]) + (
                    x[i + n] * x[j] - 1
                ) * x[j + 2 * n] / (
                    1
                    - x[j + 2 * n] * x[i + 3 * n]
                    + x[j] * x[i + n] * x[j + 2 * n] * x[i + 3 * n]
                )
        f[i] = x[i] * fa_out
        f[i + n] = x[i + n] * fa_in
        f[i + 2 * n] = x[i + 2 * n] * fb_out
        f[i + 3 * n] = x[i + 3 * n] * fb_in

    return f


def hessian_regulariser_function(b, eps):
    """Trasform input matrix in a positive defined matrix
    by adding positive quantites to the main diagonal.

    :param b:Matrix
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    bf = b + np.identity(b.shape[0]) * eps

    return bf


def hessian_regulariser_function_eigen_based(b, eps):
    """Trasform input matrix in a positive defined matrix
    by regularising eigenvalues.

    :param b:Matrix
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    t, e = scipy.linalg.eigh(b)
    ll = np.array([0 if li > eps else eps - li for li in t])
    bf = e @ (np.diag(ll) + np.diag(t)) @ e.transpose()

    return bf


def hessian_regulariser_function_old(b, eps):
    """Trasform input matrix in a positive defined matrix.
        Obsolete version.

    :param b:Matrix
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    t, e = scipy.linalg.eigh(b)
    eps = eps * np.max(t)
    ll = np.array([0 if li > eps else eps - li for li in t])
    bf = e @ (np.diag(ll) + np.diag(t)) @ e.transpose()
    return bf


@jit(nopython=True)
def expected_out_strength_crema(sol, adj):
    """Expected out-strength after CReMa.

    :param sol: CReMa solution
    :type sol: numpy.ndarray
    :param adj: Tuple containing the original topology edges list
         and link related weigths.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :return: Expected out-strength sequence
    :rtype: numpy.ndarray
    """
    n = int(sol.size / 2)
    b_out = sol[:n]
    b_in = sol[n:]
    s = np.zeros(n)

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        s[i] += w / (b_out[i] + b_in[j])
    return s


@jit(nopython=True)
def expected_out_strength_crema_sparse(sol, adj):
    """Expected out-strength after CReMa. Sparse initialisation version.

    :param sol: CReMa solution
    :type sol: numpy.ndarray
    :param adj: Tuple containing the binary problem solution.
    :type adj: (numpy.ndarray, numpy.ndarray)
    :return: Expected out-strength sequence
    :rtype: numpy.ndarray
    """
    n = int(sol.size / 2)
    b_out = sol[:n]
    b_in = sol[n:]
    s = np.zeros(n)

    x = adj[0]
    y = adj[1]

    for i in range(n):
        for j in range(n):
            if i != j:
                aux = x[i] * y[j]
                aux_entry = aux / (1 + aux)
                if aux_entry > 0:
                    s[i] += aux_entry / (b_out[i] + b_in[j])
    return s


@jit(nopython=True)
def expected_in_stregth_crema(sol, adj):
    """Expected in-strength after CReMa.

    :param sol: CReMa solution
    :type sol: numpy.ndarray
    :param adj: Tuple containing the original topology edges list
         and link related weigths.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :return: Expected in-strength sequence
    :rtype: numpy.ndarray
    """
    n = int(sol.size / 2)
    b_out = sol[:n]
    b_in = sol[n:]
    s = np.zeros(n)

    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        s[j] += w / (b_out[i] + b_in[j])

    return s


@jit(nopython=True)
def expected_in_stregth_crema_sparse(sol, adj):
    """Expected in-strength after CReMa. Sparse inisialization version.

    :param sol: CReMa solution
    :type sol: numpy.ndarray
    :param adj: Tuple containing the original topology edges list
         and link related weigths.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :return: Expected in-strength sequence
    :rtype: numpy.ndarray
    """
    n = int(sol.size / 2)
    b_out = sol[:n]
    b_in = sol[n:]
    s = np.zeros(n)

    x = adj[0]
    y = adj[1]

    for i in range(n):
        for j in range(n):
            if i != j:
                aux = x[j] * y[i]
                aux_entry = aux / (1 + aux)
                if aux_entry > 0:
                    s[i] += aux_entry / (b_out[j] + b_in[i])
    return s


def solver(
    x0,
    fun,
    step_fun,
    linsearch_fun,
    hessian_regulariser,
    fun_jac=None,
    tol=1e-6,
    eps=1e-10,
    max_steps=100,
    method="newton",
    verbose=False,
    regularise=True,
    regularise_eps=1e-3,
    full_return=False,
    linsearch=True,
):
    """Find roots of eq. fun = 0, using newton, quasinewton or
        fixed-point algorithm.

    :param x0: Initial point
    :type x0: numpy.ndarray
    :param fun: Function handle of the function to find the roots of.
    :type fun: function
    :param step_fun: Function to compute the algorithm step
    :type step_fun: function
    :param linsearch_fun: Function to compute the linsearch
    :type linsearch_fun: function
    :param hessian_regulariser: Function to regularise fun hessian
    :type hessian_regulariser: function
    :param fun_jac: Function to compute the hessian of fun, defaults to None
    :type fun_jac: function, optional
    :param tol: The solver stops when |fun|<tol, defaults to 1e-6
    :type tol: float, optional
    :param eps: The solver stops when the difference between two consecutive
        steps is less than eps, defaults to 1e-10
    :type eps: float, optional
    :param max_steps: Maximum number of steps the solver takes, defaults to 100
    :type max_steps: int, optional
    :param method: Method the solver uses to solve the problem.
        Choices are "newton", "quasinewton", "fixed-point".
        Defaults to "newton"
    :type method: str, optional
    :param verbose: If True the solver prints out information at each step,
         defaults to False
    :type verbose: bool, optional
    :param regularise: If True the solver will regularise the hessian matrix,
         defaults to True
    :type regularise: bool, optional
    :param regularise_eps: Positive value to pass to the regulariser function,
         defaults to 1e-3
    :type regularise_eps: float, optional
    :param full_return: If True the function returns information on the
        convergence, defaults to False
    :type full_return: bool, optional
    :param linsearch: If True a linsearch algorithm is implemented,
         defaults to True
    :type linsearch: bool, optional
    :return: Solution to the optimization problem
    :rtype: numpy.ndarray
    """
    tic_all = time.time()
    toc_init = 0
    tic = time.time()

    # algorithm
    beta = 0.5  # to compute alpha
    n_steps = 0
    x = x0  # initial point

    f = fun(x)
    norm = np.linalg.norm(f)
    diff = 1
    dx_old = np.zeros_like(x)

    if full_return:
        norm_seq = [norm]
        diff_seq = [diff]
        alfa_seq = []

    if verbose:
        print("\nx0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    toc_init = time.time() - tic

    toc_alfa = 0
    toc_update = 0
    toc_dx = 0
    toc_jacfun = 0

    tic_loop = time.time()

    while (
        norm > tol and n_steps < max_steps and diff > eps
    ):  # stopping condition

        x_old = x  # save previous iteration

        # f jacobian
        tic = time.time()
        if method == "newton":
            # regularise
            H = fun_jac(x)  # original jacobian
            # TODO: levare i verbose sugli eigenvalues
            if verbose:
                t, e = scipy.linalg.eigh(H)
                ml = np.min(t)
                Ml = np.max(t)
            if regularise:
                b_matrix = hessian_regulariser(
                    H, np.max(np.abs(fun(x))) * regularise_eps,
                )
                # TODO: levare i verbose sugli eigenvalues
                if verbose:
                    t, e = scipy.linalg.eigh(b_matrix)
                    new_ml = np.min(t)
                    new_Ml = np.max(t)
            else:
                b_matrix = H.__array__()
        elif method == "quasinewton":
            # quasinewton hessian approximation
            b_matrix = fun_jac(x)  # Jacobian diagonal
            if regularise:
                b_matrix = np.maximum(b_matrix,
                                      b_matrix * 0 + np.max(np.abs(fun(x)))
                                      * regularise_eps)
        toc_jacfun += time.time() - tic

        # discending direction computation
        tic = time.time()
        if method == "newton":
            dx = np.linalg.solve(b_matrix, -f)
            # dx = dx/np.linalg.norm(dx)
        elif method == "quasinewton":
            dx = -f / b_matrix
        elif method == "fixed-point":
            dx = f - x
            # TODO: hotfix to compute dx in infty cases
            for i in range(len(x)):
                if x[i] == np.infty:
                    dx[i] = np.infty
        toc_dx += time.time() - tic

        # backtraking line search
        tic = time.time()

        # Linsearch
        if linsearch and (method in ["newton", "quasinewton"]):
            alfa1 = 1
            X = (x, dx, beta, alfa1, f)
            alfa = linsearch_fun(X)
            if full_return:
                alfa_seq.append(alfa)
        elif linsearch and (method in ["fixed-point"]):
            alfa1 = 1
            X = (x, dx, dx_old, alfa1, beta, n_steps)
            alfa = linsearch_fun(X)
            if full_return:
                alfa_seq.append(alfa)
        else:
            alfa = 1

        toc_alfa += time.time() - tic

        tic = time.time()
        # solution update
        # direction= dx@fun(x).T

        x = x + alfa * dx

        dx_old = alfa * dx.copy()

        toc_update += time.time() - tic

        f = fun(x)

        # stopping condition computation
        norm = np.linalg.norm(f)
        diff_v = x - x_old
        diff_v[np.isnan(diff_v)] = 0  # to avoid nans given by inf-inf
        diff = np.linalg.norm(diff_v)

        if full_return:
            norm_seq.append(norm)
            diff_seq.append(diff)

        # step update
        n_steps += 1

        if verbose:
            print("\nstep {}".format(n_steps))
            # print("fun = {}".format(f))
            # print("dx = {}".format(dx))
            # print("x = {}".format(x))
            print("alpha = {}".format(alfa))
            print("|f(x)| = {}".format(norm))
            if method in ["newton", "quasinewton"]:
                print("F(x) = {}".format(step_fun(x)))
            print("diff = {}".format(diff))
            if method == "newton":
                print("min eig = {}".format(ml))
                print("new mim eig = {}".format(new_ml))
                print("max eig = {}".format(Ml))
                print("new max eig = {}".format(new_Ml))
                print("condition number max_eig/min_eig = {}".format(Ml / ml))
                print(
                    "new condition number max_eig/min_eig = {}".format(
                        new_Ml / new_ml
                    )
                )
                # print('\neig = {}'.format(t))

    toc_loop = time.time() - tic_loop
    toc_all = time.time() - tic_all

    if verbose:
        print("Number of steps for convergence = {}".format(n_steps))
        print("toc_init = {}".format(toc_init))
        print("toc_jacfun = {}".format(toc_jacfun))
        print("toc_alfa = {}".format(toc_alfa))
        print("toc_dx = {}".format(toc_dx))
        print("toc_update = {}".format(toc_update))
        print("toc_loop = {}".format(toc_loop))
        print("toc_all = {}".format(toc_all))

    if full_return:
        return (x, toc_all, n_steps, np.array(norm_seq),
                np.array(diff_seq), np.array(alfa_seq))
    else:
        return x


@jit(nopython=True)
def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-04, c2=0.9
):
    """Return boolean indicator whether upper wolfe condition are respected.

    :param f_old: Function old value
    :type f_old: float
    :param f_new: Function updated value
    :type f_new: float
    :param alpha: Step length
    :type alpha: float
    :param grad_f: Function gradient
    :type grad_f: numpy.ndarray
    :param p: [description]
    :type p: [type]
    :param c1: Tuning parameter, defaults to 1e-04
    :type c1: float, optional
    :param c2: Tuning parameter, defaults to 0.9
    :type c2: float, optional
    :return: Boolean indicator
    :rtype: bool
    """
    sup = f_old + c1 * alpha * np.dot(grad_f, p.T)
    return bool(f_new < sup)


@jit(nopython=True)
def linsearch_fun_DCM(xx, args):
    """Return alpha for the DBCM linsearch.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type xx: (numpy.ndarray, numpy.ndarray,
        float, float, function)
    :param args: Tuple, step function and arguments
    :type args: (function, tuple)
    :return: Working alpha
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    beta = xx[2]
    alfa = xx[3]
    f = xx[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    eps2 = 1e-2
    alfa0 = ((eps2 - 1) * x)[np.nonzero(dx)[0]] / dx[np.nonzero(dx)[0]]
    for a in alfa0:
        if a > 0:
            alfa = min(alfa, a)

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_DCM_fixed(xx):
    """Return alpha for the DBCM linsearch.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step
    :type xx: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        float, float, float)
    :return: Working alpha
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    dx_old = xx[2]
    alfa = xx[3]
    beta = xx[4]
    step = xx[5]

    eps2 = 1e-2
    alfa0 = ((eps2 - 1) * x)[np.nonzero(dx)[0]] / dx[np.nonzero(dx)[0]]
    for a in alfa0:
        if a > 0:
            alfa = min(alfa, a)

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(cond and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


@jit(nopython=True)
def linsearch_fun_crema(xx, args):
    """Return alpha for the CReM linsearch.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type xx: (numpy.ndarray, numpy.ndarray,
        float, float, function)
    :param args: Tuple, step function and arguments
    :type args: (function, tuple)
    :return: Working alpha
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    beta = xx[2]
    alfa = xx[3]
    f = xx[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    cond = sufficient_decrease_condition(s_old,
                                         -step_fun(x + alfa * dx,
                                                   arg_step_fun),
                                         alfa,
                                         f,
                                         dx)
    while (cond and i < 50):
        alfa *= beta
        i += 1
        cond = sufficient_decrease_condition(s_old,
                                             -step_fun(x + alfa * dx,
                                                       arg_step_fun),
                                             alfa,
                                             f,
                                             dx)
    return alfa


@jit(nopython=True)
def linsearch_fun_crema_fixed(xx):
    """Return alpha for the CReM linsearch.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step
    :type xx: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        float, float, float)
    :return: Working alpha
    :rtype: float
    """
    dx = xx[1]
    dx_old = xx[2]
    alfa = xx[3]
    beta = xx[4]
    step = xx[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(cond and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


@jit(nopython=True)
def linsearch_fun_DECM(xx, args):
    """Return alpha for the DECM linsearch.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type xx: (numpy.ndarray, numpy.ndarray,
        float, float, function)
    :param args: Tuple, step function and arguments
    :type args: (function, tuple)
    :return: Working alpha
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    beta = xx[2]
    alfa = xx[3]
    f = xx[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    eps2 = 1e-2
    alfa0 = ((eps2 - 1) * x)[np.nonzero(dx)[0]] / dx[np.nonzero(dx)[0]]
    for a in alfa0:
        if a > 0:
            alfa = min(alfa, a)

    # Mettere il check sulle y
    nnn = int(len(x) / 4)
    while True:
        ind_yout = np.argmax(x[2 * nnn: 3 * nnn] + alfa * dx[2 * nnn: 3 * nnn])
        ind_yin = np.argmax(x[3 * nnn:] + alfa * dx[3 * nnn:])
        cond = (x[2 * nnn: 3 * nnn][ind_yout]
                + alfa * dx[2 * nnn: 3 * nnn][ind_yout]) *\
               (x[3 * nnn][ind_yin]
                + alfa * dx[3 * nnn:][ind_yin])
        if (cond) < 1:
            break
        else:
            alfa *= beta

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_DECM_fixed(xx):
    """Return alpha for the DECM linsearch.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step
    :type xx: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        float, float, float)
    :return: Working alpha
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    dx_old = xx[2]
    alfa = xx[3]
    beta = xx[4]
    step = xx[5]

    eps2 = 1e-2
    alfa0 = ((eps2 - 1) * x)[np.nonzero(dx)[0]] / dx[np.nonzero(dx)[0]]
    for a in alfa0:
        if a > 0:
            alfa = min(alfa, a)

    # Mettere il check sulle y
    nnn = int(len(x) / 4)

    while True:
        ind_yout = np.argmax(x[2 * nnn: 3 * nnn] + alfa * dx[2 * nnn: 3 * nnn])
        ind_yin = np.argmax(x[3 * nnn:] + alfa * dx[3 * nnn:])
        cond = (x[2 * nnn: 3 * nnn][ind_yout]
                + alfa * dx[2 * nnn: 3 * nnn][ind_yout]) *\
               (x[3 * nnn:][ind_yin]
                + alfa * dx[3 * nnn:][ind_yin])
        if cond < 1:
            break
        else:
            alfa *= beta

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(cond and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


def edgelist_from_edgelist(edgelist):
    """Creates a new edgelists replacing nodes labels with indexes.
    Returns also two dictionaries that keep track of the
    nodes index-label relation.
    Works also on weighted graphs.

    :param edgelist: List of edges
    :type edgelist: list
    :return: Re-indexed list of edges, out-degrees, in-degrees,
        index to label dictionary
    :rtype: (dict, numpy.ndarray, numpy.ndarray, dict)
    """
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


class DirectedGraph:
    """Directed graph instance can be initialised with
    adjacency matrix, edgelist, degree sequence or strengths sequence.
    """
    def __init__(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
        """Initilizes all the necessary attribites for Directed graph class.

        :param adjacency: Adjacency matrix, defaults to None.
        :type adjacency: numpy.ndarray, list, scipy.sparse_matrix, optional
        :param edgelist: edgelist, defaults to None.
        :type edgelist: numpy.ndarray, list, optional
        :param degree_sequence: degrees sequence, defaults to None.
        :type degree_sequence: numpy.ndarray, optional
        :param strength_sequence: strengths sequence, defaults to None.
        :type strength_sequence: numpy.ndarray, optional
        """
        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.is_sparse = False
        self.edgelist = None
        self.dseq = None
        self.dseq_out = None
        self.dseq_in = None
        self.out_strength = None
        self.in_strength = None
        self.nz_index_sout = None
        self.nz_index_sin = None
        self.nodes_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_weighted = False
        self._initialize_graph(
            adjacency=adjacency,
            edgelist=edgelist,
            degree_sequence=degree_sequence,
            strength_sequence=strength_sequence,
        )
        self.avg_mat = None

        self.initial_guess = None
        # Reduced problem parameters
        self.is_reduced = False
        self.r_dseq = None
        self.r_dseq_out = None
        self.r_dseq_in = None
        self.r_n_out = None
        self.r_n_in = None
        self.r_invert_dseq = None
        self.r_invert_dseq_out = None
        self.r_invert_dseq_in = None
        self.r_dim = None
        self.r_multiplicity = None
        # Problem solutions
        self.x = None
        self.y = None
        self.xy = None
        self.b_out = None
        self.b_in = None
        # reduced solutions
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        # Problem (reduced) residuals
        self.residuals = None
        self.final_result = None
        self.r_beta = None
        # non-zero indices
        self.nz_index_out = None
        self.rnz_dseq_out = None
        self.nz_index_in = None
        self.rnz_dseq_in = None
        # model
        self.x0 = None
        self.error = None
        self.error_degree = None
        self.relative_error_degree = None
        self.error_strength = None
        self.relative_error_strength = None
        self.full_return = False
        self.last_model = None
        # functen
        self.args = None

    def _initialize_graph(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
        """[summary]

        :param adjacency: Adjacency matrix, defaults to None.
        :type adjacency: numpy.ndarray, list, scipy.sparse_matrix, optional
        :param edgelist: edgelist, defaults to None.
        :type edgelist: numpy.ndarray, list, optional
        :param degree_sequence: degrees sequence, defaults to None.
        :type degree_sequence: numpy.ndarray, optional
        :param strength_sequence: strengths sequence, defaults to None.
        :type strength_sequence: numpy.ndarray, optional
        :raises TypeError: [description]
        :raises TypeError: [description]
        :raises TypeError: [description]
        :raises ValueError: [description]
        :raises TypeError: [description]
        :raises TypeError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises TypeError: [description]
        :raises TypeError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises TypeError: [description]
        :raises TypeError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        """
        if adjacency is not None:
            if not isinstance(
                adjacency, (list, np.ndarray)
            ) and not scipy.sparse.isspmatrix(adjacency):
                raise TypeError(
                    ("The adjacency matrix must be passed as a list or numpy"
                     " array or scipy sparse matrix.")
                )
            elif adjacency.size > 0:
                if np.sum(adjacency < 0):
                    raise TypeError(
                        "The adjacency matrix entries must be positive."
                    )
                if isinstance(
                    adjacency, list
                ):
                    # Cast it to a numpy array: if it is given as a list
                    # it should not be too large
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.adjacency = adjacency
                    self.is_sparse = True
                if np.sum(adjacency) == np.sum(adjacency > 0):
                    self.dseq_in = in_degree(adjacency)
                    self.dseq_out = out_degree(adjacency)
                else:
                    self.dseq_in = in_degree(adjacency)
                    self.dseq_out = out_degree(adjacency)
                    self.in_strength = in_strength(adjacency).astype(
                        np.float64
                    )
                    self.out_strength = out_strength(adjacency).astype(
                        np.float64
                    )
                    self.nz_index_sout = np.nonzero(self.out_strength)[0]
                    self.nz_index_sin = np.nonzero(self.in_strength)[0]
                    self.is_weighted = True

                self.n_nodes = len(self.dseq_out)
                self.n_edges = np.sum(self.dseq_out)
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError(
                    "The edgelist must be passed as a list or numpy array."
                )
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        ("This is not an edgelist. An edgelist must be a list"
                         " or array of couples of nodes with optional weights."
                         " Is this an adjacency matrix?")
                    )
                elif len(edgelist[0]) == 2:
                    (
                        self.edgelist,
                        self.dseq_out,
                        self.dseq_in,
                        self.nodes_dict,
                    ) = edgelist_from_edgelist(edgelist)
                else:
                    (
                        self.edgelist,
                        self.dseq_out,
                        self.dseq_in,
                        self.out_strength,
                        self.in_strength,
                        self.nodes_dict,
                    ) = edgelist_from_edgelist(edgelist)
                self.n_nodes = len(self.dseq_out)
                self.n_edges = np.sum(self.dseq_out)
                self.is_initialized = True

        elif degree_sequence is not None:
            if not isinstance(degree_sequence, (list, np.ndarray)):
                raise TypeError(
                    ("The degree sequence must be passed as a list"
                     " or numpy array.")
                )
            elif len(degree_sequence) > 0:
                try:
                    int(degree_sequence[0])
                except:  # TODO: bare exception
                    raise TypeError(
                        "The degree sequence must contain numeric values."
                    )
                if (np.array(degree_sequence) < 0).sum() > 0:
                    raise ValueError("A degree cannot be negative.")
                else:
                    if len(degree_sequence) % 2 != 0:
                        raise ValueError(
                            "Strength-in/out arrays must have same length."
                        )
                    self.n_nodes = int(len(degree_sequence) / 2)
                    self.dseq_out = degree_sequence[: self.n_nodes]
                    self.dseq_in = degree_sequence[self.n_nodes:]
                    self.n_edges = np.sum(self.dseq_out)
                    self.is_initialized = True

                if strength_sequence is not None:
                    if not isinstance(strength_sequence, (list, np.ndarray)):
                        raise TypeError(
                            ("The strength sequence must be passed as a"
                             " list or numpy array.")
                        )
                    elif len(strength_sequence):
                        try:
                            int(strength_sequence[0])
                        except:  # TODO: bare exception to check
                            raise TypeError(
                                ("The strength sequence must contain"
                                 " numeric values.")
                            )
                        if (np.array(strength_sequence) < 0).sum() > 0:
                            raise ValueError("A strength cannot be negative.")
                        else:
                            if len(strength_sequence) % 2 != 0:
                                raise ValueError(
                                    ("Strength-in/out arrays must have"
                                     " same length.")
                                )
                            self.n_nodes = int(len(strength_sequence) / 2)
                            self.out_strength = strength_sequence[
                                : self.n_nodes
                            ]
                            self.in_strength = strength_sequence[
                                self.n_nodes:
                            ]
                            self.nz_index_sout = np.nonzero(self.out_strength)[
                                0
                            ]
                            self.nz_index_sin = np.nonzero(self.in_strength)[0]
                            self.is_weighted = True
                            self.is_initialized = True

        elif strength_sequence is not None:
            if not isinstance(strength_sequence, (list, np.ndarray)):
                raise TypeError(
                    ("The strength sequence must be passed as a list or"
                     " numpy array.")
                )
            elif len(strength_sequence):
                try:
                    int(strength_sequence[0])
                except:  # TODO: bare exception
                    raise TypeError(
                        "The strength sequence must contain numeric values."
                    )
                if (np.array(strength_sequence) < 0).sum() > 0:
                    raise ValueError("A strength cannot be negative.")
                else:
                    if len(strength_sequence) % 2 != 0:
                        raise ValueError(
                            "Strength-in/out arrays must have same length."
                        )
                    self.n_nodes = int(len(strength_sequence) / 2)
                    self.out_strength = strength_sequence[: self.n_nodes]
                    self.in_strength = strength_sequence[self.n_nodes:]
                    self.nz_index_sout = np.nonzero(self.out_strength)[0]
                    self.nz_index_sin = np.nonzero(self.in_strength)[0]
                    self.is_weighted = True
                    self.is_initialized = True

    def set_adjacency_matrix(self, adjacency):
        """Initializes a graph from the adjacency matrix.

        :param adjacency: Adjacency matrix
        :type adjacency: numpy.ndarray, list, scipy.sparse_matrix
        """
        if self.is_initialized:
            print(
                ("Graph already contains edges or has a degree sequence."
                 " Use clean_edges() first.")
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        """Initializes a graph from the edgelist.

        :param adjacency: Edgelist
        :type adjacency: numpy.ndarray, list
        """
        if self.is_initialized:
            print(
                ("Graph already contains edges or has a degree sequence."
                 " Use clean_edges() first.")
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequence):
        """Initializes a graph from the degrees sequence.

        :param adjacency: Degrees sequence
        :type adjacency: numpy.ndarray
        """
        if self.is_initialized:
            print(
                ("Graph already contains edges or has a degree sequence."
                 " Use clean_edges() first.")
            )
        else:
            self._initialize_graph(degree_sequence=degree_sequence)

    def clean_edges(self):
        """Deletes all initialized graph attributes
        """
        self.adjacency = None
        self.edgelist = None
        self.deg_seq = None
        self.is_initialized = False

    def _solve_problem(
        self,
        initial_guess=None,  # TODO:aggiungere un default a initial guess
        model="dcm",
        method="quasinewton",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
        regularise_eps=1e-3,
    ):

        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self.regularise = regularise
        self._initialize_problem(model, method)
        x0 = self.x0

        sol = solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=self.regularise,
            regularise_eps=regularise_eps,
            full_return=full_return,
            linsearch=linsearch,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_dcm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        self.r_x = self.r_xy[: self.rnz_n_out]
        self.r_y = self.r_xy[self.rnz_n_out:]

        self.x = self.r_x[self.r_invert_dseq]
        self.y = self.r_y[self.r_invert_dseq]

    def _set_solved_problem_dcm_new(self, solution):
        if self.full_return:
            # conversion from theta to x
            self.r_xy = np.exp(-solution[0])
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            # conversion from theta to x
            self.r_xy = np.exp(-solution)

        self.r_x = self.r_xy[: self.rnz_n_out]
        self.r_y = self.r_xy[self.rnz_n_out:]

        self.x = self.r_x[self.r_invert_dseq]
        self.y = self.r_y[self.r_invert_dseq]

    def _set_solved_problem_decm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        self.x = self.r_xy[: self.n_nodes]
        self.y = self.r_xy[self.n_nodes: 2 * self.n_nodes]
        self.b_out = self.r_xy[2 * self.n_nodes: 3 * self.n_nodes]
        self.b_in = self.r_xy[3 * self.n_nodes:]

    def _set_solved_problem_decm_new(self, solution):
        if self.full_return:
            # conversion from theta to x
            self.r_xy = np.exp(-solution[0])
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            # conversion from theta to x
            self.r_xy = np.exp(-solution)

        self.x = self.r_xy[: self.n_nodes]
        self.y = self.r_xy[self.n_nodes: 2 * self.n_nodes]
        self.b_out = self.r_xy[2 * self.n_nodes: 3 * self.n_nodes]
        self.b_in = self.r_xy[3 * self.n_nodes:]

    def _set_solved_problem(self, solution):
        model = self.last_model
        if model in ["dcm"]:
            self._set_solved_problem_dcm(solution)
        if model in ["dcm_new"]:
            self._set_solved_problem_dcm_new(solution)
        elif model in ["decm"]:
            self._set_solved_problem_decm(solution)
        elif model in ["decm_new"]:
            self._set_solved_problem_decm_new(solution)
        elif model in ["crema", "crema-sparse"]:
            self._set_solved_problem_crema(solution)

    def degree_reduction(self):
        """Carries out degree reduction for DBCM.
        The graph should be initialized.
        """
        self.dseq = np.array(list(zip(self.dseq_out, self.dseq_in)))
        (
            self.r_dseq,
            self.r_index_dseq,
            self.r_invert_dseq,
            self.r_multiplicity
        ) = np.unique(
            self.dseq,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )

        self.rnz_dseq_out = self.r_dseq[:, 0]
        self.rnz_dseq_in = self.r_dseq[:, 1]

        self.nz_index_out = np.nonzero(self.rnz_dseq_out)[0]
        self.nz_index_in = np.nonzero(self.rnz_dseq_in)[0]

        self.rnz_n_out = self.rnz_dseq_out.size
        self.rnz_n_in = self.rnz_dseq_in.size
        self.rnz_dim = self.rnz_n_out + self.rnz_n_in

        self.is_reduced = True

    def _set_initial_guess(self, model):
        if model in ["dcm"]:
            self._set_initial_guess_dcm()
        if model in ["dcm_new"]:
            self._set_initial_guess_dcm_new()
        elif model in ["decm"]:
            self._set_initial_guess_decm()
        elif model in ["decm_new"]:
            self._set_initial_guess_decm_new()
        elif model in ["crema", "crema-sparse"]:
            self._set_initial_guess_crema()

    def _set_initial_guess_dcm(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess,
        # remember that the code starts with a reduced number
        # of rows and columns.
        # remember if you insert your choice as initial choice,
        # it should be numpy.ndarray
        if ~self.is_reduced:
            self.degree_reduction()

        if isinstance(self.initial_guess, np.ndarray):
            # we reduce the full x0, it's not very honest
            # but it's better to ask to provide an already reduced x0
            self.r_x = self.initial_guess[:self.n_nodes][self.r_index_dseq]
            self.r_y = self.initial_guess[self.n_nodes:][self.r_index_dseq]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == 'degrees_minor':
                # This +1 increases the stability of the solutions.
                self.r_x = self.rnz_dseq_out / (np.sqrt(self.n_edges) + 1)
                self.r_y = self.rnz_dseq_in / (np.sqrt(self.n_edges) + 1)
            elif self.initial_guess == "random":
                self.r_x = np.random.rand(self.rnz_n_out).astype(np.float64)
                self.r_y = np.random.rand(self.rnz_n_in).astype(np.float64)
            elif self.initial_guess == "uniform":
                # All probabilities will be 1/2 initially
                self.r_x = 0.5 * np.ones(self.rnz_n_out, dtype=np.float64)
                self.r_y = 0.5 * np.ones(self.rnz_n_in, dtype=np.float64)
            elif self.initial_guess == "degrees":
                self.r_x = self.rnz_dseq_out.astype(np.float64)
                self.r_y = self.rnz_dseq_in.astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.r_x = self.rnz_dseq_out.astype(np.float64) / \
                    (2*self.n_edges)
                self.r_y = self.rnz_dseq_in.astype(np.float64)/(2*self.n_edges)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.r_x[self.rnz_dseq_out == 0] = 0
        self.r_y[self.rnz_dseq_in == 0] = 0

        self.x0 = np.concatenate((self.r_x, self.r_y))

    def _set_initial_guess_dcm_new(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code
        # starts with a reduced number of rows and columns.

        if ~self.is_reduced:
            self.degree_reduction()

        if isinstance(self.initial_guess, np.ndarray):
            # we reduce the full x0, it's not very honest
            # but it's better to ask to provide an already reduced x0
            self.r_x = self.initial_guess[:self.n_nodes][self.r_index_dseq]
            self.r_y = self.initial_guess[self.n_nodes:][self.r_index_dseq]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == 'degrees_minor':
                self.r_x = self.rnz_dseq_out / (
                    np.sqrt(self.n_edges) + 1
                )  # This +1 increases the stability of the solutions.
                self.r_y = self.rnz_dseq_in / (np.sqrt(self.n_edges) + 1)
            elif self.initial_guess == "random":
                self.r_x = np.random.rand(self.rnz_n_out).astype(np.float64)
                self.r_y = np.random.rand(self.rnz_n_in).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.r_x = 0.5 * np.ones(
                    self.rnz_n_out, dtype=np.float64
                )  # All probabilities will be 1/2 initially
                self.r_y = 0.5 * np.ones(self.rnz_n_in, dtype=np.float64)
            elif self.initial_guess == "degrees":
                self.r_x = self.rnz_dseq_out.astype(np.float64)
                self.r_y = self.rnz_dseq_in.astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.r_x = self.rnz_dseq_out.astype(np.float64) / \
                    (2*self.n_edges)
                self.r_y = self.rnz_dseq_in.astype(np.float64) / \
                    (2*self.n_edges)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        not_zero_ind_x = self.r_x != 0
        self.r_x[not_zero_ind_x] = -np.log(self.r_x[not_zero_ind_x])
        self.r_x[self.rnz_dseq_out == 0] = 1e3
        not_zero_ind_y = self.r_y != 0
        self.r_y[not_zero_ind_y] = -np.log(self.r_y[not_zero_ind_y])
        self.r_y[self.rnz_dseq_in == 0] = 1e3

        self.x0 = np.concatenate((self.r_x, self.r_y))

    def _set_initial_guess_crema(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that
        # the code starts with a reduced number of rows and columns.
        # TODO: mettere un self.is_weighted bool
        if isinstance(self.initial_guess, np.ndarray):
            self.b_out = self.initial_guess[:self.n_nodes]
            self.b_in = self.initial_guess[self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.b_out = (self.out_strength > 0).astype(
                    float
                ) / self.out_strength.sum()
                self.b_in = (self.in_strength > 0).astype(
                    float
                ) / self.in_strength.sum()
            elif self.initial_guess == "strengths_minor":
                # This +1 increases the stability of the solutions.
                self.b_out = (self.out_strength > 0).astype(float) / (
                    self.out_strength + 1
                )
                self.b_in = (self.in_strength > 0).astype(float) / (
                    self.in_strength + 1
                )
            elif self.initial_guess == "random":
                self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.b_out[self.out_strength == 0] = 0
        self.b_in[self.in_strength == 0] = 0

        self.x0 = np.concatenate((self.b_out, self.b_in))

    def _set_initial_guess_decm(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess,
        # remember that the code starts with a reduced number
        # of rows and columns.
        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes]
            self.y = self.initial_guess[self.n_nodes:2*self.n_nodes]
            self.b_out = self.initial_guess[2*self.n_nodes:3*self.n_nodes]
            self.b_in = self.initial_guess[3*self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == 'strengths':
                self.x = self.dseq_out.astype(float) / (self.n_edges + 1)
                self.y = self.dseq_in.astype(float) / (self.n_edges + 1)
                self.b_out = (
                    self.out_strength.astype(float) / self.out_strength.sum()
                )  # This +1 increases the stability of the solutions.
                self.b_in = self.in_strength.astype(float) /\
                    self.in_strength.sum()
            elif self.initial_guess == "strengths_minor":
                self.x = np.ones_like(self.dseq_out) / (self.dseq_out + 1)
                self.y = np.ones_like(self.dseq_in) / (self.dseq_in + 1)
                self.b_out = np.ones_like(self.out_strength) / (
                    self.out_strength + 1
                )
                self.b_in = np.ones_like(self.in_strength) /\
                    (self.in_strength + 1)
            elif self.initial_guess == "random":
                self.x = np.random.rand(self.n_nodes).astype(np.float64)
                self.y = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
            elif self.initial_guess == "uniform":
                # All probabilities will be 0.9 initially
                self.x = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
                self.y = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_out = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_in = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.x[self.dseq_out == 0] = 0
        self.y[self.dseq_in == 0] = 0
        self.b_out[self.out_strength == 0] = 0
        self.b_in[self.in_strength == 0] = 0

        self.x0 = np.concatenate((self.x, self.y, self.b_out, self.b_in))

    def _set_initial_guess_decm_new(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        #  trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that
        # the code starts with a reduced number of rows and columns.
        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes]
            self.y = self.initial_guess[self.n_nodes:2*self.n_nodes]
            self.b_out = self.initial_guess[2*self.n_nodes:3*self.n_nodes]
            self.b_in = self.initial_guess[3*self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.x = self.dseq_out.astype(float) / (self.n_edges + 1)
                self.y = self.dseq_in.astype(float) / (self.n_edges + 1)
                self.b_out = (
                    self.out_strength.astype(float) / self.out_strength.sum()
                )  # This +1 increases the stability of the solutions.
                self.b_in = self.in_strength.astype(float) /\
                    self.in_strength.sum()
            elif self.initial_guess == "strengths_minor":
                self.x = np.ones_like(self.dseq_out) / (self.dseq_out + 1)
                self.y = np.ones_like(self.dseq_in) / (self.dseq_in + 1)
                self.b_out = np.ones_like(self.out_strength) / (
                    self.out_strength + 1
                )
                self.b_in = np.ones_like(self.in_strength) /\
                    (self.in_strength + 1)
            elif self.initial_guess == "random":
                self.x = np.random.rand(self.n_nodes).astype(np.float64)
                self.y = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.x = 0.1 * np.ones(
                    self.n_nodes, dtype=np.float64
                )  # All probabilities will be 1/2 initially
                self.y = 0.1 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_out = 0.1 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_in = 0.1 * np.ones(self.n_nodes, dtype=np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        not_zero_ind_x = self.x != 0
        self.x[not_zero_ind_x] = -np.log(self.x[not_zero_ind_x])

        not_zero_ind_y = self.y != 0
        self.y[not_zero_ind_y] = -np.log(self.y[not_zero_ind_y])

        not_zero_ind_b_out = self.b_out != 0
        self.b_out[not_zero_ind_b_out] = -np.log(
            self.b_out[not_zero_ind_b_out])

        not_zero_ind_b_in = self.b_in != 0
        self.b_in[not_zero_ind_b_in] = -np.log(self.b_in[not_zero_ind_b_in])

        self.x[self.dseq_out == 0] = 1e3
        self.y[self.dseq_in == 0] = 1e3
        self.b_out[self.out_strength == 0] = 1e3
        self.b_in[self.in_strength == 0] = 1e3

        self.x0 = np.concatenate((self.x, self.y, self.b_out, self.b_in))

    def solution_error(self):
        """Computes the error given the solutions of the optimization problem
        """
        if self.last_model in ["dcm_new", "dcm", "crema", "crema-sparse"]:
            if (self.x is not None) and (self.y is not None):
                sol = np.concatenate((self.x, self.y))
                ex_k_out = expected_out_degree_dcm(sol)
                ex_k_in = expected_in_degree_dcm(sol)
                ex_k = np.concatenate((ex_k_out, ex_k_in))
                k = np.concatenate((self.dseq_out, self.dseq_in))
                # print(k, ex_k)
                self.expected_dseq = ex_k
                # error output
                self.error_degree = np.linalg.norm(ex_k - k, ord=np.inf)
                self.relative_error_degree = np.linalg.norm(
                    (ex_k - k) / (k + + np.exp(-100)),
                    ord=np.inf
                    )
                self.error = self.error_degree

            if (self.b_out is not None) and (self.b_in is not None):
                sol = np.concatenate([self.b_out, self.b_in])
                if self.is_sparse:
                    ex_s_out = expected_out_strength_crema_sparse(
                        sol, self.adjacency_crema
                    )
                    ex_s_in = expected_in_stregth_crema_sparse(
                        sol, self.adjacency_crema
                    )
                else:
                    ex_s_out = expected_out_strength_crema(
                        sol, self.adjacency_crema
                    )
                    ex_s_in = expected_in_stregth_crema(
                        sol, self.adjacency_crema
                    )
                ex_s = np.concatenate([ex_s_out, ex_s_in])
                s = np.concatenate([self.out_strength, self.in_strength])
                self.expected_stregth_seq = ex_s
                # error output
                self.error_strength = np.linalg.norm(ex_s - s, ord=np.inf)
                self.relative_error_strength = np.max(
                    abs(
                        (ex_s - s) / (s + np.exp(-100))
                    )
                )
                if self.adjacency_given:
                    self.error = self.error_strength
                else:
                    self.error = max(self.error_strength, self.error_degree)

        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ["decm", "decm_new"]:
            sol = np.concatenate((self.x, self.y, self.b_out, self.b_in))
            ex = expected_decm(sol)
            k = np.concatenate(
                (
                    self.dseq_out,
                    self.dseq_in,
                    self.out_strength,
                    self.in_strength,
                )
            )
            self.expected_dseq = ex[: 2 * self.n_nodes]

            self.expected_strength_seq = ex[2 * self.n_nodes:]

            # error putput
            self.error_degree = max(
                abs(
                    (
                        np.concatenate((self.dseq_out, self.dseq_in))
                        - self.expected_dseq
                    )
                )
            )
            self.error_strength = max(
                abs(
                    np.concatenate((self.out_strength, self.in_strength))
                    - self.expected_strength_seq
                )
            )
            self.relative_error_strength = max(
                abs(
                    (np.concatenate((self.out_strength, self.in_strength))
                     - self.expected_strength_seq)
                    / np.concatenate((self.out_strength, self.in_strength)
                                     + np.exp(-100))
                )
            )
            self.relative_error_degree = max(
                abs(
                    (np.concatenate((self.dseq_out, self.dseq_in))
                     - self.expected_dseq)
                    / np.concatenate((self.dseq_out, self.dseq_in)
                                     + np.exp(-100))
                 )
            )
            self.error = np.linalg.norm(ex - k, ord=np.inf)

    def _set_args(self, model):
        if model in ["crema", "crema-sparse"]:
            self.args = (
                self.out_strength,
                self.in_strength,
                self.adjacency_crema,
                self.nz_index_sout,
                self.nz_index_sin,
            )
        elif model in ["dcm", "dcm_new"]:
            self.args = (
                self.rnz_dseq_out,
                self.rnz_dseq_in,
                self.nz_index_out,
                self.nz_index_in,
                self.r_multiplicity,
            )
        elif model in ["decm", "decm_new"]:
            self.args = (
                self.dseq_out,
                self.dseq_in,
                self.out_strength,
                self.in_strength,
            )

    def _initialize_problem(self, model, method):

        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = "-"
        mod_met = mod_met.join([model, method])

        d_fun = {
            "dcm-newton": lambda x: -loglikelihood_prime_dcm(x, self.args),
            "dcm-quasinewton": lambda x: -loglikelihood_prime_dcm(
                x,
                self.args
            ),
            "dcm-fixed-point": lambda x: iterative_dcm(x, self.args),
            "dcm_new-newton": lambda x: -loglikelihood_prime_dcm_new(
                x,
                self.args
            ),
            "dcm_new-quasinewton": lambda x: -loglikelihood_prime_dcm_new(
                x,
                self.args
            ),
            "dcm_new-fixed-point": lambda x: iterative_dcm_new(x, self.args),
            "crema-newton": lambda x: -loglikelihood_prime_crema(
                x,
                self.args
            ),
            "crema-quasinewton": lambda x: -loglikelihood_prime_crema(
                x,
                self.args
            ),
            "crema-fixed-point": lambda x: -iterative_crema(x, self.args),
            "decm-newton": lambda x: -loglikelihood_prime_decm(x, self.args),
            "decm-quasinewton": lambda x: -loglikelihood_prime_decm(
                x,
                self.args
            ),
            "decm-fixed-point": lambda x: iterative_decm(x, self.args),
            "decm_new-newton": lambda x: -loglikelihood_prime_decm_new(
                x,
                self.args
            ),
            "decm_new-quasinewton": lambda x: -loglikelihood_prime_decm_new(
                x,
                self.args
            ),
            "decm_new-fixed-point": lambda x: iterative_decm_new(x, self.args),
            "crema-sparse-newton": lambda x: -loglikelihood_prime_crema_sparse(
                x,
                self.args
            ),
            "crema-sparse-quasinewton": lambda x:
                -loglikelihood_prime_crema_sparse(
                    x,
                    self.args
                ),
            "crema-sparse-fixed-point": lambda x: -iterative_crema_sparse(
                x,
                self.args
            ),
        }

        d_fun_jac = {
            "dcm-newton": lambda x: -loglikelihood_hessian_dcm(x, self.args),
            "dcm-quasinewton": lambda x: -loglikelihood_hessian_diag_dcm(
                x,
                self.args
            ),
            "dcm-fixed-point": None,
            "dcm_new-newton": lambda x: -loglikelihood_hessian_dcm_new(
                x,
                self.args
            ),
            "dcm_new-quasinewton": lambda x:
                -loglikelihood_hessian_diag_dcm_new(
                    x,
                    self.args
                ),
            "dcm_new-fixed-point": None,
            "crema-newton": lambda x: -loglikelihood_hessian_crema(
                x,
                self.args
            ),
            "crema-quasinewton": lambda x: -loglikelihood_hessian_diag_crema(
                x,
                self.args
            ),
            "crema-fixed-point": None,
            "decm-newton": lambda x: -loglikelihood_hessian_decm(x, self.args),
            "decm-quasinewton": lambda x: -loglikelihood_hessian_diag_decm(
                x,
                self.args
            ),
            "decm-fixed-point": None,
            "decm_new-newton": lambda x: -loglikelihood_hessian_decm_new(
                x,
                self.args
            ),
            "decm_new-quasinewton": lambda x:
                -loglikelihood_hessian_diag_decm_new(
                    x,
                    self.args
                ),
            "decm_new-fixed-point": None,
            "crema-sparse-newton": lambda x: -loglikelihood_hessian_crema(
                x,
                self.args
            ),
            "crema-sparse-quasinewton": lambda x:
                -loglikelihood_hessian_diag_crema_sparse(
                    x,
                    self.args
                ),
            "crema-sparse-fixed-point": None,
        }

        d_fun_step = {
            "dcm-newton": lambda x: -loglikelihood_dcm(x, self.args),
            "dcm-quasinewton": lambda x: -loglikelihood_dcm(x, self.args),
            "dcm-fixed-point": lambda x: -loglikelihood_dcm(x, self.args),
            "dcm_new-newton": lambda x: -loglikelihood_dcm_new(x, self.args),
            "dcm_new-quasinewton": lambda x: -loglikelihood_dcm_new(
                x,
                self.args
            ),
            "dcm_new-fixed-point": lambda x: -loglikelihood_dcm_new(
                x,
                self.args
            ),
            "crema-newton": lambda x: -loglikelihood_crema(x, self.args),
            "crema-quasinewton": lambda x: -loglikelihood_crema(
                x,
                self.args
            ),
            "crema-fixed-point": lambda x: -loglikelihood_crema(
                x,
                self.args
            ),
            "decm-newton": lambda x: -loglikelihood_decm(x, self.args),
            "decm-quasinewton": lambda x: -loglikelihood_decm(x, self.args),
            "decm-fixed-point": lambda x: -loglikelihood_decm(x, self.args),
            "decm_new-newton": lambda x: -loglikelihood_decm_new(x, self.args),
            "decm_new-quasinewton": lambda x: -loglikelihood_decm_new(
                x,
                self.args
            ),
            "decm_new-fixed-point": lambda x: -loglikelihood_decm_new(
                x,
                self.args
            ),
            "crema-sparse-newton": lambda x: -loglikelihood_crema_sparse(
                x,
                self.args
            ),
            "crema-sparse-quasinewton": lambda x: -loglikelihood_crema_sparse(
                x,
                self.args
            ),
            "crema-sparse-fixed-point": lambda x: -loglikelihood_crema_sparse(
                x,
                self.args
            ),
        }

        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.step_fun = d_fun_step[mod_met]
        except:  # TODO: remove bare excpets
            raise ValueError(
                'Method must be "newton","quasinewton", or "fixed-point".'
            )

        # TODO: mancano metodi
        d_pmatrix = {"dcm": pmatrix_dcm, "dcm_new": pmatrix_dcm}

        # Così basta aggiungere il decm e funziona tutto
        if model in ["dcm", "dcm_new"]:
            self.args_p = (
                self.n_nodes,
                np.nonzero(self.dseq_out)[0],
                np.nonzero(self.dseq_in)[0],
            )
            self.fun_pmatrix = lambda x: d_pmatrix[model](x, self.args_p)

        args_lin = {
            "dcm": (loglikelihood_dcm, self.args),
            "crema": (loglikelihood_crema, self.args),
            "crema-sparse": (loglikelihood_crema_sparse, self.args),
            "decm": (loglikelihood_decm, self.args),
            "dcm_new": (loglikelihood_dcm_new, self.args),
            "decm_new": (loglikelihood_decm_new, self.args),
        }

        self.args_lins = args_lin[model]

        lins_fun = {
            "dcm-newton": lambda x: linsearch_fun_DCM(x, self.args_lins),
            "dcm-quasinewton": lambda x: linsearch_fun_DCM(x, self.args_lins),
            "dcm-fixed-point": lambda x: linsearch_fun_DCM_fixed(x),
            "dcm_new-newton": lambda x: linsearch_fun_DCM_new(
                x,
                self.args_lins),
            "dcm_new-quasinewton": lambda x: linsearch_fun_DCM_new(
                x,
                self.args_lins),
            "dcm_new-fixed-point": lambda x: linsearch_fun_DCM_new_fixed(x),
            "crema-newton": lambda x: linsearch_fun_crema(x, self.args_lins),
            "crema-quasinewton": lambda x: linsearch_fun_crema(
                x,
                self.args_lins),
            "crema-fixed-point": lambda x: linsearch_fun_crema_fixed(x),
            "crema-sparse-newton": lambda x: linsearch_fun_crema(
                x,
                self.args_lins),
            "crema-sparse-quasinewton": lambda x: linsearch_fun_crema(
                x,
                self.args_lins),
            "crema-sparse-fixed-point": lambda x: linsearch_fun_crema_fixed(
                x),
            "decm-newton": lambda x: linsearch_fun_DECM(
                x,
                self.args_lins),
            "decm-quasinewton": lambda x: linsearch_fun_DECM(
                x,
                self.args_lins),
            "decm-fixed-point": lambda x: linsearch_fun_DECM_fixed(x),
            "decm_new-newton": lambda x: linsearch_fun_DECM_new(
                x,
                self.args_lins),
            "decm_new-quasinewton": lambda x: linsearch_fun_DECM_new(
                x,
                self.args_lins),
            "decm_new-fixed-point": lambda x: linsearch_fun_DECM_new_fixed(x),
        }

        self.fun_linsearch = lins_fun[mod_met]

        hess_reg = {
            "dcm": hessian_regulariser_function_eigen_based,
            "dcm_new": hessian_regulariser_function,
            "decm": hessian_regulariser_function_eigen_based,
            "decm_new": hessian_regulariser_function,
            "crema": hessian_regulariser_function,
            "crema-sparse": hessian_regulariser_function,
        }

        self.hessian_regulariser = hess_reg[model]

        if isinstance(self.regularise, str):
            if self.regularise == "eigenvalues":
                self.hessian_regulariser = \
                    hessian_regulariser_function_eigen_based
            elif self.regularise == "identity":
                self.hessian_regulariser = hessian_regulariser_function

    def _solve_problem_crema(
        self,
        initial_guess=None,
        model="crema",
        adjacency="dcm",
        method="quasinewton",
        method_adjacency="newton",
        initial_guess_adjacency="random",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
        regularise_eps=1e-3,
    ):
        if model == "crema-sparse":
            self.is_sparse = True
        else:
            self.is_sparse = False

        if not isinstance(adjacency, (list, np.ndarray, str)) and (
            not scipy.sparse.isspmatrix(adjacency)
        ):
            raise ValueError("adjacency must be a matrix or a method")
        elif isinstance(adjacency, str):
            self._solve_problem(
                initial_guess=initial_guess_adjacency,
                model=adjacency,
                method=method_adjacency,
                max_steps=max_steps,
                tol=tol,
                eps=eps,
                full_return=full_return,
                verbose=verbose,
            )
            if self.is_sparse:
                self.adjacency_crema = (self.x, self.y)
                self.adjacency_given = False
            else:
                pmatrix = self.fun_pmatrix(np.concatenate([self.x, self.y]))
                raw_ind, col_ind = np.nonzero(pmatrix)
                raw_ind = raw_ind.astype(np.int64)
                col_ind = col_ind.astype(np.int64)
                weigths_value = pmatrix[raw_ind, col_ind]
                self.adjacency_crema = (raw_ind, col_ind, weigths_value)
                self.is_sparse = False
                self.adjacency_given = False
        elif isinstance(adjacency, list):
            adjacency = np.array(adjacency).astype(np.float64)
            raw_ind, col_ind = np.nonzero(adjacency)
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif isinstance(adjacency, np.ndarray):
            adjacency = adjacency.astype(np.float64)
            raw_ind, col_ind = np.nonzero(adjacency)
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif scipy.sparse.isspmatrix(adjacency):
            raw_ind, col_ind = adjacency.nonzero()
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = (adjacency[raw_ind, col_ind].A1).astype(np.float64)
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True

        if self.is_sparse:
            self.last_model = "crema-sparse"
        else:
            self.last_model = model
 
        self.regularise = regularise
        self.full_return = full_return
        self.initial_guess = initial_guess
        self._initialize_problem(self.last_model, method)
        x0 = self.x0

        sol = solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=regularise,
            regularise_eps=regularise_eps,
            linsearch=linsearch,
            full_return=full_return,
        )

        self._set_solved_problem_crema(sol)

    def _set_solved_problem_crema(self, solution):
        if self.full_return:
            self.b_out = solution[0][: self.n_nodes]
            self.b_in = solution[0][self.n_nodes:]
            self.comput_time_crema = solution[1]
            self.n_steps_crema = solution[2]
            self.norm_seq_crema = solution[3]
            self.diff_seq_crema = solution[4]
            self.alfa_seq_crema = solution[5]
        else:
            self.b_out = solution[: self.n_nodes]
            self.b_in = solution[self.n_nodes:]

    def solve_tool(
        self,
        model,
        method,
        initial_guess=None,
        adjacency=None,
        method_adjacency='newton',
        initial_guess_adjacency="random",
        max_steps=100,
        full_return=False,
        verbose=False,
        linsearch=True,
        tol=1e-8,
        eps=1e-8,
    ):
        """The function solves the ERGM optimization problem.

        :param model: Available models are:
            - *dcm*: solves DBCM respect to the parameters *x* and "y" of the loglikelihood function, it works for uweighted directed graphs [insert ref].
            - *dcm-new*: differently from the *dcm* option, *dcm-new* considers the exponents of *x* and *y* as parameters [insert ref].
            - *decm*: solves DECM respect to the parameters *a_out*, *a_in*, *b_out* and *b_in* of the loglikelihood function, it is conceived for weighted directed graphs [insert ref].
            - *decm-new*: differently from the *ecm* option, *ecm-new* considers the exponents of *a_out*, *a_in*, *b_out* and *b_in** as parameters [insert ref].
            - *crema*: solves CReMa for a weighted directd graphs. In order to compute beta parameters, it requires information about the binary structure of the network. These can be provided by the user by using *adjacency* paramenter.
            - *crema-sparse*: alternative implementetio of *crema* for large graphs. The *creama-sparse* model doesn't compute the binary probability matrix avoing memory problems for large graphs.
        :type model: str
        :param method: Available methods to solve the given *model* are:
            - *newton*: uses Newton-Rhapson method to solve the selected model, it can be memory demanding for *crema* because it requires the computation of the entire Hessian matrix. This method is not available for *creama-sparse*.
            - *quasinewton*: uses Newton-Rhapson method with Hessian matrix approximated by its principal diagonal to find parameters maximising loglikelihood function.
            - *fixed-point*: uses a fixed-point method to find parameters maximising loglikelihood function.
        :type method: str
        :param initial_guess: Starting point solution may affect the results of the optization process. The user can provid an initial guess or choose between the following options:
            - **Binary Models**:
                - *random*: random numbers in (0, 1);
                - *uniform*: uniform initial guess in (0, 1);
                - *degrees*: initial guess of each node is proportianal to its degree;
                - *degrees_minor*: initial guess of each node is inversely proportional to its degree;
                - *chung_lu*: initial guess given by Chung-Lu formula;
            - **Weighted Models**:
                - *random*: random numbers in (0, 1);
                - *uniform*: uniform initial guess in (0, 1);
                - *strengths*: initial guess of each node is proportianal to its stength;
                - *strengths_minor*: initial guess of each node is inversely proportional to its strength;
        :type initial_guess: str, optional
        :param adjacency: Adjacency can be a binary method (defaults is *dcm-new*) or an adjacency matrix.
        :type adjacency: str or numpy.ndarray, optional
        :param method_adjacency: If adjacency is a *model*, it is the *methdod* used to solve it. Defaults to "newton".
        :type method_adjacency: str, optional
        :param initial_guess_adjacency: If adjacency is a *model*, it is the chosen initial guess. Defaults to "random".
        :type initial_guess_adjacency: str, optional
        :param max_steps: maximum number of iteration, defaults to 100.
        :type max_steps: int, optional
        :param full_return: If True the algorithm returns more statistics than the obtained solution, defaults to False.
        :type full_return: bool, optional
        :param verbose: If True the algorithm prints a bunch of statistics at each step, defaults to False.
        :type verbose: bool, optional
        :param linsearch: If True the linsearch function is active, defaults to True.
        :type linsearch: bool, optional
        :param tol: parameter controlling the tollerance of the norm the gradient function, defaults to 1e-8.
        :type tol: float, optional
        :param eps: parameter controlling the tollerance of the difference between two iterations, defaults to 1e-8.
        :type eps: float, optional
        """
        # TODO: aggiungere tutti i metodi
        if model in ["dcm", "dcm_new", "decm", "decm_new"]:
            self._solve_problem(
                initial_guess=initial_guess,
                model=model,
                method=method,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )
        elif model in ["crema", 'crema-sparse']:
            self._solve_problem_crema(
                initial_guess=initial_guess,
                model=model,
                adjacency=adjacency,
                method=method,
                method_adjacency = method_adjacency,
                initial_guess_adjacency = initial_guess_adjacency,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )


    def ensemble_sampler(self, n, cpu_n=2, output_dir="sample/", seed=10):
        """The function sample a given number of graphs in the ensemble generated from the last model solved.
        Each grpah is written as an edgelist in a `.txt` file in the output directory.
        The function is parallelised and can run on multiple cpus.

        :param n: Number of graphs to sampl
        :type n: int
        :param cpu_n: Number of cpus to use, defaults to 2
        :type cpu_n: int, optional
        :param output_dir: Name of the output directory, defaults to "sample/"
        :type output_dir: str, optional
        :param seed: Random seed, defaults to 10
        :type seed: int, optional
        :raises ValueError: [description]
        """
        # al momento funziona solo sull'ultimo problema risolto
        # unico input possibile e' la cartella dove salvare i samples
        # ed il numero di samples

        # create the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # compute the sample

        # seed specification
        np.random.seed(seed)
        s = [np.random.randint(0, 1000000) for i in range(n)]

        if self.last_model in ["dcm", "dcm_new"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_dcm_graph(
                    outfile_name=item,
                    x=self.x,
                    y=self.y,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["decm", "decm_new"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_decm_graph(
                    outfile_name=item,
                    a_out=self.x,
                    a_in=self.y,
                    b_out=self.b_out,
                    b_in=self.b_in,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["crema"]:
            if self.adjacency_given:
                # deterministic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_decm_det_graph(
                        outfile_name=item,
                        beta=(self.b_out, self.b_in),
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
            else:
                # probabilistic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_decm_prob_graph(
                        outfile_name=item,
                        beta=(self.b_out, self.b_in),
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
        elif self.last_model in ["crema-sparse"]:
            if not self.adjacency_given:
                # probabilistic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_sparse_decm_prob_graph(
                        outfile_name=item,
                        beta=(self.b_out, self.b_in),
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
        else:
            raise ValueError("insert a model")
