import numpy as np
import os
import scipy.sparse
from numba import jit, prange
import time
from netrecon.Undirected_new import *

from . import ensemble_generator as eg
# Stops Numba Warning for experimental feature
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings

warnings.simplefilter(action='ignore',
                      category=NumbaExperimentalFeatureWarning)


def degree(a):
    """Returns symmetric matrix *a* degrees sequence.

    :param a: matrix a.
    :type a: numpy.ndarray, scipy.sparse
    :return: degree sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1


def strength(a):
    """Returns symmetric matrix *a* strengths sequence.

    :param a: matrix a.
    :type a: numpy.ndarray, scipy.sparse
    :return: strengths sequence.
    :rtype: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 1).A1


def pmatrix_cm(x, args):
    """Computes and returns pmatrix of UBCM.

    :param x: solutions of UBCM.
    :type x: numpy.ndarray
    :param args: number of nodes.
    :type args: (int, )
    :return: pmatrix UBCM.
    :rtype: numpy.ndarray
    """
    n = args[0]
    f = np.zeros(shape=(n, n), dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            aux = x[i] * x[j]
            aux1 = aux / (1 + aux)
            f[i, j] = aux1
            f[j, i] = aux1
    return f


@jit(nopython=True)
def iterative_cm(x, args):
    """Computes loglikelihood parameters x at step n+1 given their value at step n for UBCM.

    :param x: loglikelihood parameters *x* at step n.
    :type x: numpy.ndarray
    :param args: degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood parameters *x* at step n+1.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        fx = 0
        for j in np.arange(n):
            if i == j:
                fx += (c[j] - 1) * (x[j] / (1 + x[j] * x[i]))
            else:
                fx += (c[j]) * (x[j] / (1 + x[j] * x[i]))
        if fx:
            f[i] = k[i] / fx
    return f


@jit(nopython=True)
def loglikelihood_cm(x, args):
    """Computes loglikelihood of UBCM.

    :param x: loglikelihood parameters *x*.
    :type x: numpy.ndarray
    :param args: degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood value.
    :rtype: float
    """
    k = args[0]
    c = args[1]
    n = len(k)
    f = 0.0
    for i in np.arange(n):
        f += c[i] * k[i] * np.log(x[i])
        for j in np.arange(n):
            if i == j:
                f -= (c[i] * (c[i] - 1) * np.log(1 + (x[i]) ** 2)) / 2
            else:
                f -= (c[i] * c[j] * np.log(1 + x[i] * x[j])) / 2
    return f


@jit(nopython=True)
def loglikelihood_prime_cm(x, args):
    """Computes loglikelihood first derivatives of UBCM.

    :param x: loglikelihood parameters *x*.
    :type x: numpy.ndarray
    :param args: degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood first derivatives.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        f[i] += c[i] * k[i] / x[i]
        for j in np.arange(n):
            if i == j:
                f[i] -= c[i] * (c[j] - 1) * (x[j] / (1 + (x[j] ** 2)))
            else:
                f[i] -= c[i] * c[j] * (x[j] / (1 + x[i] * x[j]))
    return f


@jit(nopython=True)
def loglikelihood_hessian_cm(x, args):
    """Computes second derivatives (hessian matrix) of UBCM loglikelihood function.

    :param x: loglikelihood parameters x.
    :type x: numpy.ndarray
    :param args: degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: hessian matrix.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros(shape=(n, n), dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(i, n):
            if i == j:
                aux_f = -k[i] / (x[i] * x[i]) * c[i]
                for h in range(n):
                    if i == h:
                        aux = 1 + x[h] * x[h]
                        aux_f += ((x[h] * x[h]) /
                                  (aux * aux)) * c[i] * (c[h] - 1)
                    else:
                        aux = 1 + x[i] * x[h]
                        aux_f += ((x[h] * x[h]) / (aux * aux)) * c[i] * c[h]
            else:
                aux = 1 + x[i] * x[j]
                aux_f = ((x[j] * x[j] - aux) / (aux * aux)) * c[i] * c[j]

            f[i, j] = aux_f
            f[j, i] = aux_f
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_cm(x, args):
    """Computes the diagonal of UBCM loglikelihood function hessian matrix.

    :param x: loglikelihood parameters x.
    :type x: numpy.ndarray
    :param args: degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: diagonal of the hessian matrix.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros(n, dtype=np.float64)
    for i in np.arange(n):
        f[i] -= k[i] / (x[i] * x[i]) * c[i]
        for j in np.arange(n):
            if i == j:
                aux = 1 + x[j] * x[j]
                f[i] += ((x[j] * x[j]) / (aux * aux)) * c[i] * (c[j] - 1)
            else:
                aux = 1 + x[i] * x[j]
                f[i] += ((x[j] * x[j]) / (aux * aux)) * c[i] * c[j]
    return f


@jit(nopython=True)
def iterative_crema(beta, args):
    """Computes loglikelihood parameters x at step n+1 given their value at step n for CReMa. The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: loglikelihood parameters beta at step n.
    :type beta: numpy.ndarray
    :param args: strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood parameters beta at step n+1.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]
    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        f[i] -= w / (1 + (beta[j] / beta[i]))
        f[j] -= w / (1 + (beta[i] / beta[j]))
    for i in np.arange(n):
        if s[i] != 0:
            f[i] = f[i] / s[i]
    return f


@jit(nopython=True, parallel=True, nogil=True)
def iterative_crema_sparse(beta, args):
    """Computes loglikelihood parameters x at step n+1 given their value at step n for CReMa. The UBCM pmatrix is computed inside the function in order to avoid memory errors due to the dimensions of the latter.

    :param beta: loglikelihood parameters beta at step n.
    :type beta: numpy.ndarray
    :param args: strengths sequence and UBCM solutions.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood parameters beta at step n+1.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]

    for i in prange(n):
        aux = x[i] * x
        aux_value = aux / (1+aux)
        aux = aux_value / (1 + (beta/beta[i]))
        f[i] = (-aux.sum() + aux[i])/(s[i] + np.exp(-100))
    return f


@jit(nopython=True)
def iterative_crema_sparse_2(beta, args):
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]

    for i in np.arange(n):
        for j in np.arange(i+1, n):
            aux = x[i] * x[j]
            aux_value = aux / (1 + aux)
            if aux_value > 0:
                f[i] -= aux_value / (1 + (beta[j] / beta[i]))
                f[j] -= aux_value / (1 + (beta[i] / beta[j]))
    for i in np.arange(n):
        if s[i] != 0:
            f[i] = f[i] / s[i]
    return f


@jit(nopython=True)
def loglikelihood_crema(beta, args):
    """Computes CReMa loglikelihood function. The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood value.
    :rtype: float
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = 0.0
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i in np.arange(n):
        f -= s[i] * beta[i]
    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        f += w * np.log(beta[i] + beta[j])

    return f


@jit(nopython=True, nogil=True)
def loglikelihood_crema_sparse(beta, args):
    """Computes CReMa loglikelihood function. The UBCM pmatrix is computed inside the function in order to avoid memory errors due to the dimensions of the latter.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and UBCM solutions.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood value.
    :rtype: float
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = 0.0
    x = adj[0]

    for i in np.arange(n):
        f -= s[i] * beta[i]
        for j in np.arange(0, i):
            aux = x[i] * x[j]
            aux_value = aux / (1 + aux)
            if aux_value > 0:
                f += aux_value * np.log(beta[i] + beta[j])
    return f


@jit(nopython=True)
def loglikelihood_prime_crema(beta, args):
    """Computes CReMa loglikelihood function first derivatives. The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood first derivatives.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i in np.arange(n):
        f[i] -= s[i]
    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux = beta[i] + beta[j]
        f[i] += w / aux
        f[j] += w / aux
    return f


@jit(nopython=True)
def loglikelihood_prime_crema_sparse_2(beta, args):
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]
    for i in np.arange(n):
        f[i] -= s[i]
        for j in np.arange(0, i):
            aux = x[i] * x[j]
            aux_value = aux / (1 + aux)
            if aux_value > 0:
                aux = beta[i] + beta[j]
                f[i] += aux_value / aux
                f[j] += aux_value / aux
    return f


@jit(nopython=True, parallel=True, nogil=True)
def loglikelihood_prime_crema_sparse(beta, args):
    """Computes CReMa loglikelihood function first derivatives. The UBCM pmatrix is computed inside the function in order to avoid memory errors due to the dimensions of the latter.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and UBCM solutions.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood first derivatives.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]
    for i in prange(n):
        f[i] -= s[i]
        aux = x[i] * x
        aux_value = aux / (1 + aux)
        aux = aux_value/(beta[i] + beta)
        f[i] += aux.sum() - aux[i]
    return f


@jit(nopython=True)
def loglikelihood_hessian_crema(beta, args):
    """Computes CReMa loglikelihood function second derivatives (hessian matrix). The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: hessian matrix.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros(shape=(n, n), dtype=np.float64)
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux = -w / ((beta[i] + beta[j]) ** 2)
        f[i, j] = aux
        f[j, i] = aux
        f[i, i] += aux
        f[j, j] += aux
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_crema(beta, args):
    """Computes the diagonal of CReMa loglikelihood function hessian matrix. The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: hessian matrix diagonal.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    f = np.zeros_like(s, dtype=np.float64)
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux = w / ((beta[i] + beta[j]) ** 2)
        f[i] -= aux
        f[j] -= aux
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_crema_sparse_2(beta, args):
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]
    for i in np.arange(n):
        for j in np.arange(0, i):
            if i != j:
                aux = x[i] * x[j]
                aux_value = aux / (1 + aux)
                if aux_value > 0:
                    aux = aux_value / ((beta[i] + beta[j]) ** 2)
                    f[i] -= aux
                    f[j] -= aux
    return f


@jit(nopython=True, parallel=True, nogil=True)
def loglikelihood_hessian_diag_crema_sparse(beta, args):
    """Computes the diagonal of CReMa loglikelihood function hessian matrix. The UBCM pmatrix is computed inside the function in order to avoid memory errors due to the dimensions of the latter.

    :param beta: loglikelihood parameters beta.
    :type beta: numpy.ndarray
    :param args: strengths sequence and UBCM solutions.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: hessian matrix diagonal.
    :rtype: numpy.ndarray
    """
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]
    for i in prange(n):
        aux = x[i] * x
        aux_value = aux / (1 + aux)
        aux = aux_value / ((beta[i] + beta) ** 2)
        f[i] -= aux.sum() - aux[i]
    return f


@jit(nopython=True)
def iterative_ecm(sol, args):
    """Computes loglikelihood parameters x at step n+1 given their value at step n for UECM.

    :param sol: loglikelihood parameters x and y at step n.
    :type sol: numpy.ndarray
    :param args: degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood parameters x and y at step n+1.
    :rtype: numpy.ndarray
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]

    f = np.zeros(2 * n, dtype=np.float64)
    for i in np.arange(n):
        fx = 0.0
        fy = 0.0
        for j in np.arange(n):
            if i != j:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                fx += (x[j] * aux2) / (1 - aux2 + aux1 * aux2)
                fy += (aux1 * y[j]) / ((1 - aux2) * (1 - aux2 + aux1 * aux2))
        if fx:
            f[i] = k[i] / fx
        else:
            f[i] = 0.0
        if fy:
            f[i + n] = s[i] / fy
        else:
            f[i + n] = 0.0
    return f


@jit(nopython=True)
def loglikelihood_ecm(sol, args):
    """Computes UECM loglikelihood function given the parameters x and y.

    :param sol: loglikelihood parameters x and y.
    :type sol: numpy.ndarray
    :param args: degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood function value.
    :rtype: float
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = 0.0
    for i in np.arange(n):
        f += k[i] * np.log(x[i]) + s[i] * np.log(y[i])
        for j in np.arange(0, i):
            aux = y[i] * y[j]
            f += np.log((1 - aux) / (1 - aux + x[i] * x[j] * aux))
    return f


@jit(nopython=True)
def loglikelihood_prime_ecm(sol, args):
    """Computes UECM loglikelihood function first dervatives given the parameters x and y.

    :param sol: loglikelihood parameters x and y.
    :type sol: numpy.ndarray
    :param args: degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: loglikelihood first derivatives.
    :rtype: numpy.ndarray
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(2 * n, dtype=np.float64)
    for i in np.arange(n):
        f[i] += k[i] / x[i]
        f[i + n] += s[i] / y[i]
        for j in np.arange(n):
            if i != j:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                f[i] -= (x[j] * aux2) / (1 - aux2 + aux1 * aux2)
                f[i + n] -= (aux1 * y[j]) / (
                    (1 - aux2) * (1 - aux2 + aux1 * aux2)
                )
    return f


@jit(nopython=True)
def loglikelihood_hessian_ecm(sol, args):
    """Computes UECM loglikelihood function second dervatives (hessian matrix) given the parameters x and y.

    :param sol: loglikelihood parameters x and y.
    :type sol: numpy.ndarray
    :param args: degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: hessian matrix.
    :rtype: numpy.ndarray
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(shape=(2 * n, 2 * n), dtype=np.float64)
    for i in np.arange(n):

        for j in np.arange(i, n):
            if i == j:
                f1 = -k[i] / (x[i] ** 2)
                f2 = -s[i] / ((y[i]) ** 2)
                f3 = 0.0
                for h in np.arange(n):
                    if h != i:
                        aux1 = x[i] * x[h]
                        aux2 = y[i] * y[h]
                        aux3 = (1 - aux2) ** 2
                        aux4 = (1 - aux2 + aux1 * aux2) ** 2
                        f1 += ((x[h] * aux2) ** 2) / aux4
                        f2 += (
                            (
                                aux1
                                * y[h]
                                * (
                                    aux1 * y[h] * (1 - 2 * aux2)
                                    - 2 * y[h] * (1 - aux2)
                                )
                            )
                        ) / (aux3 * aux4)
                        f3 -= (x[h] * y[h]) / aux4
                f[i, i] = f1
                f[i + n, i + n] = f2
                f[i + n, i] = f3
                f[i, i + n] = f3
            else:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                aux3 = (1 - aux2) ** 2
                aux4 = (1 - aux2 + aux1 * aux2) ** 2

                aux = -(aux2 * (1 - aux2)) / aux4
                f[i, j] = aux
                f[j, i] = aux

                aux = -(x[j] * y[i]) / aux4
                f[i, j + n] = aux
                f[j + n, i] = aux

                aux = -(aux1 * (1 - aux2 ** 2 + aux1 * (aux2 ** 2))) / (
                    aux3 * aux4
                )
                f[i + n, j + n] = aux
                f[j + n, i + n] = aux

                aux = -(x[i] * y[j]) / aux4
                f[i + n, j] = aux
                f[j, i + n] = aux

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_ecm(sol, args):
    """Computes the diagonal of UECM hessian matrix given the parameters x and y.

    :param sol: loglikelihood parameters x and y.
    :type sol: numpy.ndarray
    :param args: degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: hessian matrix diagonal.
    :rtype: numpy.ndarray
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(2 * n, dtype=np.float64)

    for i in np.arange(n):
        f[i] -= k[i] / (x[i] * x[i])
        f[i + n] -= s[i] / (y[i] * y[i])
        for j in np.arange(n):
            if j != i:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                aux3 = (1 - aux2) ** 2
                aux4 = (1 - aux2 + aux1 * aux2) ** 2
                f[i] += ((x[j] * aux2) ** 2) / aux4
                f[i + n] += (
                    aux1
                    * y[j]
                    * (aux1 * y[j] * (1 - 2 * aux2) - 2 * y[j] * (1 - aux2))
                ) / (aux3 * aux4)
    return f


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
    full_return=False,
    linsearch=True,
):
    """Find roots of eq. f = 0, using newton, quasinewton or dianati."""

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
    dx_old = np.zeros_like(x0)

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
        norm > tol and diff > eps and n_steps < max_steps
    ):  # stopping condition

        x_old = x  # save previous iteration

        # f jacobian
        tic = time.time()
        if method == "newton":
            H = fun_jac(x)  # original jacobian
            # TODO: levare i verbose sugli eigenvalues
            if verbose:
                l, e = scipy.linalg.eigh(H)
                ml = np.min(l)
                Ml = np.max(l)
            if regularise:
                B = hessian_regulariser(
                    H, np.max(np.abs(fun(x))) * 1e-3
                )
                # TODO: levare i verbose sugli eigenvalues
                if verbose:
                    l, e = scipy.linalg.eigh(B)
                    new_ml = np.min(l)
                    new_Ml = np.max(l)
            else:
                B = H.__array__()
        elif method == "quasinewton":
            # quasinewton hessian approximation
            B = fun_jac(x)  # Jacobian diagonal
            if regularise:
                B = np.maximum(B, B * 0 + np.max(np.abs(fun(x))) * 1e-3)
        toc_jacfun += time.time() - tic

        # discending direction computation
        tic = time.time()
        if method == "newton":
            dx = np.linalg.solve(B, -f)
        elif method == "quasinewton":
            dx = -f / B
        elif method == "fixed-point":
            dx = f - x
        toc_dx += time.time() - tic

        # backtraking line search
        tic = time.time()

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
        diff = np.linalg.norm(x - x_old)

        if full_return:
            norm_seq.append(norm)
            diff_seq.append(diff)

        # step update
        n_steps += 1

        if verbose == True:
            print("step {}".format(n_steps))
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

    toc_loop = time.time() - tic_loop
    toc_all = time.time() - tic_all

    if verbose == True:
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
def linsearch_fun_crema(X, args):
    """Linsearch function for CReM newton and quasinewton methods. The alfa parameters controls the size of the step such that the solution at the next iteration is better than the actual one.

    :param X: loglikelihood parameters, increment, beta parameter, alfa parameter and loglikelihood prime function.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: CReM loglikelihood function and its parameters.
    :type args: (func, tuple)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while ((not sufficient_decrease_condition(s_old,
                                              -step_fun(x + alfa * dx,
                                                        arg_step_fun),
                                              alfa,
                                              f,
                                              dx))
            and (i < 50)
           ):
        alfa *= beta
        i += 1

    return alfa


@jit(nopython=True)
def linsearch_fun_crema_fixed(X):
    """Linsearch function for CReM fixed-point method. The alfa parameters controls the size of the step such that its norm at the next step is lower than the actual one.

    :param X: increment, old increment, beta parameter, alfa parameter and iteration number.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: alfa parameter value.
    :rtype: float
    """
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            (cond == False)
            and (kk < 50)
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)

    return alfa


@jit(nopython=True)
def linsearch_fun_CM_new(X, args):
    """Linsearch function for UBCM newton and quasinewton methods. The alfa parameters controls the size of the step such that the solution at the next iteration is better than the actual one.

    :param X: loglikelihood parameters, increment, beta parameter, alfa parameter and loglikelihood prime function.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: CReM loglikelihood function and its parameters.
    :type args: (func, tuple)
    :return: alfa parameter value
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1
    # print(alfa)
    return alfa


@jit(nopython=True)
def linsearch_fun_CM_new_fixed(X):
    """Linsearch function for UBCM fixed-point method. The alfa parameters controls the size of the step such that its norm at the next step is lower than the actual one.

    :param X: increment, old increment, beta parameter, alfa parameter and iteration number.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, int)
    :type X: (numpy.ndarray, float, float, int).
    :return: alfa parameter value
    :rtype: float
    """
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            cond == False
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    # print(alfa)
    return alfa


@jit(nopython=True)
def linsearch_fun_CM(X, args):
    """Linsearch function for UBCM newton and quasinewton methods. The alfa parameters controls the size of the step such that the solution at the next iteration is better than the actual one.

    :param X: loglikelihood parameters, increment, beta parameter, alfa parameter and loglikelihood prime function.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: CReM loglikelihood function and its parameters.
    :type args: (func, tuple)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    for a in alfa0:
        if a >= 0:
            alfa = min(alfa, a)

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_CM_fixed(X):
    """Linsearch function for UBCM fixed-point method. The alfa parameters controls the size of the step such that its norm at the next step is lower than the actual one.

    :param X: loglikelihood parameters, increment, old increment, beta parameter, alfa parameter and iteration number.
    :type X: (numpy.ndarray, numpy.ndarray, numpy.ndarray float, float, int)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    for a in alfa0:
        if a >= 0:
            alfa = min(alfa, a)

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            cond == False
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


@jit(nopython=True)
def linsearch_fun_ECM_new(X, args):
    """Linsearch function for UECM newton and quasinewton methods. The alfa parameters controls the size of the step such that the solution at the next iteration is better than the actual one.

    :param X: loglikelihood parameters, increment, beta parameter, alfa parameter and loglikelihood prime function.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: CReM loglikelihood function and its parameters.
    :type args: (func, tuple)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    nnn = int(len(x) / 2)
    while True:
        ind_min_beta = (x[nnn:] + alfa * dx[nnn:]).argsort()[:2]
        cond = np.sum(x[nnn:][ind_min_beta] +
                      alfa * dx[nnn:][ind_min_beta]) > 1e-14
        if (
            cond
        ):
            break
        else:
            alfa *= beta

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1

    return alfa


@jit(nopython=True)
def linsearch_fun_ECM_new_fixed(X):
    """Linsearch function for UECM fixed-point method. The alfa parameters controls the size of the step such that its norm at the next step is lower than the actual one.

    :param X: loglikelihood parameters, increment, old increment, beta parameter, alfa parameter and iteration number.
    :type X: (numpy.ndarray, numpy.ndarray, numpy.ndarray float, float, int)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    nnn = int(len(x) / 2)
    while True:
        ind_min_beta = (x[nnn:] + alfa * dx[nnn:]).argsort()[:2]
        cond = np.sum(x[nnn:][ind_min_beta] +
                      alfa * dx[nnn:][ind_min_beta]) > 1e-14
        if (
            cond
        ):
            break
        else:
            alfa *= beta

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            (cond == False)
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)

    return alfa


@jit(nopython=True)
def linsearch_fun_ECM(X, args):
    """Linsearch function for UECM newton and quasinewton methods. The alfa parameters controls the size of the step such that the solution at the next iteration is better than the actual one.

    :param X: loglikelihood parameters, increment, beta parameter, alfa parameter and loglikelihood prime function.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: CReM loglikelihood function and its parameters.
    :type args: (func, tuple)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    for a in alfa0:
        if a >= 0:
            alfa = min(alfa, a)

    nnn = int(len(x) / 2)
    while True:
        ind_max_y = (x[nnn:] + alfa * dx[nnn:]).argsort()[-2:][::-1]
        cond = np.prod(x[nnn:][ind_max_y] + alfa * dx[nnn:][ind_max_y]) < 1
        if cond:
            break
        else:
            alfa *= beta

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1

    return alfa


@jit(nopython=True)
def linsearch_fun_ECM_fixed(X):
    """Linsearch function for UECM fixed-point method. The alfa parameters controls the size of the step such that its norm at the next step is lower than the actual one.

    :param X: loglikelihood parameters, increment, old increment, beta parameter, alfa parameter and iteration number.
    :type X: (numpy.ndarray, numpy.ndarray, numpy.ndarray float, float, int)
    :return: alfa parameter value.
    :rtype: float
    """
    x = X[0]
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    for a in alfa0:
        if a >= 0:
            alfa = min(alfa, a)

    nnn = int(len(x) / 2)
    while True:
        ind_max_y = (x[nnn:] + alfa * dx[nnn:]).argsort()[-2:][::-1]
        cond = np.prod(x[nnn:][ind_max_y] + alfa * dx[nnn:][ind_max_y]) < 1
        if cond:
            break
        else:
            alfa *= beta

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            cond == False
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)

    return alfa


@jit(nopython=True)
def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-04, c2=0.9
):
    """Return boolean indicator if upper wolfe condition are respected.

    :param f_old: loglikelihood value at the previous iteration.
    :type f_old: float
    :param f_new: loglikelihood value at the actual iteration.
    :type f_new: float
    :param alpha: alfa parameter of linsearch.
    :type alpha: float
    :param grad_f: loglikelihood prime.
    :type grad_f: numpy.ndarray
    :param p: increment at the actual iteration.
    :type p: numpy.ndarray
    :param c1: [description], defaults to 1e-04.
    :type c1: parameter wolfe conditon, optional
    :param c2: parameter wolfe conditon, defaults to 0.9.
    :type c2: float, optional
    :return: condition is satisfied.
    :rtype: bool
    """
    sup = f_old + c1 * alpha * np.dot(grad_f, p.T)

    return bool(f_new < sup)


def hessian_regulariser_function(B, eps):
    """ Guarantes that hessian matrix is definitie posive by adding identity matrix multiplied for eps.
     

    :param B: hessian matrix.
    :type B: numpy.ndarray
    :param eps: parameter multypling the identity.
    :type eps: float
    :return: regularised hessian matrix.
    :rtype: numpy.ndarray
    """
    B = (B + B.transpose()) * 0.5  # symmetrization
    Bf = B + np.identity(B.shape[0]) * eps

    return Bf


def hessian_regulariser_function_eigen_based(B, eps):
    """Guarantes that hessian matrix eigenvalues are posive by manipulating their values.

    :param B: hessian matrix.
    :type B: numpy.ndarray
    :param eps: controls the regularisation.
    :type eps: float
    :return: regularised hessian matrix.
    :rtype: numpy.ndarray
    """
    B = (B + B.transpose()) * 0.5  # symmetrization
    l, e = scipy.linalg.eigh(B)
    ll = np.array([0 if li > eps else eps - li for li in l])
    Bf = e @ (np.diag(ll) + np.diag(l)) @ e.transpose()

    return Bf


@jit(nopython=True)
def expected_degree_cm(sol):
    """Computes the expected degrees of UBCM given its solution x.

    :param sol: UBCM solutions.
    :type sol: numpy.ndarray
    :return: expected degrees sequence.
    :rtype: numpy.ndarray
    """
    ex_k = np.zeros_like(sol, dtype=np.float64)
    n = len(sol)
    for i in np.arange(n):
        for j in np.arange(n):
            if i != j:
                aux = sol[i] * sol[j]
                # print("({},{}) p = {}".format(i,j,aux/(1+aux)))
                ex_k[i] += aux / (1 + aux)
    return ex_k


@jit(nopython=True)
def expected_strength_crema(sol, adj):
    """Computes the expected strengths of CReMa given its solution beta.

    :param sol: CReMa solutions.
    :type sol: numpy.ndarray
    :param adj: adjacency/pmatrix.
    :type adj: numpy.ndarray
    :return: expected strengths sequence.
    :rtype: numpy.ndarray
    """
    ex_s = np.zeros_like(sol, dtype=np.float64)
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux = w / (sol[i] + sol[j])
        ex_s[i] += aux
        ex_s[j] += aux
    return ex_s


@jit(nopython=True)
def expected_strength_crema_sparse(sol, adj):
    """Computes the expected strengths of CReMa given its solution beta and the solutions of UBCM.

    :param sol: CReMa solutions.
    :type sol: numpy.ndarray
    :param adj: UBCM solutions.
    :type adj: numpy.ndarray
    :return: expected strengths sequence.
    :rtype: numpy.ndarray
    """
    ex_s = np.zeros_like(sol, dtype=np.float64)
    n = len(sol)
    x = adj[0]
    for i in np.arange(n):
        for j in np.arange(0, i):
            aux = x[i] * x[j]
            aux_value = aux / (1 + aux)
            if aux_value > 0:
                aux = aux_value / (sol[i] + sol[j])
                ex_s[i] += aux
                ex_s[j] += aux
    return ex_s


@jit(nopython=True)
def expected_ecm(sol):
    """Computes expected degrees and strengths sequence given solution x and y of UECM.

    :param sol: UECM solutions.
    :type sol: numpy.ndarray
    :return: expected degrees and strengths sequence.
    :rtype: numpy.ndarray
    """
    n = int(len(sol) / 2)
    x = sol[:n]
    y = sol[n:]
    ex_ks = np.zeros(2 * n, dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(n):
            if i != j:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                ex_ks[i] += (aux1 * aux2) / (1 - aux2 + aux1 * aux2)
                ex_ks[i + n] += (aux1 * aux2) / (
                    (1 - aux2 + aux1 * aux2) * (1 - aux2)
                )
    return ex_ks


def edgelist_from_edgelist(edgelist):
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


class UndirectedGraph:
    """Undirected Graph instance can be initialised with adjacency matrix, edgelist, degrees sequence or strengths sequence.

    :param adjacency: Adjacency matrix, defaults to None.
    :type adjacency: numpy.ndarray, list or scipy.sparse_matrix, optional
    :param edgelist: edgelist, defaults to None.
    :type edgelist: numpy.ndarray, list, optional
    :param degree_sequence: degrees sequence, defaults to None.
    :type degree_sequence: numpy.ndarray, optional
    :param strength_sequence: strengths sequence, defaults to None.
    :type strength_sequence: numpy.ndarray, optional
    """
    def __init__(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
        """Initilizes all the necessary attribitus for Undirected graph class.

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
        self.strength_sequence = None
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
        self.r_n = None
        self.r_invert_dseq = None
        self.r_dim = None
        self.r_multiplicity = None

        # Problem solutions
        self.x = None
        self.beta = None

        # reduced solutions
        self.r_x = None

        # Problem (reduced) residuals
        self.residuals = None
        self.final_result = None
        self.r_beta = None

        self.nz_index = None
        self.rnz_n = None

        # model
        self.x0 = None
        self.error = None
        self.error_degree = None
        self.relative_error_degree = None
        self.error_strength = None
        self.relative_error_strength = None
        self.full_return = False
        self.last_model = None

        # function
        self.args = None

    def _initialize_graph(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
        # Here we can put controls over the type of input. For instance, if the graph is directed,
        # i.e. adjacency matrix is asymmetric, the class to use must be the DiGraph,
        # or if the graph is weighted (edgelist contains triplets or matrix is not binary) or bipartite

        if adjacency is not None:
            if not isinstance(
                adjacency, (list, np.ndarray)
            ) and not scipy.sparse.isspmatrix(adjacency):
                raise TypeError(
                    "The adjacency matrix must be passed as a list or numpy array or scipy sparse matrix."
                )
            elif adjacency.size > 0:
                if np.sum(adjacency < 0):
                    raise TypeError(
                        "The adjacency matrix entries must be positive."
                    )
                if isinstance(
                    adjacency, list
                ):
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.adjacency = adjacency
                    self.is_sparse = True
                if np.sum(adjacency) == np.sum(adjacency > 0):
                    self.dseq = degree(adjacency).astype(np.float64)
                else:
                    self.dseq = degree(adjacency).astype(np.float64)
                    self.strength_sequence = strength(adjacency).astype(
                        np.float64
                    )
                    self.nz_index = np.nonzero(self.strength_sequence)[0]
                    self.is_weighted = True

                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)/2
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError(
                    "The edgelist must be passed as a list or numpy array."
                )
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        "This is not an edgelist. An edgelist must be a list or array of couples of nodes with optional weights. Is this an adjacency matrix?"
                    )
                elif len(edgelist[0]) == 2:
                    (
                        self.edgelist,
                        self.dseq,
                        self.nodes_dict,
                    ) = edgelist_from_edgelist(edgelist)
                else:
                    (
                        self.edgelist,
                        self.dseq,
                        self.strength_sequence,
                        self.nodes_dict,
                    ) = edgelist_from_edgelist(edgelist)
                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)/2
                self.is_initialized = True

        elif degree_sequence is not None:
            if not isinstance(degree_sequence, (list, np.ndarray)):
                raise TypeError(
                    "The degree sequence must be passed as a list or numpy array."
                )
            elif len(degree_sequence) > 0:
                try:
                    int(degree_sequence[0])
                except:
                    raise TypeError(
                        "The degree sequence must contain numeric values."
                    )
                if (np.array(degree_sequence) < 0).sum() > 0:
                    raise ValueError("A degree cannot be negative.")
                else:
                    self.n_nodes = int(len(degree_sequence))
                    self.dseq = degree_sequence.astype(np.float64)
                    self.n_edges = np.sum(self.dseq)/2
                    self.is_initialized = True

                if strength_sequence is not None:
                    if not isinstance(strength_sequence, (list, np.ndarray)):
                        raise TypeError(
                            "The strength sequence must be passed as a list or numpy array."
                        )
                    elif len(strength_sequence):
                        try:
                            int(strength_sequence[0])
                        except:
                            raise TypeError(
                                "The strength sequence must contain numeric values."
                            )
                        if (np.array(strength_sequence) < 0).sum() > 0:
                            raise ValueError("A strength cannot be negative.")
                        else:
                            if len(strength_sequence) != len(degree_sequence):
                                raise ValueError(
                                    "Degrees and strengths arrays must have same length."
                                )
                            self.n_nodes = int(len(strength_sequence))
                            self.strength_sequence = strength_sequence.astype(
                                np.float64
                            )
                            self.nz_index = np.nonzero(self.strength_sequence)[
                                0
                            ]
                            self.is_weighted = True
                            self.is_initialized = True

        elif strength_sequence is not None:
            if not isinstance(strength_sequence, (list, np.ndarray)):
                raise TypeError(
                    "The strength sequence must be passed as a list or numpy array."
                )
            elif len(strength_sequence):
                try:
                    int(strength_sequence[0])
                except:
                    raise TypeError(
                        "The strength sequence must contain numeric values."
                    )
                if (np.array(strength_sequence) < 0).sum() > 0:
                    raise ValueError("A strength cannot be negative.")
                else:
                    self.n_nodes = int(len(strength_sequence))
                    self.strength_sequence = strength_sequence
                    self.nz_index = np.nonzero(self.strength_sequence)[0]
                    self.is_weighted = True
                    self.is_initialized = True

    def set_adjacency_matrix(self, adjacency):
        """Initialises graph given the adjacency matrix.

        :param adjacency: ajdacency matrix.
        :type adjacency: numpy.ndarray, list, scipy.sparse_matrix
        """
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        """Initialises graph given the edgelist.

        :param edgelist: edgelist.
        :type edgelist: numpy.ndarray, list
        """
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequence):
        """Initialises graph given the degree sequence.

        :param degree_sequence: degrees sequence.
        :type degree_sequence: numpy.ndarray
        """
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(degree_sequence=degree_sequence)

    def clean_edges(self):
        """
        Deletes all the initialiased attributes.

        """
        self.adjacency = None
        self.edgelist = None
        self.deg_seq = None
        self.is_initialized = False

    def _solve_problem(
        self,
        initial_guess=None,
        model="cm",
        method="quasinewton",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
    ):
        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self.regularise = regularise
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
            full_return=full_return,
            linsearch=linsearch,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_cm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        self.r_x = self.r_xy
        if self.last_model == "cm":
            self.x = self.r_x[self.r_invert_dseq]
        elif self.last_model == "cm-new":
            self.x = np.exp(-self.r_x[self.r_invert_dseq])

    def _set_solved_problem(self, solution):
        model = self.last_model
        if model in ["cm", "cm-new"]:
            self._set_solved_problem_cm(solution)
        elif model in ["ecm", "ecm-new"]:
            self._set_solved_problem_ecm(solution)
        elif model in ["crema", "crema-sparse"]:
            self._set_solved_problem_crema(solution)

    def degree_reduction(self):
        """
        Carries out degree reduction.
        """
        self.r_dseq, self.r_index_dseq, self.r_invert_dseq, self.r_multiplicity = np.unique(
            self.dseq,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )

        self.rnz_n = self.r_dseq.size

        self.is_reduced = True

    def _set_initial_guess(self, model):
        """Calls the proper set initial guess function given the selected *model*.

        :param model: Selected model.
        :type model: str
        """

        if model in ["cm", "cm-new"]:
            self._set_initial_guess_cm()
        elif model in ["ecm", "ecm-new"]:
            self._set_initial_guess_ecm()
        elif model in ["crema", "crema-sparse"]:
            self._set_initial_guess_crema()

    def _set_initial_guess_cm(self):
        """Sets the initial guess for UBCM given the choice made by the user.

        :raises ValueError: raises value error if the selected *initial_guess* is not among the exisisting ones.
        :raises TypeError: raises type error if the selected *initial_guess* is wrong.
        """

        if ~self.is_reduced:
            self.degree_reduction()

        if isinstance(self.initial_guess, np.ndarray):
            self.r_x = self.initial_guess[self.r_index_dseq]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "degrees_minor":
                self.r_x = self.r_dseq / (
                    np.sqrt(self.n_edges) + 1
                )  # This +1 increases the stability of the solutions.
            elif self.initial_guess == "random":
                self.r_x = np.random.rand(self.rnz_n).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.r_x = 0.5 * np.ones(
                    self.rnz_n, dtype=np.float64
                )  # All probabilities will be 1/2 initially
            elif self.initial_guess == "degrees":
                self.r_x = self.r_dseq.astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.r_x = self.r_dseq.astype(np.float64)/(2*self.n_edges)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.r_x[self.r_dseq == 0] = 0

        if isinstance(self.initial_guess, str):
            if self.last_model == "cm":
                self.x0 = self.r_x
            elif self.last_model == "cm-new":
                self.r_x[self.r_x != 0] = -np.log(self.r_x[self.r_x != 0])
                self.x0 = self.r_x
        elif isinstance(self.initial_guess, np.ndarray):
            self.x0 = self.r_x

    def _set_initial_guess_crema(self):
        """Sets the initial guess for CReMa given the choice made by the user.

        :raises ValueError: raises value error if the selected *initial_guess* is not among the exisisting ones.
        :raises TypeError: raises type error if the selected *initial_guess* is wrong.
        """

        if isinstance(self.initial_guess, np.ndarray):
            self.beta = self.initial_guess
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.beta = (self.strength_sequence > 0).astype(
                    float
                ) / self.strength_sequence.sum()
            elif self.initial_guess == "strengths_minor":
                self.beta = (self.strength_sequence > 0).astype(float) / (
                    self.strength_sequence + 1
                )
            elif self.initial_guess == "random":
                self.beta = np.random.rand(self.n_nodes).astype(np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.beta[self.strength_sequence == 0] = 0

        self.x0 = self.beta

    def _set_initial_guess_ecm(self):
        """Sets the initial guess for UECM given the choice made by the user.

        :raises ValueError: raises value error if the selected *initial_guess* is not among the exisisting ones.
        :raises TypeError: raises type error if the selected *initial_guess* is wrong.
        """
        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes]
            self.y = self.initial_guess[self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.x = self.dseq.astype(float) / (
                    self.n_edges + 1
                )  # This +1 increases the stability of the solutions.
                self.y = (
                    self.strength_sequence.astype(float)
                    / self.strength_sequence.sum()
                )
            elif self.initial_guess == "strengths_minor":
                self.x = np.ones_like(self.dseq, dtype=np.float64) / (
                    self.dseq + 1
                )
                self.y = np.ones_like(self.strength_sequence, dtype=np.float64) / (
                    self.strength_sequence + 1
                )
            elif self.initial_guess == "random":
                self.x = np.random.rand(self.n_nodes).astype(np.float64)
                self.y = np.random.rand(self.n_nodes).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.x = 0.001 * np.ones(self.n_nodes, dtype=np.float64)
                self.y = 0.001 * np.ones(self.n_nodes, dtype=np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.x[self.dseq == 0] = 0
        self.y[self.strength_sequence == 0] = 0

        if isinstance(self.initial_guess, str):
            if self.last_model == "ecm":
                self.x0 = np.concatenate((self.x, self.y))
            elif self.last_model == "ecm-new":
                self.x[self.x != 0] = -np.log(self.x[self.x != 0])
                self.y[self.y != 0] = -np.log(self.y[self.y != 0])
                self.x0 = np.concatenate((self.x, self.y))
        elif isinstance(self.initial_guess, np.ndarray):
            self.x0 = np.concatenate((self.x, self.y))

    # DA SISTEMARE
    def solution_error(self):
        """
        Computes the error given the solutions to the optimisation problem.
        """
        if self.last_model in ["cm", "cm-new", "crema", "crema-sparse"]:
            if self.x is not None:
                ex_k = expected_degree_cm(self.x)
                # print(k, ex_k)
                self.expected_dseq = ex_k
                # error output
                self.error_degree = np.linalg.norm(ex_k - self.dseq, ord=np.inf)
                self.relative_error_degree = np.linalg.norm((ex_k - self.dseq)/(self.dseq + np.exp(-100)), ord=np.inf)
                self.error = self.error_degree

            if self.beta is not None:
                if self.is_sparse:
                    ex_s = expected_strength_crema_sparse(
                        self.beta, self.adjacency_crema
                    )
                else:
                    ex_s = expected_strength_crema(
                        self.beta, self.adjacency_crema
                    )
                self.expected_stregth_seq = ex_s
                # error output
                self.error_strength = np.linalg.norm(
                    ex_s - self.strength_sequence, ord=np.inf
                )
                self.relative_error_strength = np.max(
                    abs(
                     (ex_s - self.strength_sequence) / (self.strength_sequence + np.exp(-100))
                    )
                )

                if self.adjacency_given:
                    self.error = self.error_strength
                else:
                    self.error = max(self.error_strength, self.error_degree)

        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ["ecm", "ecm-new"]:
            sol = np.concatenate((self.x, self.y))
            ex = expected_ecm(sol)
            k = np.concatenate((self.dseq, self.strength_sequence))
            self.expected_dseq = ex[: self.n_nodes]
            self.expected_strength_seq = ex[self.n_nodes:]

            # error output
            self.error_degree = np.linalg.norm(self.expected_dseq -
                                               self.dseq, ord=np.inf)
            self.error_strength = np.linalg.norm(self.expected_strength_seq -
                                                 self.strength_sequence,
                                                 ord=np.inf)
            self.relative_error_strength = max(
                abs(
                    (self.strength_sequence -
                     self.expected_strength_seq)/self.strength_sequence
                    )
                )

            self.relative_error_degree = max(
                abs(
                    (self.dseq - self.expected_dseq)/self.dseq
                    )
                )
            self.error = max(self.error_strength, self.error_degree)

    def _set_args(self, model):
        """Sets up functions arguments given for the selected *model*.

        :param model: model name.
        :type model: str
        """

        if model in ["crema", "crema-sparse"]:
            self.args = (
                self.strength_sequence,
                self.adjacency_crema,
                self.nz_index,
            )
        elif model in ["cm", "cm-new"]:
            self.args = (self.r_dseq, self.r_multiplicity)
        elif model in ["ecm", "ecm-new"]:
            self.args = (self.dseq, self.strength_sequence)

    def _initialize_problem(self, model, method):
        """Initialises all the functions and parameters necessary used by the selected 'model' and 'method'.

        :param model: model name.
        :type model: str
        :param method: method name.
        :type method: str
        :raises ValueError: Raise an error if model/method choice is not valid.
        """

        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = "-"
        mod_met = mod_met.join([model, method])

        d_fun = {
            "cm-newton": lambda x: -loglikelihood_prime_cm(x, self.args),
            "cm-quasinewton": lambda x: -loglikelihood_prime_cm(x, self.args),
            "cm-fixed-point": lambda x: iterative_cm(x, self.args),
            "crema-newton": lambda x: -loglikelihood_prime_crema(
                x, self.args
            ),
            "crema-quasinewton": lambda x: -loglikelihood_prime_crema(
                x, self.args
            ),
            "crema-fixed-point": lambda x: -iterative_crema(x, self.args),
            "ecm-newton": lambda x: -loglikelihood_prime_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -loglikelihood_prime_ecm(
                x, self.args
            ),
            "ecm-fixed-point": lambda x: iterative_ecm(x, self.args),
            "crema-sparse-newton": lambda x: -loglikelihood_prime_crema_sparse(
                x, self.args
            ),
            "crema-sparse-quasinewton": lambda x: -loglikelihood_prime_crema_sparse(
                x, self.args
            ),
            "crema-sparse-fixed-point": lambda x: -iterative_crema_sparse(
                x, self.args
            ),
            "cm-new-newton": lambda x: -loglikelihood_prime_cm_new(
                x, self.args
            ),
            "cm-new-quasinewton": lambda x: -loglikelihood_prime_cm_new(
                x, self.args
            ),
            "cm-new-fixed-point": lambda x: iterative_cm_new(x, self.args),
            "ecm-new-newton": lambda x: -loglikelihood_prime_ecm_new(
                x, self.args
            ),
            "ecm-new-quasinewton": lambda x: -loglikelihood_prime_ecm_new(
                x, self.args
            ),
            "ecm-new-fixed-point": lambda x: iterative_ecm_new(x, self.args),
        }

        d_fun_jac = {
            "cm-newton": lambda x: -loglikelihood_hessian_cm(x, self.args),
            "cm-quasinewton": lambda x: -loglikelihood_hessian_diag_cm(
                x, self.args
            ),
            "cm-fixed-point": None,
            "crema-newton": lambda x: -loglikelihood_hessian_crema(
                x, self.args
            ),
            "crema-quasinewton": lambda x: -loglikelihood_hessian_diag_crema(
                x, self.args
            ),
            "crema-fixed-point": None,
            "ecm-newton": lambda x: -loglikelihood_hessian_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -loglikelihood_hessian_diag_ecm(
                x, self.args
            ),
            "ecm-fixed-point": None,
            "crema-sparse-newton": lambda x: -loglikelihood_hessian_crema(
                x, self.args
            ),
            "crema-sparse-quasinewton": lambda x: -loglikelihood_hessian_diag_crema_sparse(
                x, self.args
            ),
            "crema-sparse-fixed-point": None,
            "cm-new-newton": lambda x: -loglikelihood_hessian_cm_new(
                x, self.args
            ),
            "cm-new-quasinewton": lambda x: -loglikelihood_hessian_diag_cm_new(
                x, self.args
            ),
            "cm-new-fixed-point": None,
            "ecm-new-newton": lambda x: -loglikelihood_hessian_ecm_new(
                x, self.args
            ),
            "ecm-new-quasinewton": lambda x: -loglikelihood_hessian_diag_ecm_new(
                x, self.args
            ),
            "ecm-new-fixed-point": None,
        }

        d_fun_stop = {
            "cm-newton": lambda x: -loglikelihood_cm(x, self.args),
            "cm-quasinewton": lambda x: -loglikelihood_cm(x, self.args),
            "cm-fixed-point": lambda x: -loglikelihood_cm(x, self.args),
            "crema-newton": lambda x: -loglikelihood_crema(x, self.args),
            "crema-quasinewton": lambda x: -loglikelihood_crema(
                x, self.args
            ),
            "crema-fixed-point": lambda x: -loglikelihood_crema(
                x, self.args
            ),
            "ecm-newton": lambda x: -loglikelihood_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -loglikelihood_ecm(x, self.args),
            "ecm-fixed-point": lambda x: -loglikelihood_ecm(x, self.args),
            "crema-sparse-newton": lambda x: -loglikelihood_crema_sparse(
                x, self.args
            ),
            "crema-sparse-quasinewton": lambda x: -loglikelihood_crema_sparse(
                x, self.args
            ),
            "crema-sparse-fixed-point": lambda x: -loglikelihood_crema_sparse(
                x, self.args
            ),
            "cm-new-newton": lambda x: -loglikelihood_cm_new(x, self.args),
            "cm-new-quasinewton": lambda x: -loglikelihood_cm_new(
                x, self.args
            ),
            "cm-new-fixed-point": lambda x: -loglikelihood_cm_new(
                x, self.args
            ),
            "ecm-new-newton": lambda x: -loglikelihood_ecm_new(x, self.args),
            "ecm-new-quasinewton": lambda x: -loglikelihood_ecm_new(
                x, self.args
            ),
            "ecm-new-fixed-point": lambda x: -loglikelihood_ecm_new(
                x, self.args
            ),
        }
        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.step_fun = d_fun_stop[mod_met]
        except:
            raise ValueError(
                'Method must be "newton","quasi-newton", or "fixed-point".'
            )

        d_pmatrix = {
            "cm": pmatrix_cm,
            "cm-new": pmatrix_cm,
        }

        if model in ["cm", "cm-new"]:
            self.args_p = (self.n_nodes, np.nonzero(self.dseq)[0])
            self.fun_pmatrix = lambda x: d_pmatrix[model](x, self.args_p)

        args_lin = {
            "cm": (loglikelihood_cm, self.args),
            "crema": (loglikelihood_crema, self.args),
            "crema-sparse": (loglikelihood_crema_sparse, self.args),
            "ecm": (loglikelihood_ecm, self.args),
            "cm-new": (loglikelihood_cm_new, self.args),
            "ecm-new": (loglikelihood_ecm_new, self.args),
        }

        self.args_lins = args_lin[model]

        lins_fun = {
            "cm-newton": lambda x: linsearch_fun_CM(x, self.args_lins),
            "cm-quasinewton": lambda x: linsearch_fun_CM(x, self.args_lins),
            "cm-fixed-point": lambda x: linsearch_fun_CM_fixed(x),
            "crema-newton": lambda x: linsearch_fun_crema(x, self.args_lins),
            "crema-quasinewton": lambda x: linsearch_fun_crema(x, self.args_lins),
            "crema-fixed-point": lambda x: linsearch_fun_crema_fixed(x),
            "crema-sparse-newton": lambda x: linsearch_fun_crema(x, self.args_lins),
            "crema-sparse-quasinewton": lambda x: linsearch_fun_crema(x, self.args_lins),
            "crema-sparse-fixed-point": lambda x: linsearch_fun_crema_fixed(x),
            "ecm-newton": lambda x: linsearch_fun_ECM(x, self.args_lins),
            "ecm-quasinewton": lambda x: linsearch_fun_ECM(x, self.args_lins),
            "ecm-fixed-point": lambda x: linsearch_fun_ECM_fixed(x),
            "cm-new-newton": lambda x: linsearch_fun_CM_new(x, self.args_lins),
            "cm-new-quasinewton": lambda x: linsearch_fun_CM_new(x, self.args_lins),
            "cm-new-fixed-point": lambda x: linsearch_fun_CM_new_fixed(x),
            "ecm-new-newton": lambda x: linsearch_fun_ECM_new(x, self.args_lins),
            "ecm-new-quasinewton": lambda x: linsearch_fun_ECM_new(x, self.args_lins),
            "ecm-new-fixed-point": lambda x: linsearch_fun_ECM_new_fixed(x)
        }

        self.fun_linsearch = lins_fun[mod_met]

        hess_reg = {
            "cm": hessian_regulariser_function_eigen_based,
            "cm-new": hessian_regulariser_function,
            "ecm": hessian_regulariser_function_eigen_based,
            "ecm-new": hessian_regulariser_function,
            "crema": hessian_regulariser_function,
            "crema-sparse": hessian_regulariser_function,
        }

        self.hessian_regulariser = hess_reg[model]

        if isinstance(self.regularise, str):
            if self.regularise == "eigenvalues":
                self.hessian_regulariser = hessian_regulariser_function_eigen_based
            elif self.regularise == "identity":
                self.hessian_regulariser = hessian_regulariser_function

    def _solve_problem_crema(
        self,
        initial_guess=None,
        model="crema",
        adjacency="cm",
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

            # aggiungere check sul modello passato per l'adjacency matrix

            self._solve_problem(
                initial_guess=initial_guess_adjacency,
                model=adjacency,
                method=method_adjacency,
                max_steps=max_steps,
                tol=tol,
                eps=eps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                regularise=regularise
            )

            if self.is_sparse:
                self.adjacency_crema = (self.x,)
                self.adjacency_given = False
            else:
                pmatrix = self.fun_pmatrix(self.x)
                raw_ind, col_ind = np.nonzero(np.triu(pmatrix))
                raw_ind = raw_ind.astype(np.int64)
                col_ind = col_ind.astype(np.int64)
                weigths_value = pmatrix[raw_ind, col_ind]
                self.adjacency_crema = (raw_ind, col_ind, weigths_value)
                self.is_sparse = False
                self.adjacency_given = False
        elif isinstance(adjacency, list):
            adjacency = np.array(adjacency).astype(np.float64)
            raw_ind, col_ind = np.nonzero(np.triu(adjacency))
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif isinstance(adjacency, np.ndarray):
            adjacency = adjacency.astype(np.float64)
            raw_ind, col_ind = np.nonzero(np.triu(adjacency))
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif scipy.sparse.isspmatrix(adjacency):
            raw_ind, col_ind = scipy.sparse.triu(adjacency).nonzero()
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
            linsearch = linsearch
            regularise = regularise

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
            linsearch=linsearch,
            full_return=full_return,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_crema(self, solution):
        if self.full_return:
            self.beta = solution[0]
            self.comput_time_crema = solution[1]
            self.n_steps_crema = solution[2]
            self.norm_seq_crema = solution[3]
            self.diff_seq_crema = solution[4]
            self.alfa_seq_crema = solution[5]
        else:
            self.beta = solution

    def _set_solved_problem_ecm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        if self.last_model == "ecm":
            self.x = self.r_xy[: self.n_nodes]
            self.y = self.r_xy[self.n_nodes:]
        elif self.last_model == "ecm-new":
            self.x = np.exp(-self.r_xy[:self.n_nodes])
            self.y = np.exp(-self.r_xy[self.n_nodes:])

    def solve_tool(
        self,
        model,
        method,
        initial_guess=None,
        adjacency="cm-new",
        method_adjacency="newton",
        initial_guess_adjacency="random",
        max_steps=100,
        full_return=False,
        verbose=False,
        linsearch=True,
        tol=1e-8,
        eps=1e-8,
    ):
        """[summary]

        :param model: Available models are:

            - *cm*: solves UBCM respect to the parameters *x* of the loglikelihood function, it works for uweighted undirected graphs [insert ref].
            - *cm-new*: differently from the *cm* option, *cm-new* considers the exponents of *x* as parameters [insert ref].
            - *ecm*: solves UECM respect to the parameters *x* and *y* of the loglikelihood function, it is conceived for weighted undirected graphs [insert ref].
            - *ecm-new*: differently from the *ecm* option, *ecm-new* considers the exponents of *x* and *y* as parameters [insert ref].
            - *crema*: solves CReMa for a weighted undirectd graphs. In order to compute beta parameters, it requires information about the binary structure of the network. These can be provided by the user by using *adjacency* paramenter.
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
        :param adjacency: Adjacency can be a binary method (defaults is *cm-new*) or an adjacency matrix.
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
        if model in ["cm", "cm-new", "ecm", "ecm-new"]:
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
        elif model in ["crema", "crema-sparse"]:
            self._solve_problem_crema(
                initial_guess=initial_guess,
                model=model,
                adjacency=adjacency,
                method=method,
                method_adjacency=method_adjacency,
                initial_guess_adjacency=initial_guess_adjacency,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )

    def ensemble_sampler(self, n, cpu_n=2, output_dir="sample/", seed=10):
        # al momento funziona solo sull'ultimo problema risolto

        # create the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # compute the sample

        # seed specification
        np.random.seed(seed)
        s = [np.random.randint(0, 1000000) for i in range(n)]

        if self.last_model in ["cm", "cm_new"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_cm_graph(
                    outfile_name=item,
                    x=self.x,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["ecm", "ecm_new"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_ecm_graph(
                    outfile_name=item,
                    x=self.x,
                    y=self.y,
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
                    eg.ensemble_sampler_crema_ecm_det_graph(
                        outfile_name=item,
                        beta=self.beta,
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
                    eg.ensemble_sampler_crema_ecm_prob_graph(
                        outfile_name=item,
                        beta=self.beta,
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
                    eg.ensemble_sampler_crema_sparse_ecm_prob_graph(
                        outfile_name=item,
                        beta=self.beta,
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1

        else:
            raise ValueError("insert a model")
