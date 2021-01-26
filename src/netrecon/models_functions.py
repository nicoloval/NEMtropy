import numpy as np
import scipy.sparse
from numba import jit
from . import solver_functions as sof
# Stops Numba Warning for experimental feature
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings

warnings.simplefilter(
    action='ignore',
    category=NumbaExperimentalFeatureWarning)

# CREMA functions
# ---------------


@jit(nopython=True)
def iterative_crema(beta, args):
    """Returns the next CReMa iterative step for the fixed-point method.
    The DBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Previous iterative step.
    :type beta: numpy.ndarray
    :param args: Out and in strengths sequences,
        adjacency binary/probability matrix,
        and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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


@jit(nopython=True, parallel=True)
def iterative_crema_sparse(beta, args):
    """Returns the next CReMa iterative step for the fixed-point method.
    The DBCM pmatrix is computed inside the function.
    Alternative version not in use.

    :param beta: Previous iterative step.
    :type beta: numpy.ndarray
    :param args: Out and in strengths sequences,
        adjacency matrix, and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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
def iterative_crema_sparse_2(beta, args):
    """Returns the next CReMa iterative step for the fixed-point method.
    The DBCM pmatrix is computed inside the function.
    Alternative version not in use.

    :param beta: Previous iterative step.
    :type beta: numpy.ndarray
    :param args: Out and in strengths sequences,
        adjacency matrix, and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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


@jit(nopython=True)
def loglikelihood_crema(beta, args):
    """Returns CReMa loglikelihood function evaluated in beta.
    The DBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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
    """Returns CReMa loglikelihood function evaluated  in beta.
    The DBCM pmatrix is computed inside the function.
    Sparse initialisation version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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
    The DBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient value.
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


@jit(nopython=True, parallel=True)
def loglikelihood_prime_crema_sparse(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
    The DBCM pmatrix is pre-computed and explicitly passed.
    Sparse initialization version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
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
def loglikelihood_prime_crema_sparse_2(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
    The DBCM pmatrix is computed inside the function.
    Sparse initialization version.
    Alternative version not used.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient value.
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


@jit(nopython=True)
def loglikelihood_hessian_crema(beta, args):
    """Returns CReMa loglikelihood hessian function evaluated in beta.
    The DBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
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
    """Returns the diagonal of CReMa loglikelihood hessian function
    evaluated in beta. The DBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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


@jit(nopython=True, parallel=True)
def loglikelihood_hessian_diag_crema_sparse(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian function
    evaluated in beta. The DBCM pmatrix is computed inside the function.
    Sparse initialization version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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
def loglikelihood_hessian_diag_crema_sparse_2(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian function
    evaluated in beta. The DBCM pmatrix is pre-computed and explicitly passed.
    Sparse initialization version.
    Alternative version not in use.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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


@jit(nopython=True)
def linsearch_fun_crema(xx, args):
    """Linsearch function for CReMa newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f.
    :type xx: (numpy.ndarray, numpy.ndarray,
        float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (function, tuple)
    :return: Working alpha.
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
    cond = sof.sufficient_decrease_condition(s_old,
                                         -step_fun(x + alfa * dx,
                                                   arg_step_fun),
                                         alfa,
                                         f,
                                         dx)
    while ((not cond) and i < 50):
        alfa *= beta
        i += 1
        cond = sof.sufficient_decrease_condition(s_old,
                                             -step_fun(x + alfa * dx,
                                                       arg_step_fun),
                                             alfa,
                                             f,
                                             dx)
    return alfa


@jit(nopython=True)
def linsearch_fun_crema_fixed(xx):
    """Linsearch function for CReMa fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type xx: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        float, float, float)
    :return: Working alpha.
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
        while((not cond) and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa

# DBCM functions
# --------------

@jit(nopython=True)
def iterative_dcm(x, args):
    """Returns the next DBCM iterative step for the fixed-point method.

    :param x: Previous iterative step.
    :type x: numpy.ndarray
    :param args: Out and in strengths sequences, non zero out and in indices,
        and classes cardinalities sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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
def loglikelihood_dcm(x, args):
    """Returns DBCM loglikelihood function evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, non zero out and in indices,
        and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient.
        Out and in degrees sequences, non zero out and in indices,
        and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
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

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Tuple containing out and in degrees sequences,
        non zero out and in indices, and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
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
def loglikelihood_hessian_diag_dcm(x, args):
    """Returns the diagonal of DBCM loglikelihood hessian function
    evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Out and in degrees sequences, non zero out and in indices,
        and classes cardinality sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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
def pmatrix_dcm(x, args):
    """Computes and returns the probability matrix induced by DBCM.

    :param x: DBCM solution.
    :type x: numpy.ndarray
    :param args: Problem dimension, out and in indices of non-zero nodes.
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
def expected_out_degree_dcm(sol):
    """Expected out-degree after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected out-degree sequence.
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

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected in-degree sequence.
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
def linsearch_fun_DCM(xx, args):
    """Linsearch function for DBCM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
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
    cond = sof.sufficient_decrease_condition(s_old,
                                         -step_fun(x + alfa * dx,
                                                   arg_step_fun),
                                         alfa,
                                         f,
                                         dx)
    while ((not cond) and (i < 50)):
        alfa *= beta
        i += 1
        cond = sof.sufficient_decrease_condition(s_old,
                                             -step_fun(x + alfa * dx,
                                                       arg_step_fun),
                                             alfa,
                                             f,
                                             dx)
    return alfa


@jit(nopython=True)
def linsearch_fun_DCM_fixed(xx):
    """Linsearch function for DBCM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type xx: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: Working alpha.
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
        while((not cond) and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa

# DECM functions
# --------------

@jit(nopython=True)
def iterative_decm(x, args):
    """Returns the next iterative step for the DECM Model.

    :param : Previous iterative step.
    :type : numpy.ndarray
    :param args: Out and in degrees sequences, and out and in strengths
        sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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
def loglikelihood_decm(x, args):
    """Returns DECM loglikelihood function evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood value.
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

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient.
        Out and in degrees sequences, and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood gradient.
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
    """Returns the diagonal of DECM loglikelihood hessian function
    evaluated in x.

    :param x: Evaluating point *x*. 
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian.
        Out and in degrees sequences, and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian.
        Out and in degrees sequences, and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood hessian matrix.
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
def linsearch_fun_DECM(xx, args):
    """Linsearch function for DECM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
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
        cond = ((x[2 * nnn: 3 * nnn][ind_yout]
                + alfa * dx[2 * nnn: 3 * nnn][ind_yout])
                * (x[3 * nnn:][ind_yin]
                + alfa * dx[3 * nnn:][ind_yin]))
        if (cond) < 1:
            break
        else:
            alfa *= beta

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    cond = sof.sufficient_decrease_condition(s_old,
                                         -step_fun(x + alfa * dx,
                                                   arg_step_fun),
                                         alfa, f, dx)
    while ((not cond) and i < 50):
        alfa *= beta
        i += 1
        cond = sof.sufficient_decrease_condition(s_old,
                                             -step_fun(x + alfa * dx,
                                                       arg_step_fun),
                                             alfa, f, dx)
    return alfa


@jit(nopython=True)
def linsearch_fun_DECM_fixed(xx):
    """Linsearch function for DECM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: Working alpha.
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
        while((not cond) and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa




@jit(nopython=True)
def expected_decm(x):
    """Expected parameters after the DECM.
       It returns a concatenated array of out-degrees, in-degrees,
       out-strengths, in-strengths.

    :param x: DECM solution.
    :type x: numpy.ndarray
    :return: Expected parameters sequence.
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

# DBCM exponential functions
# --------------------------

@jit(nopython=True)
def linsearch_fun_DCM_new(X, args):
    """Linsearch function for DBCM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on DBCM exponential version.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
    :rtype: float
    """
    # TODO: change X to xx
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
        sof.sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        is False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_DCM_new_fixed(X):
    """Linsearch function for DBCM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on DBCM exponential version.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: Working alpha.
    :rtype: float
    """
    # TODO: change X to xx
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx, ord=2) < np.linalg.norm(dx_old, ord=2)
        while((not cond)
                and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx[dx != np.infty], ord=2) < \
                np.linalg.norm(dx_old[dx_old != np.infty], ord=2)
    return alfa

# DECM exponential functions
# --------------------------

@jit(nopython=True)
def linsearch_fun_DECM_new(X, args):
    """Linsearch function for DECM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on DECM exponential version.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type X: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
    :rtype: float
    """
    # TODO: change X to xx
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    # Mettere il check sulle y
    nnn = int(len(x) / 4)
    ind_yout = np.argmin(x[2 * nnn: 3 * nnn])
    ind_yin = np.argmin(x[3 * nnn:])
    tmp = x[2 * nnn: 3 * nnn][ind_yout] + x[3 * nnn:][ind_yin]
    while True:
        ind_yout = np.argmin(
            x[2 * nnn: 3 * nnn] + alfa * dx[2 * nnn: 3 * nnn]
        )
        ind_yin = np.argmin(x[3 * nnn:] + alfa * dx[3 * nnn:])
        cond = (x[2 * nnn: 3 * nnn][ind_yout]
                + alfa * dx[2 * nnn: 3 * nnn][ind_yout]) + \
            (x[3 * nnn:][ind_yin]
             + alfa * dx[3 * nnn:][ind_yin])
        if (cond) > tmp * 0.01:
            break
        else:
            alfa *= beta

    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sof.sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        is False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_DECM_new_fixed(X):
    """Linsearch function for DECM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on DECM exponential version.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type X: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: Working alpha.
    :rtype: float
    """
    # TODO: change X to xx
    x = X[0]
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    # Mettere il check sulle y
    nnn = int(len(x) / 4)
    ind_yout = np.argmin(x[2 * nnn: 3 * nnn])
    ind_yin = np.argmin(x[3 * nnn:])
    tmp = x[2 * nnn: 3 * nnn][ind_yout] + x[3 * nnn:][ind_yin]

    while True:
        ind_yout = np.argmin(
            x[2 * nnn: 3 * nnn] + alfa * dx[2 * nnn: 3 * nnn]
        )
        ind_yin = np.argmin(x[3 * nnn:] + alfa * dx[3 * nnn:])
        cond = (x[2 * nnn: 3 * nnn][ind_yout]
                + alfa * dx[2 * nnn: 3 * nnn][ind_yout]) + \
            (x[3 * nnn:][ind_yin]
             + alfa * dx[3 * nnn:][ind_yin])
        if (cond) > tmp * 0.01:
            break
        else:
            alfa *= beta

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx, ord=2) < \
            np.linalg.norm(dx_old, ord=2)
        while((not cond)
              and kk < 50):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx[dx != np.infty], ord=2) < \
                np.linalg.norm(dx_old[dx_old != np.infty], ord=2)

    return alfa
