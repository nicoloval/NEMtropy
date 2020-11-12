import numpy as np
import os
import scipy.sparse
from numba import jit
import time
from Undirected_new import *
import random
import itertools
# import pathos.multiprocessing as mp
import multiprocessing as mp


def degree(a):
    """returns matrix A out degrees

    :param a: numpy.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1


def strength(a):
    """returns matrix A out degrees

    :param a: numpy.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 1).A1


def pmatrix_cm(x, args):
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
                f[i] -=  c[i] * c[j] * (x[j] / (1 + x[i] * x[j]))
    return f


@jit(nopython=True)
def loglikelihood_hessian_cm(x, args):
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
                        aux_f += ((x[h] * x[h]) / (aux * aux)) * c[i] * (c[h] - 1)
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
def iterative_CReAMa(beta, args):
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


@jit(nopython=True)
def iterative_CReAMa_sparse(beta, args):
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros_like(s, dtype=np.float64)
    x = adj[0]

    for i in np.arange(n):
        for j in np.arange(n):
            if i != j:
                aux = x[i] * x[j]
                aux_value = aux / (1 + aux)
                if aux_value > 0:
                    f[i] -= aux_value / (1 + (beta[j] / beta[i]))
    for i in np.arange(n):
        if s[i] != 0:
            f[i] = f[i] / s[i]
    return f


@jit(nopython=True)
def loglikelihood_CReAMa(beta, args):
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


@jit(nopython=True)
def loglikelihood_CReAMa_sparse(beta, args):
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
def loglikelihood_prime_CReAMa(beta, args):
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
def loglikelihood_prime_CReAMa_sparse(beta, args):
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


@jit(nopython=True)
def loglikelihood_hessian_CReAMa(beta, args):
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
def loglikelihood_hessian_CReAMa_sparse(beta, args):
    s = args[0]
    adj = args[1]
    n = len(s)
    f = np.zeros(shape=(n, n), dtype=np.float64)
    x = adj[0]
    for i in np.arange(n):
        for j in np.arange(0, i):
            aux = x[i] * x[j]
            aux_value = aux / (1 + aux)
            if aux_value > 0:
                aux = -aux_value / ((beta[i] + beta[j]) ** 2)
                f[i, j] = aux
                f[j, i] = aux
                f[i, i] += aux
                f[j, j] += aux
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_CReAMa(beta, args):
    s = args[0]
    adj = args[1]
    n = len(s)
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
def loglikelihood_hessian_diag_CReAMa_sparse(beta, args):
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


@jit(nopython=True)
def iterative_ecm(sol, args):
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
    eps=1e-16,
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
            if regularise == True:
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

        if linsearch:
            alfa1 = 1
            X = (x, dx, beta, alfa1, f)
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
            print("fun = {}".format(f))
            print("dx = {}".format(dx))
            print("x = {}".format(x))
            print("|f(x)| = {}".format(norm))
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


@jit(forceobj=True)
def linsearch_fun_CReAMa(X, args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]

    i = 0
    s_old = step_fun(x)
    while (
        sufficient_decrease_condition(
            s_old, step_fun(x + alfa * dx), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1

    return alfa


@jit(forceobj=True)
def linsearch_fun_CM_new(X, args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]

    i = 0
    s_old = step_fun(x)
    while (
        sufficient_decrease_condition(
            s_old, step_fun(x + alfa * dx), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1
    # print(alfa)
    return alfa


@jit(forceobj=True)
def linsearch_fun_CM(X, args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]

    # print(alfa)

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    for a in alfa0:
        if a >= 0:
            alfa = min(alfa, a)
    # print(alfa)
    i = 0
    s_old = step_fun(x)
    while (
        sufficient_decrease_condition(
            s_old, step_fun(x + alfa * dx), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1
    # print(alfa)
    return alfa


@jit(forceobj=True)
def linsearch_fun_ECM_new(X, args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]

    nnn = int(len(x) / 2)
    while True:
        ind_min_beta = (x[nnn:] + alfa * dx[nnn:]).argsort()[:2]
        if (
            np.sum(x[nnn:][ind_min_beta] + alfa * dx[nnn:][ind_min_beta])
            > 1e-14
        ):
            break
        else:
            alfa *= beta

    i = 0
    s_old = step_fun(x)
    while (
        sufficient_decrease_condition(
            s_old, step_fun(x + alfa * dx), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1

    return alfa


@jit(forceobj=True)
def linsearch_fun_ECM(X, args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]

    eps2 = 1e-2
    alfa0 = (eps2 - 1) * x / dx
    for a in alfa0:
        if a >= 0:
            alfa = min(alfa, a)

    nnn = int(len(x) / 2)
    while True:
        ind_max_y = (x[nnn:] + alfa * dx[nnn:]).argsort()[-2:][::-1]
        if np.prod(x[nnn:][ind_max_y] + alfa * dx[nnn:][ind_max_y]) < 1:
            break
        else:
            alfa *= beta

    i = 0
    s_old = step_fun(x)
    while (
        sufficient_decrease_condition(
            s_old, step_fun(x + alfa * dx), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1

    return alfa


@jit(forceobj=True)
def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-04, c2=0.9
):
    """return boolean indicator if upper wolfe condition are respected."""
    sup = f_old + c1 * alpha * grad_f @ p.T

    return bool(f_new < sup)


def hessian_regulariser_function(B, eps):
    """Trasform input matrix in a positive defined matrix
    by adding positive quantites to the main diagonal.

    input:
    B: np.ndarray, Hessian matrix
    eps: float, positive quantity to add

    output:
    Bf: np.ndarray, Regularised Hessian

    """

    B = (B + B.transpose()) * 0.5  # symmetrization
    Bf = B + np.identity(B.shape[0]) * eps

    return Bf


def hessian_regulariser_function_eigen_based(B, eps):
    """Trasform input matrix in a positive defined matrix
    by regularising eigenvalues.

    input:
    B: np.ndarray, Hessian matrix
    eps: float, positive quantity to add

    output:
    Bf: np.ndarray, Regularised Hessian

    """

    B = (B + B.transpose()) * 0.5  # symmetrization
    l, e = scipy.linalg.eigh(B)
    ll = np.array([0 if li > eps else eps - li for li in l])
    Bf = e @ (np.diag(ll) + np.diag(l)) @ e.transpose()

    return Bf


@jit(nopython=True)
def expected_degree_cm(sol):
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
def expected_strength_CReAMa(sol, adj):
    ex_s = np.zeros_like(sol, dtype=np.float64)
    n = len(sol)
    raw_ind = adj[0]
    col_ind = adj[1]
    weigths_val = adj[2]

    for i, j, w in zip(raw_ind, col_ind, weigths_val):
        aux = w / (sol[i] + sol[j])
        ex_s[i] += aux
        ex_s[j] += aux
    return ex_s


@jit(nopython=True)
def expected_strength_CReAMa_sparse(sol, adj):
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


@jit(nopython=True)
def beta_ecm(b):
    return(np.log(1-(b)))


@jit(nopython=True)
def alfa_ecm( chi, b):
    return((np.exp(-chi)*(1.0 - np.exp(-b)))/(np.exp(-b)))


def edgelist_from_edgelist(edgelist):
    """
    Creates a new edgelist with the indexes of the nodes instead of the names.
    Returns also two dictionaries that keep track of the nodes.
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
    def __init__(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
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
                ):  # Cast it to a numpy array: if it is given as a list it should not be too large
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

                # self.edgelist, self.deg_seq = edgelist_from_adjacency(adjacency)
                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)
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
                self.n_edges = np.sum(self.dseq)
                self.is_initialized = True
                if self.n_nodes > 2000:
                    self.is_sparse = True

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
                    self.n_edges = np.sum(self.dseq)
                    self.is_initialized = True
                    if self.n_nodes > 2000:
                        self.is_sparse = True

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
                    if self.n_nodes > 2000:
                        self.is_sparse = True

    def set_adjacency_matrix(self, adjacency):
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequence):
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(degree_sequence=degree_sequence)

    def clean_edges(self):
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
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
    ):

        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self._initialize_problem(model, method)
        x0 = self.x0

        sol = solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser = self.hessian_regulariser,
            tol=tol,
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
        elif model in ["CReAMa", "CReAMA-sparse"]:
            self._set_solved_problem_CReAMa(solution)

    def degree_reduction(self):
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

        if model in ["cm", "cm-new"]:
            self._set_initial_guess_cm()
        elif model in ["ecm", "ecm-new"]:
            self._set_initial_guess_ecm()
        elif model in ["CReAMa", "CReAMa-sparse"]:
            self._set_initial_guess_CReAMa()


    def _set_initial_guess_cm(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.

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
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.r_x[self.r_dseq == 0] = 0

        self.x0 = self.r_x


    def _set_initial_guess_CReAMa(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.

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
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.x0 = self.beta


    def _set_initial_guess_ecm(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes] 
            self.y = self.initial_guess[self.n_nodes:] 
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths_minor":
                self.x = self.dseq.astype(float) / (
                    self.n_edges + 1
                )  # This +1 increases the stability of the solutions.
                self.y = (
                    self.strength_sequence.astype(float)
                    / self.strength_sequence.sum()
                )
            elif self.initial_guess == "strengths":
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
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.x[self.dseq == 0] = 0
        self.y[self.strength_sequence == 0] = 0

        self.x0 = np.concatenate((self.x, self.y))

    # DA SISTEMARE
    def solution_error(self):
        if self.last_model in ["cm", "cm-new", "CReAMa", "CReAMa-sparse"]:
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
                    ex_s = expected_strength_CReAMa_sparse(
                        self.beta, self.adjacency_CReAMa
                    )
                else:
                    ex_s = expected_strength_CReAMa(
                        self.beta, self.adjacency_CReAMa
                    )
                self.expected_stregth_seq = ex_s
                # error output
                self.error_strength = np.linalg.norm(
                    ex_s - self.strength_sequence, ord=np.inf
                )
                self.relative_error_strength = np.max(
                     (ex_s - self.strength_sequence) / (self.strength_sequence + np.exp(-100))
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
            self.expected_strength_seq = ex[self.n_nodes :]

            # error output
            self.error_degree = np.linalg.norm(self.expected_dseq - self.dseq,ord=np.inf)
            self.error_strength = np.linalg.norm(self.expected_strength_seq - self.strength_sequence,ord=np.inf)
            self.relative_error_strength = max(
                abs(
                    (self.strength_sequence - self.expected_strength_seq)/ self.strength_sequence
                    )
                )

            self.relative_error_degree = max(
                abs(
                    (self.dseq - self.expected_dseq)/ self.dseq
                    )
                )
            self.error = max(self.error_strength, self.error_degree)

    def _set_args(self, model):

        if model in ["CReAMa", "CReAMa-sparse"]:
            self.args = (
                self.strength_sequence,
                self.adjacency_CReAMa,
                self.nz_index,
            )
        elif model in ["cm", "cm-new"]:
            self.args = (self.r_dseq, self.r_multiplicity)
        elif model in ["ecm", "ecm-new"]:
            self.args = (self.dseq, self.strength_sequence)

    def _initialize_problem(self, model, method):

        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = "-"
        mod_met = mod_met.join([model, method])

        d_fun = {
            "cm-newton": lambda x: -loglikelihood_prime_cm(x, self.args),
            "cm-quasinewton": lambda x: -loglikelihood_prime_cm(x, self.args),
            "cm-fixed-point": lambda x: iterative_cm(x, self.args),
            "CReAMa-newton": lambda x: -loglikelihood_prime_CReAMa(
                x, self.args
            ),
            "CReAMa-quasinewton": lambda x: -loglikelihood_prime_CReAMa(
                x, self.args
            ),
            "CReAMa-fixed-point": lambda x: -iterative_CReAMa(x, self.args),
            "ecm-newton": lambda x: -loglikelihood_prime_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -loglikelihood_prime_ecm(
                x, self.args
            ),
            "ecm-fixed-point": lambda x: iterative_ecm(x, self.args),
            "CReAMa-sparse-newton": lambda x: -loglikelihood_prime_CReAMa_sparse(
                x, self.args
            ),
            "CReAMa-sparse-quasinewton": lambda x: -loglikelihood_prime_CReAMa_sparse(
                x, self.args
            ),
            "CReAMa-sparse-fixed-point": lambda x: -iterative_CReAMa_sparse(
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
            "CReAMa-newton": lambda x: -loglikelihood_hessian_CReAMa(
                x, self.args
            ),
            "CReAMa-quasinewton": lambda x: -loglikelihood_hessian_diag_CReAMa(
                x, self.args
            ),
            "CReAMa-fixed-point": None,
            "ecm-newton": lambda x: -loglikelihood_hessian_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -loglikelihood_hessian_diag_ecm(
                x, self.args
            ),
            "ecm-fixed-point": None,
            "CReAMa-sparse-newton": lambda x: -loglikelihood_hessian_CReAMa_sparse(
                x, self.args
            ),
            "CReAMa-sparse-quasinewton": lambda x: -loglikelihood_hessian_diag_CReAMa_sparse(
                x, self.args
            ),
            "CReAMa-sparse-fixed-point": None,
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
            "CReAMa-newton": lambda x: -loglikelihood_CReAMa(x, self.args),
            "CReAMa-quasinewton": lambda x: -loglikelihood_CReAMa(
                x, self.args
            ),
            "CReAMa-fixed-point": lambda x: -loglikelihood_CReAMa(
                x, self.args
            ),
            "ecm-newton": lambda x: -loglikelihood_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -loglikelihood_ecm(x, self.args),
            "ecm-fixed-point": lambda x: -loglikelihood_ecm(x, self.args),
            "CReAMa-sparse-newton": lambda x: -loglikelihood_CReAMa_sparse(
                x, self.args
            ),
            "CReAMa-sparse-quasinewton": lambda x: -loglikelihood_CReAMa_sparse(
                x, self.args
            ),
            "CReAMa-sparse-fixed-point": lambda x: -loglikelihood_CReAMa_sparse(
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

        self.args_lins = (self.step_fun,)

        lins_fun = {
            "cm": lambda x: linsearch_fun_CM(x, self.args_lins),
            "CReAMa": lambda x: linsearch_fun_CReAMa(x, self.args_lins),
            "CReAMa-sparse": lambda x: linsearch_fun_CReAMa(x, self.args_lins),
            "ecm": lambda x: linsearch_fun_ECM(x, self.args_lins),
            "cm-new": lambda x: linsearch_fun_CM_new(x, self.args_lins),
            "ecm-new": lambda x: linsearch_fun_ECM_new(x, self.args_lins),
        }

        self.fun_linsearch = lins_fun[model]
        
        hess_reg = {
            "cm" : hessian_regulariser_function_eigen_based,
            "cm-new" : hessian_regulariser_function,
            "ecm" : hessian_regulariser_function_eigen_based,
            "ecm-new" : hessian_regulariser_function,
            "CReAMa" : hessian_regulariser_function,
            "CReAMa-sparse" : hessian_regulariser_function,
        }
        
        self.hessian_regulariser = hess_reg[model]

    def _solve_problem_CReAMa(
        self,
        initial_guess=None,
        model="CReAMa",
        adjacency="cm",
        method="quasinewton",
        max_steps=100,
        tol=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
    ):
        if not isinstance(adjacency, (list, np.ndarray, str)) and (
            not scipy.sparse.isspmatrix(adjacency)
        ):
            raise ValueError("adjacency must be a matrix or a method")
        elif isinstance(adjacency, str):
            
            # aggiungere check sul modello passato per l'adjacency matrix
            
            self._solve_problem(
                initial_guess=initial_guess,
                model=adjacency,
                method=method,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                regularise=regularise
            )
            if self.is_sparse:
                self.adjacency_CReAMa = (self.x,)
                self.adjacency_given = False
            else:
                pmatrix = self.fun_pmatrix(self.x)
                raw_ind, col_ind = np.nonzero(np.triu(pmatrix))
                raw_ind = raw_ind.astype(np.int64)
                col_ind = col_ind.astype(np.int64)
                weigths_value = pmatrix[raw_ind, col_ind]
                self.adjacency_CReAMa = (raw_ind, col_ind, weigths_value)
                self.is_sparse = False
                self.adjacency_given = False
        elif isinstance(adjacency, list):
            adjacency = np.array(adjacency).astype(np.float64)
            raw_ind, col_ind = np.nonzero(np.triu(adjacency))
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_CReAMa = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif isinstance(adjacency, np.ndarray):
            adjacency = adjacency.astype(np.float64)
            raw_ind, col_ind = np.nonzero(np.triu(adjacency))
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_CReAMa = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif scipy.sparse.isspmatrix(adjacency):
            raw_ind, col_ind = scipy.sparse.triu(adjacency).nonzero()
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = (adjacency[raw_ind, col_ind].A1).astype(np.float64)
            self.adjacency_CReAMa = (raw_ind, col_ind, weigths_value)
            self.is_sparse = True
            self.adjacency_given = True

        if self.is_sparse:
            self.last_model = "CReAMa-sparse"
        else:
            self.last_model = model
            linsearch=linsearch,
            regularise=regularise
        
        self.regularise = regularise
        self.full_return = full_return
        self.initial_guess = "strengths"
        self._initialize_problem(self.last_model, method)
        x0 = self.x0

        sol = solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser = self.hessian_regulariser,
            tol=tol,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=regularise,
            linsearch=linsearch,
            full_return=full_return,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_CReAMa(self, solution):
        if self.full_return:
            self.beta = solution[0]
            self.comput_time_creama = solution[1]
            self.n_steps_creama = solution[2]
            self.norm_seq_creama = solution[3]
            self.diff_seq_creama = solution[4]
            self.alfa_seq_creama = solution[5]

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
            self.y = self.r_xy[self.n_nodes :]
        elif self.last_model == "ecm-new":
            self.x = np.exp(-self.r_xy[: self.n_nodes])
            self.y = np.exp(-self.r_xy[self.n_nodes :])


    def solve_tool(
        self,
        model,
        method,
        initial_guess=None,
        adjacency=None,
        max_steps=100,
        full_return=False,
        verbose=False,
        tol=1e-8,
    ):
        """function to switch around the various problems"""
        # TODO: aggiungere tutti i metodi
        if model in ["cm", "cm-new", "ecm", "ecm-new"]:
            self._solve_problem(
                initial_guess=initial_guess,
                model=model,
                method=method,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                tol=tol,
            )
        elif model in ["CReAMa","CReAMa-sparse"]:
            self._solve_problem_CReAMa(
                initial_guess=initial_guess,
                model=model,
                adjacency=adjacency,
                method=method,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                tol=tol,
            )
        

    def _solve_problem_ECM(
        self,
        initial_guess=None,
        model="CReAMa",
        adjacency="cm",
        method="quasinewton",
        max_steps=100,
        tol=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
    ):      
        if adjacency in ["cm", "cm-new"]:
            self._solve_problem_CReAMa(
                initial_guess=initial_guess,
                model="CReAMa",
                adjacency=adjacency,
                method=method,
                max_steps=max_steps,
                tol=tol,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                regularise=regularise,
            )
        else:
            raise ValueError("Adjacency value can be 'cm' or 'cm-new'.")
        
        if full_return:
            sol = (np.concatenate([self.x, self.beta]),
                   (self.comput_time, self.comput_time_creama),
                   (self.n_steps, self.n_steps_creama),
                   (self.norm_seq, self.norm_seq_creama),
                   (self.diff_seq, self.diff_seq_creama),
                   (self.alfa_seq, self.alfa_seq_creama))
        else:
            sol = np.concatenate(self.x, self.beta)
        
        self.last_model = "ecm-two-steps"
        self._set_solved_problem_ecm(sol)
        
    
    def ensemble_sampler(self, n, output_dir="sample/", seed=10):
        # al momento funziona solo sull'ultimo problema risolto
        # unico input possibile e' la cartella dove salvare i samples
        # ed il numero di samples

        if self.last_model in ["cm", "cm_new"]:
            # self.x
            # joblib 
            self.p_ij = lambda inds : p_ij_cm(inds, (self.x)),
            # multiprocessing
            # self.is_a_link = self.is_a_link_cm
        elif self.last_model in ["ecm", "ecm_new"]:
            # self.x
            # self.y
            self.p_ij = lambda inds : p_ij_ecm(inds, (self.x, self.y)),
            self.q_ij = lambda inds : q_ij_ecm(inds, (self.x, self.y)),
        elif self.last_model in ["ecm-two-steps"]:
            self.chi = None
            # self.b_CReM
        elif self.last_model in ["CReAMa", "CReAMa-sparse"]:
            self.beta = None
        else: 
            raise ValueError("insert a model")

        # create the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # compute the sample
        np.random.seed(seed)
        s = [np.random.randint(0,1000000) for i in range(n)]

        if self.last_model in ["cm", "cm_new"]:
            iter_files = iter(output_dir + "{}.txt".format(i) for i in range(n))
            # itertools.starmap(self.ensemble_sampler_binary_single_graph, iter_files)
            i = 0
            for item in iter_files:
                self.ensemble_sampler_binary_single_graph(item, seed=s[i])
                i += 1

        elif self.last_model in ["ecm", "ecm_new", "ecm-two-steps", "CReAMa", "CReAMa-sparse"]:
            place_holder=None
            # self.ensemble_sampler_weighted(n=n, output_dir=output_dir, seed)
        else: 
            raise ValueError("insert a model")


    @jit(nopython=True)
    def is_a_link(self, ind):
        p = random.random()
        p_ensemble = self.p_ij(ind)
        return p < p_ensemble


    @jit(nopython=True)
    def p_ij_cm(inds, args):
        i, j = inds
        x = args[0]
        xij = x[i]*x[j]
        return xij/(1 + xij)


    def ensemble_sampler_binary_single_graph(self, outfile_name,seed=None):
        # produce and write a single undirected binary graph
        if seed is not None:
            np.random.seed(seed)

        cpu_n = 2  #TODO: deve essere in input
        x = self.x
        inds = range(len(x))

        # put together inputs for pool 
        # iter_ = itertools.product(zip(inds,x), zip(inds,x))
        # print(list(zip(inds, x)))
        iter_ = iter(((i, xi),(j, xj), np.random.randint(0,1000000)) for i,xi in zip(inds,x) for j,xj in zip(inds,x) if i<j) 

        # compute existing edges
        with mp.Pool(processes=cpu_n) as pool:
            edges_list = pool.starmap(is_a_link_cm, iter_)

        # removing None
        edges_list[:] = (value for value in edges_list if value is not None)

        # debug
        # print(edges_list)

        # edgelist writing
        with open(outfile_name, "w") as outfile:
            outfile.write(
                "".join(
                    "{} {}\n".format(str(i),str(j)) for (i,j) in edges_list
                    )
                )

        return outfile_name 
                

    def ensemble_sampler_weighted(self, output_name):
        return None


def is_a_link_cm(args_1, args_2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    (i, xi) = args_1
    (j, xj) = args_2
    p = np.random.random()
    xij = xi*xj
    p_ensemble =  xij/(1 + xij)
    if p < p_ensemble:
        return (i, j)
    else:
        return None

