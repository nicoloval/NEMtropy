import numpy as np
import scipy.sparse
from numba import jit
import time
from . import Directed_graph_Class as sample
# Stops Numba Warning for experimental feature
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings

warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

# dcm functions


@jit(nopython=True)
def iterative_dcm_new_bis(theta, args):
    """run only on non-zero indexes"""
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n = len(k_out)
    nz_index_out = args[2]
    nz_index_in = args[3]
    # nz_index_out = range(n)
    # nz_index_in = range(n)
    c = args[4]

    f = np.zeros(2 * n, dtype=np.float64)
    x = np.exp(-theta)

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
    ff = -np.log(
        np.array(
            [tmp[i] / f[i] if tmp[i] != 0 else -np.infty for i in range(2 * n)]
        )
    )
    # ff = -np.log(tmp/f)

    return ff


@jit(nopython=True)
def iterative_dcm_new(theta, args):
    """run on all indexes"""
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n = len(k_out)
    # nz_index_out = args[2]
    # nz_index_in = args[3]
    nz_index_out = range(n)
    nz_index_in = range(n)
    c = args[4]

    f = np.zeros(2 * n, dtype=np.float64)
    x = np.exp(-theta)

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
    ff = -np.log(tmp / f)

    return ff


@jit(nopython=True)
def loglikelihood_dcm_new(theta, args):
    """loglikelihood function for dcm
    reduced, not-zero indexes
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    # n = len(k_out)
    # nz_index_out = range(n)
    # nz_index_in = range(n)

    c = args[4]
    n = len(k_out)

    f = 0
    x = np.exp(-theta)

    for i in nz_index_out:
        f -= c[i] * k_out[i] * theta[i]
        for j in nz_index_in:
            if i != j:
                f -= c[i] * c[j] * np.log(1 + np.exp(-theta[i] - theta[n + j]))
            else:
                f -= (
                    c[i]
                    * (c[i] - 1)
                    * np.log(1 + np.exp(-theta[i] - theta[n + j]))
                )

    for j in nz_index_in:
        f -= c[j] * k_in[j] * theta[j + n]

    return f


@jit(nopython=True)
def loglikelihood_prime_dcm_new(theta, args):
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_in)

    f = np.zeros(2 * n)
    x = np.exp(-theta)

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
        # f[i] = x[i]*fx - k_out[i]
        f[i] = x[i] * fx - c[i] * k_out[i]

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
        # f[j+n] = fy*x[j+n] - k_in[j]
        f[j + n] = fy * x[j + n] - c[j] * k_in[j]

    return f


@jit(nopython=True)
def loglikelihood_hessian_dcm_new(theta, args):
    k_out = args[0]
    k_in = args[1]
    nz_out_index = args[2]
    nz_in_index = args[3]
    c = args[4]
    n = len(k_out)

    out = np.zeros((2 * n, 2 * n))  # hessian matrix
    x = np.exp(-theta)

    for h in nz_out_index:
        tmp_sum = 0
        for i in nz_in_index:
            if i == h:
                const = c[h] * (c[h] - 1)
                # const = (c[h] - 1)
            else:
                const = c[h] * c[i]
                # const = c[i]

            tmp = x[i + n] * x[h]
            tmp_sum += const * (tmp) / (1 + tmp) ** 2
            out[h, i + n] = -const * tmp / (1 + tmp) ** 2
        out[h, h] = -tmp_sum

    for i in nz_in_index:
        tmp_sum = 0
        for h in nz_out_index:
            if i == h:
                const = c[h] * (c[h] - 1)
                # const = (c[i] - 1)
            else:
                const = c[h] * c[i]
                # const = c[h]

            tmp = x[h] * x[i + n]
            tmp_sum += const * (tmp) / (1 + tmp) ** 2
            out[i + n, h] = -const * tmp / (1 + tmp) ** 2
        out[i + n, i + n] = -tmp_sum
    return out


@jit(nopython=True)
def loglikelihood_hessian_diag_dcm_new(theta, args):
    # problem fixed paprameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_in)

    f = -np.zeros(2 * n)
    x = np.exp(-theta)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i != j:
                const = c[i] * c[j]
                # const = c[j]
            else:
                const = c[i] * (c[j] - 1)
                # const = (c[i] - 1)

            tmp = x[j + n] * x[i]
            fx += const * (tmp) / (1 + tmp) ** 2
        # original prime
        f[i] = -fx

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i != j:
                const = c[i] * c[j]
                # const = c[i]
            else:
                const = c[i] * (c[j] - 1)
                # const = (c[j] - 1)

            tmp = x[i] * x[j + n]
            fy += const * (tmp) / (1 + tmp) ** 2
        # original prime
        f[j + n] = -fy

    # f[f == 0] = 1

    return f


@jit(nopython=True)
def expected_out_degree_dcm_new(sol):
    n = int(len(sol) / 2)
    ex_k = np.zeros(n, dtype=np.float64)

    for i in np.arange(n):
        for j in np.arange(n):
            if i != j:
                aux = np.exp(-sol[i]) * np.exp(-sol[j])
                ex_k[i] += aux / (1 + aux)
    return ex_k


@jit(nopython=True)
def expected_in_degree_dcm_new(theta):
    sol = np.exp(-theta)
    n = int(len(sol) / 2)
    a_out = sol[:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += a_in[i] * a_out[j] / (1 + a_in[i] * a_out[j])

    return k


@jit(nopython=True)
def expected_decm_new(theta):
    """"""
    # casadi MX function calculation
    x = np.exp(-theta)
    n = int(len(x) / 4)
    f = np.zeros_like(x, np.float64)

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


@jit(nopython=True)
def linsearch_fun_DCM_new(X, args):
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
        sample.sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_DCM_new_fixed(X):
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx, ord = 2) < np.linalg.norm(dx_old, ord = 2)
        while(
            cond == False
            and kk<50
            ):
            alfa *= beta
            kk +=1
            cond = np.linalg.norm(alfa*dx[dx!=np.infty], ord = 2) < np.linalg.norm(dx_old[dx_old!=np.infty], ord = 2)
    return alfa


# decm functions


@jit(nopython=True)
def loglikelihood_decm_new(x, args):
    """not reduced"""
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    f = 0
    for i in range(n):
        if k_out[i]:
            f -= k_out[i] * x[i]
        if k_in[i]:
            f -= k_in[i] * x[i + n]
        if s_out[i]:
            f -= s_out[i] * (x[i + 2 * n])
        if s_in[i]:
            f -= s_in[i] * (x[i + 3 * n])
        for j in range(n):
            if i != j:
                tmp = np.exp(-x[i + 2 * n] - x[j + 3 * n])
                f -= np.log(1 + np.exp(-x[i] - x[j + n]) * tmp / (1 - tmp))

    return f


# @jit(nopython=True)
def loglikelihood_prime_decm_new(theta, args):
    """not reduced"""
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    x = np.exp(-theta)

    f = np.zeros(4 * n)
    for i in range(n):
        fa_out = 0
        fb_out = 0
        fa_in = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp0 = x[i + 2 * n] * x[j + 3 * n]
                tmp1 = x[i] * x[j + n]
                fa_out += tmp0 * tmp1 / (1 - tmp0 + tmp0 * tmp1)
                fb_out += tmp0 * tmp1 / ((1 - tmp0 + tmp0 * tmp1) * (1 - tmp0))

                tmp0 = x[j + 2 * n] * x[i + 3 * n]
                tmp1 = x[j] * x[i + n]
                fa_in += tmp0 * tmp1 / (1 - tmp0 + tmp0 * tmp1)
                fb_in += tmp0 * tmp1 / ((1 - tmp0 + tmp0 * tmp1) * (1 - tmp0))

        f[i] = -k_out[i] + fa_out
        f[i + 2 * n] = -s_out[i] + fb_out
        f[i + n] = -k_in[i] + fa_in
        f[i + 3 * n] = -s_in[i] + fb_in

    return f


@jit(nopython=True)
def loglikelihood_hessian_decm_new(theta, args):

    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]

    n = len(k_out)
    f = np.zeros((n * 4, n * 4))
    x = np.exp(-theta)

    a_out = x[:n]
    a_in = x[n : 2 * n]
    b_out = x[2 * n : 3 * n]
    b_in = x[3 * n :]

    for i in range(n):
        for j in range(n):
            if j != i:
                tmp0 = a_out[i] * a_in[j]
                tmp1 = b_out[i] * b_in[j]
                # diagonal elements
                f[i, i] -= (
                    (1 - tmp1) * tmp0 * tmp1 / ((1 + (-1 + tmp0) * tmp1) ** 2)
                )
                f[i + 2 * n, i + 2 * n] -= (
                    tmp0
                    * tmp1
                    * (1 + (tmp0 - 1) * (tmp1 ** 2))
                    / ((1 - tmp1) * (1 + (tmp0 - 1) * tmp1)) ** 2
                )
                # out of diagonal
                f[i, j + n] = (
                    -(1 - tmp1) * tmp0 * tmp1 / (1 + (tmp0 - 1) * tmp1) ** 2
                )
                f[j + n, i] = f[i, j + n]
                f[i, i + 2 * n] -= tmp0 * tmp1 / (1 + (tmp0 - 1) * tmp1) ** 2
                f[i + 2 * n, i] = f[i, i + 2 * n]
                f[i, j + 3 * n] = -tmp0 * tmp1 / (1 + (tmp0 - 1) * tmp1) ** 2
                f[j + 3 * n, i] = f[i, j + 3 * n]
                f[i + 2 * n, j + 3 * n] = (
                    -tmp0
                    * tmp1
                    * (1 + (tmp0 - 1) * tmp1 ** 2)
                    / (((1 - tmp1) ** 2) * ((1 + (tmp0 - 1) * tmp1) ** 2))
                )
                f[j + 3 * n, i + 2 * n] = f[i + 2 * n, j + 3 * n]

                tmp0 = a_out[j] * a_in[i]
                tmp1 = b_out[j] * b_in[i]
                # diagonal elements
                f[i + n, i + n] -= (
                    (1 - tmp1) * tmp0 * tmp1 / (1 + (tmp0 - 1) * tmp1) ** 2
                )
                f[i + 3 * n, i + 3 * n] -= (
                    tmp0
                    * tmp1
                    * (1 + (tmp0 - 1) * tmp1 ** 2)
                    / (((1 - tmp1) ** 2) * ((1 + (tmp0 - 1) * tmp1) ** 2))
                )
                # out of diagonal
                f[i + n, j + 2 * n] = (
                    -tmp0 * tmp1 / (1 + (tmp0 - 1) * tmp1) ** 2
                )
                f[j + 2 * n, i + n] = f[i + n, j + 2 * n]
                f[i + n, i + 3 * n] -= (
                    tmp0 * tmp1 / (1 + (tmp0 - 1) * tmp1) ** 2
                )
                f[i + 3 * n, i + n] = f[i + n, i + 3 * n]

    f = f
    for i in range(4 * n):
        for j in range(4 * n):
            if np.isnan(f[i, j]):
                f[i, j] = 0

    # print(np.isnan(f))
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_decm_new(theta, args):

    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]

    n = len(k_out)
    f = np.zeros(n * 4)
    x = np.exp(-theta)

    a_out = x[:n]
    a_in = x[n : 2 * n]
    b_out = x[2 * n : 3 * n]
    b_in = x[3 * n :]

    for i in range(n):
        for j in range(n):
            if j != i:
                tmp0 = a_out[i] * a_in[j]
                tmp1 = b_out[i] * b_in[j]
                # diagonal elements
                f[i] -= (
                    (1 - tmp1) * tmp0 * tmp1 / ((1 - tmp1 + tmp0 * tmp1) ** 2)
                )
                f[i + 2 * n] -= (
                    tmp0
                    * tmp1
                    * (1 - tmp1 ** 2 + tmp0 * (tmp1 ** 2))
                    / ((1 - tmp1) * (1 - tmp1 + tmp0 * tmp1)) ** 2
                )

                tmp0 = a_out[j] * a_in[i]
                tmp1 = b_out[j] * b_in[i]
                # diagonal elements
                f[i + n] -= (
                    (1 - tmp1) * tmp0 * tmp1 / (1 - tmp1 + tmp0 * tmp1) ** 2
                )
                f[i + 3 * n] -= (
                    tmp0
                    * tmp1
                    * (1 - tmp1 ** 2 + tmp0 * tmp1 ** 2)
                    / (((1 - tmp1) ** 2) * ((1 - tmp1 + tmp0 * tmp1) ** 2))
                )

    f = f

    return f


@jit(nopython=True)
def iterative_decm_new(theta, args):
    """not reduced"""
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    x = np.exp(-theta)

    f = np.zeros(4 * n)
    for i in range(n):
        fa_out = 0
        fb_out = 0
        fa_in = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp0 = x[i] * x[j + n]
                tmp1 = x[i + 2 * n] * x[j + 3 * n]

                fa_out += x[j + n] * tmp1 / (1 - tmp1 + tmp0 * tmp1)

                fb_out += (
                    tmp0
                    * x[j + 3 * n]
                    / ((1 - tmp1) * (1 - tmp1 + tmp0 * tmp1))
                )

                tmp0 = x[j] * x[i + n]
                tmp1 = x[j + 2 * n] * x[i + 3 * n]

                fa_in += x[j + n] * tmp1 / (1 - tmp1 + tmp0 * tmp1)

                fb_in += (
                    tmp0
                    * x[j + 3 * n]
                    / ((1 - tmp1) * (1 - tmp1 + tmp0 * tmp1))
                )

        if k_out[i]:
            f[i] = -np.log(k_out[i] / fa_out)
        else:
            f[i] = 1e3

        if s_out[i]:
            f[i + 2 * n] = -np.log(s_out[i] / fb_out)
        else:
            f[i + 2 * n] = 1e3

        if k_in[i]:
            f[i + n] = -np.log(k_in[i] / fa_in)
        else:
            f[i + n] = 1e3

        if s_in[i]:
            f[i + 3 * n] = -np.log(s_in[i] / fb_in)
        else:
            f[i + 3 * n] = 1e3

    return f


@jit(nopython=True)
def iterative_decm_new_old(theta, args):
    """not reduced"""
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]
    n = len(k_out)

    x = np.exp(-theta)

    f = np.zeros(4 * n)
    for i in range(n):
        fa_out = 0
        fb_out = 0
        fa_in = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp0 = x[i + 2 * n] * x[j + 3 * n]
                tmp1 = tmp0 / (1 - tmp0)
                tmp2 = x[i] * x[j + n]

                fa_out += x[j + n] * tmp1 / (1 + tmp2 * tmp1)

                fb_out += (
                    tmp2 * x[j + 3 * n] / ((1 + tmp0 * tmp2) * (1 - tmp0))
                )

                tmp0 = x[j + 2 * n] * x[i + 3 * n]
                tmp1 = tmp0 / (1 - tmp0)
                tmp2 = x[j] * x[i + n]

                fa_in += x[j] * tmp1 / (1 + tmp2 * tmp1)

                fb_in += tmp2 * x[j + 2 * n] / ((1 + tmp0 * tmp2) * (1 - tmp0))

        # print('this',fb_out)
        f[i] = -np.log(k_out[i] / fa_out)
        f[i + 2 * n] = -np.log(s_out[i] / fb_out)
        f[i + n] = -np.log(k_in[i] / fa_in)
        f[i + 3 * n] = -np.log(s_in[i] / fb_in)

    return f


@jit(nopython=True)
def linsearch_fun_DECM_new(X, args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    arg_step_fun = args[1]
    

    # Mettere il check sulle y
    nnn = int(len(x) / 4)
    ind_yout = np.argmin(x[2 * nnn : 3 * nnn])
    ind_yin = np.argmin(x[3 * nnn :])
    tmp = x[2 * nnn : 3 * nnn][ind_yout] + x[3 * nnn :][ind_yin]
    while True:
        ind_yout = np.argmin(
            x[2 * nnn : 3 * nnn] + alfa * dx[2 * nnn : 3 * nnn]
        )
        ind_yin = np.argmin(x[3 * nnn :] + alfa * dx[3 * nnn :])
        cond = (x[2 * nnn : 3 * nnn][ind_yout] + alfa * dx[2 * nnn : 3 * nnn][ind_yout]) + (x[3 * nnn :][ind_yin] + alfa * dx[3 * nnn :][ind_yin])
        if (cond) > tmp * 0.01:
            break
        else:
            alfa *= beta
    
    i = 0
    s_old = -step_fun(x, arg_step_fun)
    while (
        sample.sufficient_decrease_condition(
            s_old, -step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
        )
        == False
        and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa


@jit(nopython=True)
def linsearch_fun_DECM_new_fixed(X):
    x = X[0]
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    # Mettere il check sulle y
    nnn = int(len(x) / 4)
    ind_yout = np.argmin(x[2 * nnn : 3 * nnn])
    ind_yin = np.argmin(x[3 * nnn :])
    tmp = x[2 * nnn : 3 * nnn][ind_yout] + x[3 * nnn :][ind_yin]
    
    while True:
        ind_yout = np.argmin(
            x[2 * nnn : 3 * nnn] + alfa * dx[2 * nnn : 3 * nnn]
        )
        ind_yin = np.argmin(x[3 * nnn :] + alfa * dx[3 * nnn :])
        cond = (x[2 * nnn : 3 * nnn][ind_yout] + alfa * dx[2 * nnn : 3 * nnn][ind_yout]) + (x[3 * nnn :][ind_yin] + alfa * dx[3 * nnn :][ind_yin])
        #print(cond)
        if (cond) > tmp * 0.01:
            break
        else:
            alfa *= beta

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx, ord = 2) < np.linalg.norm(dx_old, ord = 2)
        while(
            cond == False
            and kk<50
            ):
            alfa *= beta
            kk +=1
            cond = np.linalg.norm(alfa*dx[dx!=np.infty], ord = 2) < np.linalg.norm(dx_old[dx_old!=np.infty], ord = 2)

    return alfa