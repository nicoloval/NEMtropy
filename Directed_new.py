import numpy as np
import scipy.sparse
from numba import jit
import time


@jit(nopython=True)
def iterative_dcm_new_bis(theta, args):
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n = len(k_out)
    nz_index_out = args[2]
    nz_index_in = args[3]
    # nz_index_out = range(n)
    # nz_index_in = range(n)
    c = args[4]

    f = np.zeros(2*n, dtype=np.float64 )
    x = np.exp(-theta)

    for i in nz_index_out:
        for j in nz_index_in:
            if j != i:
                f[i] += c[j]*x[j+n]/(1 + x[i]*x[j+n])
            else:
                f[i] += (c[j] - 1)*x[j+n]/(1 + x[i]*x[j+n])

    for j in nz_index_in:
        for i in nz_index_out:
            if j != i:
                f[j+n] += c[i]*x[i]/(1 + x[i]*x[j+n])
            else:
                f[j+n] += (c[i] - 1)*x[i]/(1 + x[i]*x[j+n])

    tmp = np.concatenate((k_out, k_in))
    ff = -np.log(np.array([tmp[i]/f[i] if tmp[i] != 0 else -np.infty for i in range(2*n)]))
    # ff = -np.log(tmp/f)

    return ff



@jit(nopython=True)
def iterative_dcm_new(theta, args):
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n = len(k_out)
    # nz_index_out = args[2]
    # nz_index_in = args[3]
    nz_index_out = range(n)
    nz_index_in = range(n)
    c = args[4]

    f = np.zeros(2*n, dtype=np.float64 )
    x = np.exp(-theta)

    for i in nz_index_out:
        for j in nz_index_in:
            if j != i:
                f[i] += c[j]*x[j+n]/(1 + x[i]*x[j+n])
            else:
                f[i] += (c[j] - 1)*x[j+n]/(1 + x[i]*x[j+n])

    for j in nz_index_in:
        for i in nz_index_out:
            if j != i:
                f[j+n] += c[i]*x[i]/(1 + x[i]*x[j+n])
            else:
                f[j+n] += (c[i] - 1)*x[i]/(1 + x[i]*x[j+n])

    tmp = np.concatenate((k_out, k_in))
    # ff = np.array([tmp[i]/f[i] if tmp[i] != 0 else 0 for i in range(2*n)])
    ff = -np.log(tmp/f)

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
    c = args[4]
    n = len(k_out)

    f = 0
    x = np.exp(-theta)

    for i in nz_index_out:
        f -= c[i]*k_out[i]*theta[i]
        for j in nz_index_in:
            if i != j:
                f -= c[i]*c[j]*np.log(1 + np.exp(-theta[i]-theta[n+j]))
            else:
                f -= c[i]*(c[i] - 1)*np.log(1 + np.exp(-theta[i]-theta[n+j]))

    for j in nz_index_in:
            f -= c[j]*k_in[j]*theta[j+n]

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

    f = np.zeros(2*n)
    x = np.exp(-theta)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i!= j:
                # const = c[i]*c[j]
                const = c[j]
            else:
                # const = c[i]*(c[j] - 1)
                const = (c[j] - 1)

            fx += const*x[j+n]/(1 + x[i]*x[j+n])
        # original prime
        f[i] = x[i]*fx - k_out[i]

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i!= j:
                # const = c[i]*c[j]
                const = c[i]
            else:
                # const = c[i]*(c[j] - 1)
                const = (c[j] - 1)

            fy += const*x[i]/(1 + x[j+n]*x[i])
        # original prime
        f[j+n] = fy*x[j+n] - k_in[j]

    return f


@jit(nopython=True)
def loglikelihood_hessian_dcm_new(theta, args):
    k_out = args[0]
    k_in = args[1]
    nz_out_index = args[2]
    nz_in_index = args[3]
    c = args[4]
    n = len(k_out)

    out = -np.ones((2*n, 2*n))  # hessian matrix
    x = np.exp(-theta)

    for h in nz_out_index:
        tmp_sum = 0
        for i in nz_in_index:
            if i == h:
                # const = c[h]*(c[h] - 1)
                const = (c[h] - 1)
            else:
                # const = c[h]*c[i]
                const = c[i]

            tmp = x[i+n]*x[h]
            tmp_sum += const*(tmp)/(1 + tmp)**2
            out[h, i+n] = const*tmp/(1 + tmp)**2
        out[h, h] = tmp_sum


    for i in nz_in_index:
        tmp_sum = 0
        for h in nz_out_index:
            if i == h:
                # const = c[h]*(c[h] - 1)
                const = (c[i] - 1)
            else:
                # const = c[h]*c[i]
                const = c[h]

            tmp = x[h]*x[i+n]
            tmp_sum += const*(tmp)/(1 + tmp)**2
            out[i+n, h] = -const*tmp/(1 + tmp)**2
        out[i+n, i+n] = tmp_sum
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

    f = -np.ones(2*n)
    x = np.exp(-theta)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i!= j:
                # const = c[i]*c[j]
                const = c[j]
            else:
                # const = c[i]*(c[j] - 1)
                const = (c[i] - 1)

            tmp = x[i+n]*x[h]
            fx += const*(tmp)/(1 + tmp)**2
        # original prime
        f[i] = fx

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i!= j:
                # const = c[i]*c[j]
                const = c[i]
            else:
                # const = c[i]*(c[j] - 1)
                const = (c[j] - 1)


            tmp = x[h]*x[i+n]
            fy += const*(tmp)/(1 + tmp)**2
        # original prime
        f[j+n] = fy

    # f[f == 0] = 1

    return f


@jit(nopython=True)
def expected_out_degree_dcm_new(sol):
    ex_k = np.zeros_like(sol, dtype=np.float64)
    n = int(len(sol)/2)

    for i in np.arange(n):
        for j in np.arange(n):
            if i!=j:
                aux = np.exp(-sol[i])*np.exp(-sol[j])
                ex_k[i] += aux/(1+aux)
    return ex_k
