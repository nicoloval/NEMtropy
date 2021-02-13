import numpy as np
from numba import jit, prange
from . import solver_functions as sof


# UBCM functions
# --------------


@jit(nopython=True)
def iterative_cm(x, args):
    """Returns the next UBCM iterative step for the fixed-point method.

    :param x: Previous iterative step.
    :type x: numpy.ndarray
    :param args: Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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
    """Returns UBCM loglikelihood function evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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
    """Returns UBCM loglikelihood gradient function evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
    :rtype: float
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
    """Returns UBCM loglikelihood hessian function evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
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
    """Returns the diagonal of the UBCM loglikelihood hessian function
    evaluated in x.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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


def pmatrix_cm(x, args):
    """Computes and returns the probability matrix induced by UBCM.

    :param x: Solutions of UBCM.
    :type x: numpy.ndarray
    :param args: Number of nodes.
    :type args: (int, )
    :return: UBCM probability matrix.
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
def linsearch_fun_CM(X, args):
    """Linsearch function for UBCM newton and quasinewton methods.
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
    # TODO: change X to xx
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
def linsearch_fun_CM_fixed(X):
    """Linsearch function for UBCM fixed-point method.
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
    # TODO: change X to xx
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
            cond is False
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    return alfa


@jit(nopython=True)
def expected_degree_cm(sol):
    """Computes the expected degrees of UBCM given the solution x.

    :param sol: UBCM solutions.
    :type sol: numpy.ndarray
    :return: Expected degrees sequence.
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

# UBCM exponential functions
# --------------------------


@jit(nopython=True)
def iterative_cm_exp(x, args):
    """Returns the next UBCM iterative step for the fixed-point method.
    It is based on the exponential version of the UBCM.

    :param x: Previous iterative step.
    :type x: numpy.ndarray
    :param args: Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    x1 = np.exp(-x)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        fx = 0
        for j in np.arange(n):
            if i == j:
                fx += (c[j] - 1) * (x1[j] / (1 + x1[j] * x1[i]))
            else:
                fx += (c[j]) * (x1[j] / (1 + x1[j] * x1[i]))
        if fx:
            f[i] = -np.log(k[i] / fx)

    return f


@jit(nopython=True)
def loglikelihood_cm_exp(x, args):
    """Returns UBCM loglikelihood function evaluated in x.
    It is based on the exponential version of the UBCM.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
    :rtype: float
    """
    k = args[0]
    c = args[1]
    n = len(k)
    x1 = np.exp(-x)
    f = 0.0
    for i in np.arange(n):
        f -= c[i] * k[i] * x[i]
        for j in np.arange(n):
            if i == j:
                f -= (c[i] * (c[i] - 1) * np.log(1 + (x1[i]) ** 2)) / 2
            else:
                f -= (c[i] * c[j] * np.log(1 + x1[i] * x1[j])) / 2
    return f


@jit(nopython=True)
def loglikelihood_prime_cm_exp(x, args):
    """Returns UBCM loglikelihood gradient function evaluated in beta.
    It is based on the exponential version of the UBCM.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
    :rtype: float
    """
    k = args[0]
    c = args[1]
    n = len(k)
    x1 = np.exp(-x)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        f[i] -= c[i]*k[i]
        for j in np.arange(n):
            if i == j:
                aux = x1[i] ** 2
                f[i] += c[i] * (c[i] - 1) * (aux / (1 + aux))
            else:
                aux = x1[i] * x1[j]
                f[i] += c[i] * c[j] * (aux / (1 + aux))
    return f


@jit(nopython=True)
def loglikelihood_hessian_cm_exp(x, args):
    """Returns UBCM loglikelihood hessian function evaluated in beta.
    It is based on the exponential version of the UBCM.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    x1 = np.exp(-x)
    f = np.zeros(shape=(n, n), dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(i, n):
            if i == j:
                aux_f = 0
                for h in range(n):
                    if i == h:
                        aux = x1[h] ** 2
                        aux_f -= (aux / (1 + aux) ** 2) * c[i] * (c[h] - 1)
                    else:
                        aux = x1[i] * x1[h]
                        aux_f -= ((aux) / (1 + aux) ** 2) * c[i] * c[h]
            else:
                aux = x1[i] * x1[j]
                aux_f = -((aux) / (1 + aux) ** 2) * c[i] * c[j]

            f[i, j] = aux_f
            f[j, i] = aux_f
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_cm_exp(x, args):
    """Returns the diagonal of the UBCM loglikelihood hessian function
    evaluated in x.
    It is based on the exponential version of the UBCM.

    :param x: Evaluating point *x*.
    :type x: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Degrees and classes cardinality sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
    :rtype: numpy.ndarray
    """
    k = args[0]
    c = args[1]
    n = len(k)
    x1 = np.exp(-x)
    f = np.zeros(n, dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(n):
            if i == j:
                aux = x1[j] ** 2
                f[i] -= (aux / (1 + aux)) * c[i] * (c[j] - 1)
            else:
                aux = x1[i] * x1[j]
                f[i] -= (aux / (1 + aux)) * c[i] * c[j]
    return f


@jit(nopython=True)
def linsearch_fun_CM_exp(X, args):
    """Linsearch function for UBCM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on UBCM exponential version.

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
    # print(alfa)
    return alfa


@jit(nopython=True)
def linsearch_fun_CM_exp_fixed(X):
    """Linsearch function for UBCM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on UBCM exponential version.

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
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            not cond
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
    # print(alfa)
    return alfa

# UECM functions
# --------------


@jit(nopython=True)
def iterative_ecm(sol, args):
    """Returns the next UECM iterative step for the fixed-point method.

    :param sol: Previous iterative step.
    :type sol: numpy.ndarray
    :param args: Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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
    """Returns UECM loglikelihood function evaluated in sol.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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
    """Returns DECM loglikelihood gradient function evaluated in sol.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
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
    """Returns DBCM loglikelihood hessian function evaluated in sol.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian. Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
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
    """Returns the diagonal of UECM loglikelihood hessian function
    evaluated in sol.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Hessian matrix diagonal.
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


@jit(nopython=True)
def linsearch_fun_ECM(X, args):
    """Linsearch function for UECM newton and quasinewton methods.
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
    # TODO: change X to xx
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
def linsearch_fun_ECM_fixed(X):
    """Linsearch function for UECM fixed-point method.
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
    # TODO: change X to xx
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
            cond is False
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)

    return alfa


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

# UECM exponential functions
# --------------------------


@jit(nopython=True)
def iterative_ecm_exp(sol, args):
    """Returns the next DECM iterative step for the fixed-point method.
    It is based on the exponential version of the UECM.

    :param sol: Previous iterative step.
    :type sol: numpy.ndarray
    :param args: Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = np.exp(-sol[:n])
    y = np.exp(-sol[n:])

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
            f[i] = -np.log(k[i] / fx)
        else:
            f[i] = 0.0
        if fy:
            f[i + n] = -np.log(s[i] / fy)
        else:
            f[i + n] = 0.0
    return f


@jit(nopython=True)
def loglikelihood_ecm_exp(sol, args):
    """Returns UECM loglikelihood function evaluated in sol.
    It is based on the exponential version of the UECM.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
    :rtype: float
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = np.exp(-sol[:n])
    y = np.exp(-sol[n:])

    f = 0.0
    for i in np.arange(n):
        f -= k[i] * (-np.log(x[i])) + s[i] * (-np.log(y[i]))
        for j in np.arange(0, i):
            aux1 = x[i] * x[j]
            aux2 = y[i] * y[j]
            f -= np.log(1 + (aux1 * (aux2 / (1 - aux2))))
    return f


@jit(nopython=True)
def loglikelihood_prime_ecm_exp(sol, args):
    """Returns DECM loglikelihood gradient function evaluated in sol.
    It is based on the exponential version of the UECM.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
    :rtype: numpy.ndarray
    """
    k = args[0]
    s = args[1]

    n = len(k)

    x = np.exp(-sol[:n])
    y = np.exp(-sol[n:])

    f = np.zeros(2 * n, dtype=np.float64)
    for i in np.arange(n):
        f[i] -= k[i]
        f[i + n] -= s[i]
        for j in np.arange(n):
            if i != j:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                f[i] += (aux1 * aux2) / (1 - aux2 + aux1 * aux2)
                f[i + n] += (aux1 * aux2) / (
                    (1 - aux2) * (1 - aux2 + aux1 * aux2)
                )
    return f


@jit(nopython=True)
def loglikelihood_hessian_ecm_exp(sol, args):
    """Returns DBCM loglikelihood hessian function evaluated in sol.
    It is based on the exponential version of the UECM.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
    :rtype: numpy.ndarray
    """
    k = args[0]

    n = len(k)

    x = np.exp(-sol[:n])
    y = np.exp(-sol[n:])

    f = np.zeros(shape=(2 * n, 2 * n), dtype=np.float64)
    for i in np.arange(n):

        for j in np.arange(i, n):
            if i == j:
                f1 = 0.0
                f2 = 0.0
                f3 = 0.0
                for h in np.arange(n):
                    if h != i:
                        aux1 = x[i] * x[h]
                        aux2 = y[i] * y[h]
                        aux3 = aux1 * aux2
                        f1 -= ((aux3) * (1 - aux2)) / ((1 - aux2 + aux3) ** 2)
                        f2 -= (
                            aux3 * (1 - (aux2 ** 2) + aux1 * (aux2 ** 2))
                        ) / (((1 - aux2) ** 2) * ((1 - aux2 + aux3) ** 2))
                        f3 -= aux3 / ((1 - aux2 + aux3) ** 2)
                f[i, i] = f1
                f[i + n, i + n] = f2
                f[i + n, i] = f3
                f[i, i + n] = f3
            else:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                aux3 = aux1 * aux2
                aux4 = (1 - aux2 + aux3) ** 2

                aux = -aux3 * (1 - aux2) / aux4
                f[i, j] = aux
                f[j, i] = aux

                aux = -aux3 / aux4
                f[i, j + n] = aux
                f[j + n, i] = aux

                aux = -(aux3 * (1 - (aux2 ** 2) + aux1 * (aux2 ** 2))) / (
                    ((1 - aux2) ** 2) * ((1 - aux2 + aux3) ** 2)
                )
                f[i + n, j + n] = aux
                f[j + n, i + n] = aux

                aux = -aux3 / aux4
                f[i + n, j] = aux
                f[j, i + n] = aux

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_ecm_exp(sol, args):
    """Returns the diagonal of UECM loglikelihood hessian function
    evaluated in sol.
    It is based on the exponential version of the UECM.

    :param sol: Evaluating point *sol*.
    :type sol: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian function.
        Degrees and strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Hessian matrix diagonal.
    :rtype: numpy.ndarray
    """
    k = args[0]

    n = len(k)

    x = np.exp(-sol[:n])
    y = np.exp(-sol[n:])

    f = np.zeros(2 * n, dtype=np.float64)

    for i in np.arange(n):
        for j in np.arange(n):
            if j != i:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                aux3 = aux1 * aux2
                f[i] -= ((aux3) * (1 - aux2)) / ((1 - aux2 + aux3) ** 2)
                f[i + n] -= (
                    aux1 * aux2 * (1 - (aux2 ** 2) + aux1 * (aux2 ** 2))
                ) / (((1 - aux2) ** 2) * ((1 - aux2 + aux3) ** 2))
    return f


@jit(nopython=True)
def linsearch_fun_ECM_exp(X, args):
    """Linsearch function for UECM newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on UBCM exponential version.

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
def linsearch_fun_ECM_exp_fixed(X):
    """Linsearch function for UECM fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    This function works on UBCM exponential version.

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
            (not cond)
            and kk < 50
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)

    return alfa


# CREMA UNDIRECTED functions
# -------------------------

@jit(nopython=True)
def iterative_crema_undirected(beta, args):
    """Returns the next CReMa iterative step for the fixed-point method.
    The UBCM pmatrix is pre-compute and explicitly passed.

    :param beta: Previous iterative step.
    :type beta: numpy.ndarray
    :param args: Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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


@jit(nopython=True, parallel=True)
def iterative_crema_undirected_sparse(beta, args):
    """Returns the next CReMa iterative step for the fixed-point method.
    The UBCM pmatrix is computed inside the function.

    :param beta: Previous iterative step..
    :type beta: numpy.ndarray
    :param args: Strengths sequence and adjacency matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
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
    """Returns the next CReMa iterative step for the fixed-point method.
    The UBCM pmatrix is computed inside the function.
    Alternative version not in use.

    :param beta: Previous iterative step..
    :type beta: numpy.ndarray
    :param args: Strengths sequence and adjacency matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray
    """
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
def loglikelihood_crema_undirected(beta, args):
    """Returns CReMa loglikelihood function evaluated in beta.
    The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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


@jit(nopython=True)
def loglikelihood_crema_undirected_sparse(beta, args):
    """Computes CReMa loglikelihood function evaluated in beta.
    The UBCM pmatrix is computed inside the function.
    Sparse initialisation version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Strengths sequence and adjacency matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
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
def loglikelihood_prime_crema_undirected(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
    The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient value.
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


@jit(nopython=True, parallel=True)
def loglikelihood_prime_crema_undirected_sparse(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
    The UBCM pmatrix is pre-computed and explicitly passed.
    Sparse initialization version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient value.
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
def loglikelihood_prime_crema_sparse_2(beta, args):
    """Returns CReMa loglikelihood gradient function evaluated in beta.
    The UBCM pmatrix is computed inside the function.
    Sparse initialization version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient value.
    :rtype: numpy.ndarray
    """
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
def loglikelihood_hessian_crema_undirected(beta, args):
    """Returns CReMa loglikelihood hessian function evaluated in beta.
    The UBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
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
def loglikelihood_hessian_diag_crema_undirected(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian function
    evaluated in beta. The DBCM pmatrix is pre-computed and explicitly passed.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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


@jit(nopython=True, parallel=True)
def loglikelihood_hessian_diag_crema_undirected_sparse(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian function
    evaluated in beta. The DBCM pmatrix is pre-computed and explicitly passed.
    Sparse initialization version.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
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
def loglikelihood_hessian_diag_crema_sparse_2(beta, args):
    """Returns the diagonal of CReMa loglikelihood hessian function
    evaluated in beta. The UBCM pmatrix is computed inside the function.
    Sparse initialization version.
    Alternative version not in use.

    :param beta: Evaluating point *beta*.
    :type beta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient function.
        Strengths sequence and adjacency binary/probability matrix.
    :type args: (numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
    :rtype: numpy.ndarray
    """
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
def linsearch_fun_crema_undirected(X, args):
    """Linsearch function for CReMa newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.

    :param X: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type X: (numpy.ndarray, numpy.ndarray,
        float, float, func)
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
    while ((not sof.sufficient_decrease_condition(s_old,
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
def linsearch_fun_crema_undirected_fixed(X):
    """Linsearch function for CReMa fixed-point method.
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
    # TODO: change X to xx
    dx = X[1]
    dx_old = X[2]
    alfa = X[3]
    beta = X[4]
    step = X[5]

    if step:
        kk = 0
        cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)
        while(
            (not cond)
            and (kk < 50)
             ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa*dx) < np.linalg.norm(dx_old)

    return alfa


@jit(nopython=True)
def expected_strength_crema_undirected(sol, adj):
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
def expected_strength_crema_undirected_sparse(sol, adj):
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

    :param: Previous iterative step.
    :type: numpy.ndarray
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
def iterative_dcm_exp(theta, args):
    """Returns the next DBCM iterative step for the fixed-point [1]_ [2]_.
        It is based on the exponential version of the DBCM.
        This version only runs on non-zero indices.

    :param theta: Previous iterative step.
    :type theta: numpy.ndarray
    :param args: Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [1] Squartini, Tiziano, and Diego Garlaschelli.
        "Analytical maximum-likelihood method to detect patterns
        in real networks."
        New Journal of Physics 13.8 (2011): 083001.
        `https://arxiv.org/abs/1103.0701 <https://arxiv.org/abs/1103.0701>`_

    .. [2] Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
        "Unbiased sampling of network ensembles."
        New Journal of Physics 17.2 (2015): 023052.
        `https://arxiv.org/abs/1406.1197 <https://arxiv.org/abs/1406.1197>`_
    """
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
def iterative_dcm_exp_2(theta, args):
    """Returns the next DBCM iterative step for the fixed-point.
        It is based on the exponential version of the DBCM.
        This version only runs on non-zero indices.

    :param theta: Previous iterative step.
    :type theta: numpy.ndarray
    :param args: Out and in strengths sequences, adjacency matrix,
        and non zero out and in indices.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray
    """
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
def loglikelihood_dcm_exp(theta, args):
    """Returns DBCM [*]_ [*]_ loglikelihood function evaluated in theta.
    It is based on the exponential version of the DBCM.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and non zero out and in indices,
        and classes cardinalities sequence.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood value.
    :rtype: float

    .. rubric: References
    .. [*] Squartini, Tiziano, and Diego Garlaschelli.
        "Analytical maximum-likelihood method to detect patterns
        in real networks."
        New Journal of Physics 13.8 (2011): 083001.
        `https://arxiv.org/abs/1103.0701 <https://arxiv.org/abs/1103.0701>`_

    .. [*] Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
        "Unbiased sampling of network ensembles."
        New Journal of Physics 17.2 (2015): 023052.
        `https://arxiv.org/abs/1406.1197 <https://arxiv.org/abs/1406.1197>`_
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
def loglikelihood_prime_dcm_exp(theta, args):
    """Returns DBCM [*]_ [*]_ loglikelihood gradient function evaluated in theta.
    It is based on the exponential version of the DBCM.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood gradient.
        Out and in degrees sequences, and non zero out and in indices,
        and the sequence of classes cardinalities.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood gradient.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [*] Squartini, Tiziano, and Diego Garlaschelli.
        "Analytical maximum-likelihood method to detect patterns
        in real networks."
        New Journal of Physics 13.8 (2011): 083001.
        `https://arxiv.org/abs/1103.0701 <https://arxiv.org/abs/1103.0701>`_

    .. [*] Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
        "Unbiased sampling of network ensembles."
        New Journal of Physics 17.2 (2015): 023052.
        `https://arxiv.org/abs/1406.1197 <https://arxiv.org/abs/1406.1197>`_
    """
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
def loglikelihood_hessian_dcm_exp(theta, args):
    """Returns DBCM [*]_ [*]_ loglikelihood hessian function evaluated in theta.
    It is based on the exponential version of the DBCM.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and non zero out and in indices,
        and the sequence of classes cardinalities.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian matrix.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [*] Squartini, Tiziano, and Diego Garlaschelli.
        "Analytical maximum-likelihood method to detect patterns
        in real networks."
        New Journal of Physics 13.8 (2011): 083001.
        `https://arxiv.org/abs/1103.0701 <https://arxiv.org/abs/1103.0701>`_

    .. [*] Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
        "Unbiased sampling of network ensembles."
        New Journal of Physics 17.2 (2015): 023052.
        `https://arxiv.org/abs/1406.1197 <https://arxiv.org/abs/1406.1197>`_
    """
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
def loglikelihood_hessian_diag_dcm_exp(theta, args):
    """Returns the diagonal of the DBCM [*]_ [*]_ loglikelihood hessian
    function evaluated in theta. It is based on DBCM exponential version.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and non zero out and in indices,
        and the sequence of classes cardinalities.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [*] Squartini, Tiziano, and Diego Garlaschelli.
        "Analytical maximum-likelihood method to detect patterns
        in real networks."
        New Journal of Physics 13.8 (2011): 083001.
        `https://arxiv.org/abs/1103.0701 <https://arxiv.org/abs/1103.0701>`_

    .. [*] Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
        "Unbiased sampling of network ensembles."
        New Journal of Physics 17.2 (2015): 023052.
        `https://arxiv.org/abs/1406.1197 <https://arxiv.org/abs/1406.1197>`_
    """
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
def expected_out_degree_dcm_exp(sol):
    """Expected out-degrees after the DBCM. It is based on DBCM
    exponential version.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Out-degrees DBCM expectation.
    :rtype: numpy.ndarray
    """
    n = int(len(sol) / 2)
    ex_k = np.zeros(n, dtype=np.float64)

    for i in np.arange(n):
        for j in np.arange(n):
            if i != j:
                aux = np.exp(-sol[i]) * np.exp(-sol[j])
                ex_k[i] += aux / (1 + aux)
    return ex_k


@jit(nopython=True)
def expected_in_degree_dcm_exp(theta):
    """Expected in-degrees after the DBCM. It is based on DBCM
    exponential version.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: In-degrees DBCM expectation.
    :rtype: numpy.ndarray
    """
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
def linsearch_fun_DCM_exp(X, args):
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
def linsearch_fun_DCM_exp_fixed(X):
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
def iterative_decm_exp(theta, args):
    """Returns the next iterative step for the DECM.
    It is based on the exponential version of the DBCM.

    :param theta: Previous iterative step.
    :type theta: numpy.ndarray
    :param args: Out and in degrees sequences, and out and in strengths
        sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray
    """
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
def iterative_decm_exp_2(theta, args):
    """Returns the next iterative step for the DECM.
    It is based on the exponential version of the DBCM.

    :param theta: Previous iterative step.
    :type theta: numpy.ndarray
    :param args: Out and in degrees sequences, and out and in strengths
        sequences..
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray, numpy.ndarray)
    :return: Next iterative step.
    :rtype: numpy.ndarray
    """
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
def loglikelihood_decm_exp(x, args):
    """Returns DECM [*]_ loglikelihood function evaluated in theta.
    It is based on the exponential version of the DECM.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and out and in strengths sequences
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood value.
    :rtype: float

    .. rubric: References
    .. [*] Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli.
        "A faster horse on a safer trail: generalized inference for the
        efficient reconstruction of weighted networks."
        New Journal of Physics 22.5 (2020): 053053.
        `https://arxiv.org/abs/1811.09829 <https://arxiv.org/abs/1811.09829>`_
    """
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
def loglikelihood_prime_decm_exp(theta, args):
    """Returns DECM [*]_ loglikelihood gradient function evaluated in theta.
    It is based on the exponential version of the DECM.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood gradient.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [*] Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli.
        "A faster horse on a safer trail: generalized inference for the
        efficient reconstruction of weighted networks."
        New Journal of Physics 22.5 (2020): 053053.
        `https://arxiv.org/abs/1811.09829 <https://arxiv.org/abs/1811.09829>`_
    """
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
def loglikelihood_hessian_decm_exp(theta, args):
    """Returns DECM [*]_ loglikelihood hessian function evaluated in theta.
    It is based on the exponential version of the DECM.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood function.
        Out and in degrees sequences, and out and in strengths sequences..
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood hessian matrix.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [*] Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli.
        "A faster horse on a safer trail: generalized inference for the
        efficient reconstruction of weighted networks."
        New Journal of Physics 22.5 (2020): 053053.
        `https://arxiv.org/abs/1811.09829 <https://arxiv.org/abs/1811.09829>`_
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]

    n = len(k_out)
    f = np.zeros((n * 4, n * 4))
    x = np.exp(-theta)

    a_out = x[:n]
    a_in = x[n: 2 * n]
    b_out = x[2 * n: 3 * n]
    b_in = x[3 * n:]

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
def loglikelihood_hessian_diag_decm_exp(theta, args):
    """Returns the diagonal of the DECM [*]_ loglikelihood hessian
    function evaluated in *theta*. It is based on the DECM exponential version.

    :param theta: Evaluating point *theta*.
    :type theta: numpy.ndarray
    :param args: Arguments to define the loglikelihood hessian.
        Out and in degrees sequences, and out and in strengths sequences.
    :type args: (numpy.ndarray, numpy.ndarray, numpy.ndarray,
        numpy.ndarray)
    :return: Loglikelihood hessian diagonal.
    :rtype: numpy.ndarray

    .. rubric: References
    .. [*] Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli.
        "A faster horse on a safer trail: generalized inference for the
        efficient reconstruction of weighted networks."
        New Journal of Physics 22.5 (2020): 053053.
        `https://arxiv.org/abs/1811.09829 <https://arxiv.org/abs/1811.09829>`_
    """
    k_out = args[0]
    k_in = args[1]
    s_out = args[2]
    s_in = args[3]

    n = len(k_out)
    f = np.zeros(n * 4)
    x = np.exp(-theta)

    a_out = x[:n]
    a_in = x[n: 2 * n]
    b_out = x[2 * n: 3 * n]
    b_in = x[3 * n:]

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
def expected_decm_exp(theta):
    """Expected parameters after the DBCM.
    It returns a concatenated array of out-degrees and in-degrees.
    It is based on DBCM exponential version.

    :param theta: DBCM solution.
    :type x: numpy.ndarray
    :return: DBCM expected parameters sequence.
    :rtype: numpy.ndarray
    """
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
def linsearch_fun_DECM_exp(X, args):
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
def linsearch_fun_DECM_exp_fixed(X):
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

# CREMA functions
# ---------------


@jit(nopython=True)
def iterative_crema_directed(beta, args):
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
def iterative_crema_directed_sparse(beta, args):
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
def iterative_crema_directed_sparse_2(beta, args):
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
def loglikelihood_crema_directed(beta, args):
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
def loglikelihood_crema_directed_sparse(beta, args):
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
def loglikelihood_prime_crema_directed(beta, args):
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
def loglikelihood_prime_crema_directed_sparse(beta, args):
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
def loglikelihood_prime_crema_directed_sparse_2(beta, args):
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
def loglikelihood_hessian_crema_directed(beta, args):
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
def loglikelihood_hessian_diag_crema_directed(beta, args):
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
def loglikelihood_hessian_diag_crema_directed_sparse(beta, args):
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
def loglikelihood_hessian_diag_crema_directed_sparse_2(beta, args):
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
def expected_out_strength_crema_directed(sol, adj):
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
def expected_out_strength_crema_directed_sparse(sol, adj):
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
def expected_in_stregth_crema_directed(sol, adj):
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
def expected_in_stregth_crema_directed_sparse(sol, adj):
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
def linsearch_fun_crema_directed(xx, args):
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
def linsearch_fun_crema_directed_fixed(xx):
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
