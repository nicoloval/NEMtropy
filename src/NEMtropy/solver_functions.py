import numpy as np
import scipy.sparse
import scipy
from numba import jit
import time


def matrix_regulariser_function(b, eps):
    """Trasforms input matrix in a positive defined matrix
    by adding positive quantites to the main diagonal.

    :param b: Matrix.
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix.
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    bf = b + np.identity(b.shape[0]) * eps

    return bf


def matrix_regulariser_function_eigen_based(b, eps):
    """Trasform input matrix in a positive defined matrix
    by regularising eigenvalues.

    :param b: Matrix.
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix.
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    t, e = scipy.linalg.eigh(b)
    ll = np.array([0 if li > eps else eps - li for li in t])
    bf = e @ (np.diag(ll) + np.diag(t)) @ e.transpose()

    return bf


@jit(nopython=True)
def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-04, c2=0.9
):
    """Returns True if upper wolfe condition is respected.

    :param f_old: Function value at previous iteration.
    :type f_old: float
    :param f_new: Function value at current iteration.
    :type f_new: float
    :param alpha: Alpha parameter of linsearch.
    :type alpha: float
    :param grad_f: Function gradient.
    :type grad_f: numpy.ndarray
    :param p: Current iteration increment.
    :type p: numpy.ndarray
    :param c1: Tuning parameter, defaults to 1e-04.
    :type c1: float, optional
    :param c2: Tuning parameter, defaults to 0.9.
    :type c2: float, optional
    :return: Condition validity.
    :rtype: bool
    """
    sup = f_old + c1 * alpha * np.dot(grad_f, p.T)
    return bool(f_new < sup)


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
    :param tol: The solver stops when \|fun|<tol, defaults to 1e-6
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
        # to avoid nans given by inf-inf
        diff_v[np.isnan(diff_v)] = -1
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
