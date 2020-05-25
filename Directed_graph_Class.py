import numpy as np
import scipy.sparse
from numba import jit
import time

@jit(nopython=True)
def pmatrix_dcm(x,args):
    """
    Function evaluating the P-matrix of DCM given the solution of the underlying model
    """
    n = args[0]
    index_out = args[1]
    index_in = args[2]
    P = np.zeros((n,n),dtype=np.int64)
    xout = x[:n]
    yin = x[n:]
    for ii in index_out:
        for jj in index_in:
            aux = xout[ii]*yin[jj]
            P[ii,jj] = aux/(1+aux)
    return P


@jit(nopython=True)
def loglikelihood_CReAMa(beta,args):
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    
    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]
    
    f=0
    
    for ii in nz_index_out:
        f -= s_out[ii] * beta_out[ii] + s_in[ii] * beta_in[ii]
        for jj in nz_index_in:
            if (ii!=jj) and (adj[ii,jj]!=0):
                f += adj[ii,jj] * np.log(beta_out[ii] + beta_in[jj])
    
    return f


@jit(nopython=True)
def loglikelihood_prime_CReAMa(beta, args):
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    
    aux_n = len(s_out)

    beta_out = beta[0:aux_n]
    beta_in = beta[aux_n:2*aux_n]

    aux_F = np.zeros_like(beta)

    for ii in nz_index_out:

        aux_F[ii] = s_out[ii]
        aux_F[ii+aux_n] -= s_in[ii]
        for jj in nz_index_in:
            if ii != jj:
                if adj[ii, jj] > 0:
                    aux_F[ii] += adj[ii, jj]/(beta_out[ii]+beta_in[jj])
                if adj[jj, ii] > 0:
                    aux_F[ii+aux_n] += adj[jj, ii]/(beta_in[ii]+beta_out[jj])

    return (aux_F)


@jit(nopython=True)
def loglikelihood_hessian_CReAMa(beta, args):
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n  = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(shape=(2*aux_n, 2*aux_n))

    for ii in np.arange(f.shape[0]):
        for jj in np.arange(f.shape[0]):

            # diagonal first n elements
            if (ii == jj) & (ii < aux_n) & (jj < aux_n):
                for kk in np.arange(aux_n):
                    if adj[ii, kk] > 0:
                        f[ii, jj] -= adj[ii, kk] / \
                            ((beta_out[ii]+beta_in[kk])**2)

            # diagonal second n elements
            elif (ii == jj) & (ii >= aux_n) & (jj >= aux_n):
                for kk in np.arange(aux_n):
                    if adj[kk, ii-aux_n] > 0:
                        f[ii, jj] -= adj[kk, ii-aux_n] / \
                            ((beta_out[kk]+beta_in[ii-aux_n])**2)

            # off-diagonal top right elements
            elif (ii != jj) & (ii < aux_n) & (jj >= aux_n):
                if adj[ii, jj-aux_n] > 0:
                    f[ii, jj] = -adj[ii, jj-aux_n] / \
                        ((beta_out[ii]+beta_in[jj-aux_n])**2)

            # off-diagonal bottom left elements
            elif (ii != jj) & (ii >= aux_n) & (jj < aux_n):
                if adj[jj, ii-aux_n] > 0:
                    f[ii, jj] = -adj[jj, ii-aux_n] / \
                        (beta_out[jj] + beta_in[ii-aux_n])

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_CReAMa(beta, args):
    s_out = args[0]
    s_in = args[1]
    adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]

    aux_n  = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(2*aux_n)

    for ii in nz_index_out:
        for jj in nz_index_in:
            if adj[ii,jj]>0:
                f[ii] -= adj[ii, jj] / \
                         ((beta_out[ii]+beta_in[jj])**2)

    for ii in nz_index_in:
        for jj in nz_index_out:
            if adj[ii,jj]>0:
                f[ii+aux_n] -= adj[ii, jj] / \
                               ((beta_out[jj] + beta_in[ii])**2)

    return f




def random_binary_matrix_generator_nozeros(n, sym=False, seed=None):
        if sym == False:
                np.random.seed(seed = seed)
                A = np.random.randint(0, 2, size=(n, n))
                # zeros on the diagonal
                for i in range(n):
                    A[i, i] = 0
                k_in = np.sum(A, axis=0)
                for ind, k in enumerate(k_in):
                        if k==0:
                                A[0, ind] = 1
                k_out = np.sum(A, axis=1)
                for ind, k in enumerate(k_out):
                        if k==0:
                                A[ind, 0] = 1
        return A


@jit(nopython=True)
def loglikelihood_dcm(x, args):
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
    for i in nz_index_out:
        f += c[i]*k_out[i]*np.log(x[i])
        for j in nz_index_in:
            if i != j:
                f -= c[i]*c[j]*np.log(1 + x[i]*x[n+j])
            else:
                f -= c[i]*(c[i] - 1)*np.log(1 + x[i]*x[n+j])

    for j in nz_index_in:
            f += c[j]*k_in[j]*np.log(x[j+n])

    return f


@jit(nopython=True)
def loglikelihood_dcm_notrd(x, args):
    """loglikelihood function for dcm
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    n = len(k_in)

    f = 0
    for i in nz_index_out:
        f += k_out[i]*np.log(x[i])
        for j in nz_index_in:
            if i != j:
                f -= np.log(1 + x[i]*x[n+j])

    for j in nz_index_in:
            f += k_in[j]*np.log(x[j+n])

    return f


@jit(nopython=True)
def loglikelihood_dcm_loop(x, args):
    """loglikelihood function for dcm
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n_out = len(k_out)
    n_in = len(k_in)
    f = 0
    for i in range(n_out):
        f += k_out[i]*np.log(x[i])
        for j in range(n_in):
            f -= np.log(1 + x[i]*x[n_out+j])

    for j in range(n_in):
        f += k_in[j]*np.log(x[j+n_out])

    return f


@jit(nopython=True)
def loglikelihood_prime_dcm(x, args):
    """iterative function for loglikelihood gradient dcm
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_in)

    f = np.zeros(2*n)

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
        f[i] = -fx + k_out[i]/x[i]

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
        f[j+n] = -fy + k_in[j]/x[j+n]

    return f


@jit(nopython=True)
def iterative_dcm(x, args):
    """Return the next iterative step for the Directed Configuration Model Reduced version.

    :param numpy.ndarray v: old iteration step 
    :param numpy.ndarray par: constant parameters of the cm function
    :return: next iteration step 
    :rtype: numpy.ndarray
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

    f = np.zeros(2*n)

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
    ff = tmp/f 

    return ff 


@jit(nopython=True)
def iterative_dcm_old(x, args):
    """Return the next iterative step for the Directed Configuration Model Reduced version.

    :param numpy.ndarray v: old iteration step 
    :param numpy.ndarray par: constant parameters of the cm function
    :return: next iteration step 
    :rtype: numpy.ndarray
    """

    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n = len(k_out)
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]

    f = np.zeros(2*n)

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
    ff = np.array([tmp[i]/f[i] for i in range(2*n)])

    return ff 


@jit(nopython=True)
def loglikelihood_prime_dcm_notrd(x, args):
    """iterative function for loglikelihood gradient dcm
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    n = len(k_in)

    f = np.zeros(2*n)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i!= j:
                fx += x[j+n]/(1 + x[i]*x[j+n])
        # original prime
        f[i] = -fx + k_out[i]/x[i]

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i!= j:
                fy += x[i]/(1 + x[j+n]*x[i])
        # original prime
        f[j+n] = -fy + k_in[j]/x[j+n]

    return f


@jit(nopython=True)
def loglikelihood_prime_dcm_loop(x, args):
    """iterative function for loglikelihood gradient dcm
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n_out = len(k_out)
    n_in = len(k_in)

    f = np.zeros(n_out+n_in)

    for i in range(n_out):
        fx = 0
        for j in range(n_in):
            fx += x[j+n_out]/(1 + x[i]*x[j+n_out])
        # original prime
        f[i] = -fx + k_out[i]/x[i]

    for j in range(n_in):
        fy = 0
        for i in range(n_out):
            fy += x[i]/(1 + x[j+n_out]*x[i])
        # original prime
        f[j+n_out] = -fy + k_in[j]/x[j+n_out]

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_dcm(x, args):
    """hessian diagonal of dcm loglikelihood
    """
    # problem fixed paprameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    c = args[4]
    n = len(k_in)

    f = np.zeros(2*n)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i!= j:
                # const = c[i]*c[j]
                const = c[j]
            else:
                # const = c[i]*(c[j] - 1)
                const = (c[j] - 1)
            
            fx += const*x[j+n]*x[j+n]/((1 + x[i]*x[j+n])*(1 + x[i]*x[j+n]))
        # original prime
        f[i] = fx - k_out[i]/(x[i]*x[i])

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i!= j:
                # const = c[i]*c[j]
                const = c[i]
            else:
                # const = c[i]*(c[j] - 1)
                const = (c[j] - 1)
            
            fy += const*x[i]*x[i]/((1 + x[j+n]*x[i])*(1 + x[j+n]*x[i]))
        # original prime
        f[j+n] = fy - k_in[j]/(x[j+n]*x[j+n])

    # f[f == 0] = 1

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_dcm_notrd(x, args):
    """hessian diagonal of dcm loglikelihood
    """
    # problem fixed paprameters
    k_out = args[0]
    k_in = args[1]
    nz_index_out = args[2]
    nz_index_in = args[3]
    n = len(k_in)

    f = np.zeros(2*n)

    for i in nz_index_out:
        fx = 0
        for j in nz_index_in:
            if i!= j:
                fx += x[j+n]*x[j+n]/((1 + x[i]*x[j+n])*(1 + x[i]*x[j+n]))
        # original prime
        f[i] = fx - k_out[i]/(x[i]*x[i])

    for j in nz_index_in:
        fy = 0
        for i in nz_index_out:
            if i!= j:
                fy += x[i]*x[i]/((1 + x[j+n]*x[i])*(1 + x[j+n]*x[i]))
        # original prime
        f[j+n] = fy - k_in[j]/(x[j+n]*x[j+n])

    # f[f == 0] = 1

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_dcm_loop(x, args):
    """hessian diagonal of dcm loglikelihood
    """
    # problem fixed parameters
    k_out = args[0]
    k_in = args[1]
    n_out = len(k_out)
    n_in = len(k_in)

    f = np.zeros(n_out + n_in)

    for i in range(n_out):
        fx = 0
        for j in range(n_in):
            fx += x[j+n_out]*x[j+n_out]/((1 + x[i]*x[j+n_out])*(1 + x[i]*x[j+n_out]))
        # original prime
        f[i] = fx - k_out[i]/(x[i]*x[i])

    for j in range(n_in):
        fy = 0
        for i in range(n_out):
            fy += x[i]*x[i]/((1 + x[j+n_out]*x[i])*(1 + x[j+n_out]*x[i]))
        # original prime
        f[j+n_out] = fy - k_in[j]/(x[j+n_out]*x[j+n_out])

    return f


@jit(nopython=True)
def expected_out_degree_dcm(sol):
    n = int(len(sol)/ 2)
    a_out = sol[:n]
    a_in = sol[n:]

    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += a_in[j]*a_out[i] / (1 + a_in[j]*a_out[i])

    return k


@jit(nopython=True)
def expected_in_degree_dcm(sol):
    n = int(len(sol)/2)
    a_out = sol[:n]
    a_in = sol[n:]
    k = np.zeros(n)  # allocate k
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i] += a_in[i]*a_out[j]/(1 + a_in[i]*a_out[j])

    return k


def solver(x0, fun, g, fun_jac=None, tol=1e-6, eps=1e-10, max_steps=100, method='newton', verbose=False, regularise=False, full_return = False):
    """Find roots of eq. f = 0, using newton, quasinewton or dianati.
    """

    tic_all = time.time()
    toc_init = 0
    tic = time.time()

    stop_fun = g  # TODO: change g to stop_fun in the header and all functions below

    # algorithm
    beta = .5  # to compute alpha
    n_steps = 0
    x = x0  # initial point

    norm = np.linalg.norm(fun(x), ord=np.inf)
    diff = 1

    if full_return:
        norm_seq = [norm]

    if verbose:
        print('\nx0 = {}'.format(x))
        print('|f(x0)| = {}'.format(norm))

    toc_init = time.time() - tic

    toc_alfa = 0
    toc_update = 0
    toc_dx = 0
    toc_jacfun = 0

    tic_loop = time.time()

    while norm > tol and diff > tol and n_steps < max_steps:  # stopping condition
        x_old = x  # save previous iteration

        # f jacobian
        tic = time.time()
        if method == 'newton':
            H = fun_jac(x)  # original jacobian
            # check the hessian is positive definite
            # l, e = np.linalg.eigh(H)
            l, e = np.linalg.eig(H)
            ml = np.min(l)
            # if it's not positive definite -> regularise
            if ml < eps:
                regularise = True
            # regularisation
            if regularise == True:
                B = hessian_regulariser_function(H, eps)
                l, e = np.linalg.eigh(B)
                new_ml = np.min(l)
            else:
                B = H.__array__()
        elif method == 'quasinewton':
            # quasinewton hessian approximation
            B = fun_jac(x)  # Jacobian diagonal
            if regularise == True:
                B = np.maximum(B, B*0 + 1e-8)
        toc_jacfun += time.time() - tic

        # discending direction computation
        tic = time.time()
        if method == 'newton':
            dx = np.linalg.solve(B, - fun(x))
        elif method == 'quasinewton':
            dx = - fun(x)/B
        elif method == 'fixed-point':
            dx = fun(x) - x
        toc_dx += time.time() - tic

        # backtraking line search
        tic = time.time()
        """
        alfa = 0.1
        eps2=1e-2
        alfa0 = (eps2-1)*x/dx
        for a in alfa0:
            if a>=0:
                alfa = min(alfa, a)
        """
        alfa = 1 
        i = 0
        while sufficient_decrease_condition(stop_fun(x), \
            stop_fun(x + alfa*dx), alfa, fun(x), dx) == False and i<50:
            alfa *= beta
            i +=1

        #TODO: instafix, to check
        if method == 'fixed-point':
            alfa = 1
        """
        if method == 'fixed-point':
            alfa = 0.1
            eps2 = 1e-2
            alfa0 = (eps2-1)*x/dx
            for a in alfa0:
                if a >= 0:
                    alfa = min(alfa, a)
        """
       

        toc_alfa += time.time() - tic

        tic = time.time()
        # solution update
        # direction= dx@fun(x).T
        if method in ['newton', 'quasinewton']:
            x = x + alfa*dx
        if method in ['fixed-point']:
            x = alfa*(x + dx)
        toc_update += time.time() - tic

        # stopping condition computation
        norm = np.linalg.norm(fun(x), ord=np.inf)
        diff = np.linalg.norm(x - x_old)

        if full_return:
            norm_seq.append(norm)

        # step update
        n_steps += 1

        if verbose == True:
            print('step {}'.format(n_steps))
            print('alpha = {}'.format(alfa))
            # print('B = {}'.format(B))
            # print('fun = {}'.format(fun(x)))
            print('dx = {}'.format(dx))
            print('x = {}'.format(x))
            print('|f(x)| = {}'.format(norm))
            if method in ['newton', 'quasinewton']:
                print('B = {}'.format(B))

    toc_loop = time.time() - tic_loop
    toc_all = time.time() - tic_all

    if verbose == True:
        print('Number of steps for convergence = {}'.format(n_steps))
        print('toc_init = {}'.format(toc_init))
        print('toc_jacfun = {}'.format(toc_jacfun))
        print('toc_alfa = {}'.format(toc_alfa))
        print('toc_dx = {}'.format(toc_dx))
        print('toc_update = {}'.format(toc_update))
        print('toc_loop = {}'.format(toc_loop))
        print('toc_all = {}'.format(toc_all))

    if full_return:
        return (x, toc_all, n_steps, np.array(norm_seq))
    else:
        return x


def sufficient_decrease_condition(f_old, f_new, alpha, grad_f, p, c1=1e-04 , c2=.9):
    """return boolean indicator if upper wolfe condition are respected.
    """
    # print(f_old, f_new, alpha, grad_f, p)
    
    #print ('f_old',f_old)
    #print ('c1',c1)
    #print('alpha',alpha)
    #print ('grad_f',grad_f)
    #print('p.T',p.T)

    sup = f_old + c1 *alpha*grad_f@p.T
    return bool(f_new < sup)


def edgelist_from_edgelist(edgelist):
    """
        Creates a new edgelist with the indexes of the nodes instead of the names.
        Returns also two dictionaries that keep track of the nodes.
        """
    if len(edgelist[0]) == 2:
        nodetype = type(edgelist[0][0])
        edgelist = np.array(edgelist, dtype=np.dtype(
            [('source', nodetype), ('target', nodetype)]))
    else:
        nodetype = type(edgelist[0][0])
        weigthtype = type(edgelist[0][2])
        # Vorrei mettere una condizione sul weighttype che deve essere numerico
        edgelist = np.array(edgelist, dtype=np.dtype(
            [('source', nodetype), ('target', nodetype), ('weigth', weigthtype)]))
    # If there is a loop we count it twice in the degree of the node.
    unique_nodes = np.unique(np.concatenate(
        (edgelist['source'], edgelist['target'])), return_counts=False)
    out_degree = np.zeros_like(unique_nodes)
    in_degree = np.zeros_like(unique_nodes)
    nodes_dict = dict(enumerate(unique_nodes))
    inv_nodes_dict = {v: k for k, v in nodes_dict.items()}
    if len(edgelist[0]) == 2:
        edgelist_new = [(inv_nodes_dict[edge[0]], inv_nodes_dict[edge[1]])
                         for edge in edgelist]
        edgelist_new = np.array(edgelist_new, dtype=np.dtype(
            [('source', int), ('target', int)]))
    else:
        edgelist_new = [(inv_nodes_dict[edge[0]],
                         inv_nodes_dict[edge[1]], edge[2]) for edge in edgelist]
        edgelist_new = np.array(edgelist_new, dtype=np.dtype(
            [('source', int), ('target', int), ('weigth', weigthtype)]))
    out_indices, out_counts = np.unique(
        edgelist_new['source'], return_counts=True)
    in_indices, in_counts = np.unique(
        edgelist_new['target'], return_counts=True)
    out_degree[out_indices] = out_counts
    in_degree[in_indices] = in_counts
    if len(edgelist[0]) == 3:
        out_strength = np.zeros_like(unique_nodes,dtype=weigthtype)
        in_strength = np.zeros_like(unique_nodes,dtype=weigthtype)
        out_counts_strength = np.array(
            [edgelist_new[edgelist_new['source'] == i]['weigth'].sum() for i in out_indices])
        in_counts_strength = np.array(
            [edgelist_new[edgelist_new['target'] == i]['weigth'].sum() for i in in_indices])
        out_strength[out_indices] = out_counts_strength
        in_strength[in_indices] = in_counts_strength
        return edgelist_new, out_degree, in_degree, out_strength, in_strength, nodes_dict
    return edgelist_new, out_degree, in_degree, nodes_dict


class DirectedGraph:
    def __init__(self, adjacency=None, edgelist=None, degree_sequence=None, strength_sequence=None):
        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.sparse_adjacency = None
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
        self._initialize_graph(adjacency=adjacency, edgelist=edgelist,
                               degree_sequence=degree_sequence, strength_sequence=strength_sequence)
        self.avg_mat = None
        self.x = None
        self.y = None
        self.xy = None
        self.b_out = None
        self.b_in = None
        

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
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        # Problem (reduced) residuals
        self.residuals = None
        self.final_result = None
        self.r_beta = None

        self.nz_index_out = None
        self.rnz_dseq_out = None
        self.nz_index_in = None
        self.rnz_dseq_in = None
        
        # model
        self.error = None
        self.full_return = False

        # function
        self.args = None


    def _initialize_graph(self, adjacency=None, edgelist=None, degree_sequence=None, strength_sequence=None):
        # Here we can put controls over the type of input. For instance, if the graph is directed,
        # i.e. adjacency matrix is asymmetric, the class to use must be the DiGraph,
        # or if the graph is weighted (edgelist contains triplets or matrix is not binary) or bipartite


        if adjacency is not None:
            if not isinstance(adjacency, (list, np.ndarray)) and not scipy.sparse.isspmatrix(adjacency):
                raise TypeError('The adjacency matrix must be passed as a list or numpy array or scipy sparse matrix.')
            elif adjacency.size > 0:
                if (adjacency<0).any():
                    raise TypeError('The adjacency matrix entries must be positive.')
                if isinstance(adjacency, list): # Cast it to a numpy array: if it is given as a list it should not be too large
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.sparse_adjacency = adjacency
                if np.sum(adjacency)==np.sum(adjacency>0):
                    self.dseq_in = np.sum(adjacency, axis=0)
                    self.dseq_out = np.sum(adjacency, axis=1)
                else:
                    self.dseq_in = np.sum(adjacency>0, axis=0)
                    self.dseq_out = np.sum(adjacency>0, axis=1)
                    self.in_strength = np.sum(adjacency, axis=0)
                    self.out_strength = np.sum(adjacency, axis=1)
                    self.nz_index_sout = np.nonzero(self.out_strength)[0]
                    self.nz_index_sin = np.nonzero(self.in_strength)[0]
                    self.is_weighted = True
                    
                # self.edgelist, self.deg_seq = edgelist_from_adjacency(adjacency)
                self.n_nodes = len(self.dseq_out)
                self.n_edges = np.sum(self.dseq_out)
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError('The edgelist must be passed as a list or numpy array.')
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        'This is not an edgelist. An edgelist must be a list or array of couples of nodes with optional weights. Is this an adjacency matrix?')
                elif len(edgelist[0])==2:
                    self.edgelist, self.dseq_out, self.dseq_in, self.nodes_dict = edgelist_from_edgelist(edgelist)
                else:
                    self.edgelist, self.dseq_out, self.dseq_in, self.out_strength, self.in_strength, self.nodes_dict = edgelist_from_edgelist(edgelist)
                self.n_nodes = len(self.dseq_out)
                self.n_edges = np.sum(self.dseq_out)
                self.is_initialized = True

        elif degree_sequence is not None:
            if not isinstance(degree_sequence, (list, np.ndarray)):
                raise TypeError('The degree sequence must be passed as a list or numpy array.')
            elif len(degree_sequence) > 0:
                try:
                    int(degree_sequence[0])
                except:
                    raise TypeError('The degree sequence must contain numeric values.')
                if (np.array(degree_sequence) < 0).sum() > 0:
                        raise ValueError('A degree cannot be negative.')
                else:
                    if len(degree_sequence)%2 !=0:
                        raise ValueError('Strength-in/out arrays must have same length.')
                    self.n_nodes = int(len(degree_sequence)/2)
                    self.dseq_out = degree_sequence[:self.n_nodes]
                    self.dseq_in = degree_sequence[self.n_nodes:]
                    self.n_edges = np.sum(self.dseq_out)
                    self.is_initialized = True
                if strength_sequence is not None:
                    if not isinstance(strength_sequence, (list, np.ndarray)):
                        raise TypeError('The strength sequence must be passed as a list or numpy array.')
                    elif len(strength_sequence):
                        try:
                            int(strength_sequence[0])
                        except:
                            raise TypeError('The strength sequence must contain numeric values.')
                        if (np.array(strength_sequence)<0).sum() >0:
                            raise ValueError('A strength cannot be negative.')
                        else:
                            if len(strength_sequence)%2 !=0:
                                raise ValueError('Strength-in/out arrays must have same length.')
                            self.n_nodes = int(len(strength_sequence)/2)
                            self.out_strength = strength_sequence[:self.n_nodes]
                            self.in_strength = strength_sequence[self.n_nodes:]
                            self.nz_index_sout = np.nonzero(self.out_strength)[0]
                            self.nz_index_sin = np.nonzero(self.in_strength)[0]
                            self.is_weighted = True
                            self.is_initialized = True

        elif strength_sequence is not None:
            if not isinstance(strength_sequence, (list, np.ndarray)):
                raise TypeError('The strength sequence must be passed as a list or numpy array.')
            elif len(strength_sequence):
                try:
                    int(strength_sequence[0])
                except:
                    raise TypeError('The strength sequence must contain numeric values.')
                if (np.array(strength_sequence)<0).sum() >0:
                    raise ValueError('A strength cannot be negative.')
                else:
                    if len(strength_sequence)%2 !=0:
                        raise ValueError('Strength-in/out arrays must have same length.')
                    self.n_nodes = int(len(strength_sequence)/2)
                    self.out_strength = strength_sequence[:self.n_nodes]
                    self.in_strength = strength_sequence[self.n_nodes:]
                    self.nz_index_sout = np.nonzero(self.out_strength)[0]
                    self.nz_index_sin = np.nonzero(self.in_strength)[0]
                    self.is_weighted = True
                    self.is_initialized = True


    def set_adjacency_matrix(self, adjacency):
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(adjacency=adjacency)


    def set_edgelist(self, edgelist):
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(edgelist=edgelist)


    def set_degree_sequences(self, degree_sequence):
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(degree_sequence=degree_sequence)


    def clean_edges(self):
        self.adjacency = None
        self.edgelist = None
        self.deg_seq = None
        self.is_initialized = False


    def _solve_problem(self, initial_guess=None, model='dcm', method='quasinewton', max_steps=100, full_return=False, verbose=False):

        self.initial_guess = initial_guess
        self._initialize_problem(model, method)
        x0 = np.concatenate((self.r_x, self.r_y))
        # print('x0', x0)
        # print('index',self.nz_index_out, self.nz_index_in)

        sol =  solver(x0, fun=self.fun, fun_jac=self.fun_jac, g=self.stop_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return)

        self._set_solved_problem(sol)


    def _set_solved_problem(self, solution):
        if ~self.full_return:
            self.r_xy = solution 
            
            self.r_x = self.r_xy[:self.rnz_n_out]
            self.r_y = self.r_xy[self.rnz_n_out:]

            self.x = self.r_x[self.r_invert_dseq]
            self.y = self.r_y[self.r_invert_dseq]
        
        
    def degree_reduction(self):
        self.dseq = np.array(list(zip(self.dseq_out, self.dseq_in)))
        self.r_dseq, self.r_invert_dseq, self.r_multiplicity = np.unique(self.dseq, return_index=False, return_inverse=True, return_counts=True, axis=0)

        self.rnz_dseq_out = self.r_dseq[:,0]
        self.rnz_dseq_in = self.r_dseq[:,1]

        self.nz_index_out = np.nonzero(self.rnz_dseq_out)[0]
        self.nz_index_in = np.nonzero(self.rnz_dseq_in)[0]

        self.rnz_n_out = self.rnz_dseq_out.size
        self.rnz_n_in = self.rnz_dseq_in.size
        self.rnz_dim = self.rnz_n_out + self.rnz_n_in

        self.is_reduced = True
        
    def _initialize_problem(self, model, method):
        if ~self.is_reduced:
           self.degree_reduction()

        self._set_initial_guess(model, method)

        if method in ['newton', 'quasinewton']:
            d_fun = {
                    'dcm': loglikelihood_prime_dcm,
                    }

            d_fun_jac = {
                    'dcm': loglikelihood_hessian_diag_dcm,
                    }

            d_fun_stop = {
                    'dcm': loglikelihood_dcm,
                    }

            self.args = (self.rnz_dseq_out, self.rnz_dseq_in, self.nz_index_out, self.nz_index_in, self.r_multiplicity)

            self.fun = lambda x: -d_fun[model](x, self.args)
            self.fun_jac = lambda x: -d_fun_jac[model](x, self.args)
            self.stop_fun = lambda x: -d_fun_stop[model](x, self.args)

        elif method in ['fixed-point']:
            d_fun = {
                    'dcm': iterative_dcm,
                    }

            d_fun_stop = {
                    'dcm': loglikelihood_dcm,
                    }

            self.args = (self.rnz_dseq_out, self.rnz_dseq_in, self.nz_index_out, self.nz_index_in, self.r_multiplicity)

            self.fun = lambda x: -d_fun[model](x, self.args)
            self.fun_jac = None 
            self.stop_fun = lambda x: -d_fun_stop[model](x, self.args)


    def _set_initial_guess(self, model, method):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if self.initial_guess is None:
            self.r_x = self.rnz_dseq_out / (np.sqrt(self.n_edges) + 1)  # This +1 increases the stability of the solutions.
            self.r_y = self.rnz_dseq_in / (np.sqrt(self.n_edges) + 1)
        elif self.initial_guess == 'random':
            self.r_x = np.random.rand(self.rnz_n_out).astype(np.float64)
            self.r_y = np.random.rand(self.rnz_n_in).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.r_x = 0.5*np.ones(self.rnz_n_out, dtype=np.float64)  # All probabilities will be 1/2 initially
            self.r_y = 0.5*np.ones(self.rnz_n_in, dtype=np.float64)
        elif self.initial_guess == 'degrees':
            self.r_x = self.rnz_dseq_out.astype(np.float64)
            self.r_y = self.rnz_dseq_in.astype(np.float64)

        self.r_x[self.rnz_dseq_out == 0] = 0
        self.r_y[self.rnz_dseq_in == 0] = 0


    def solution_error(self):
        sol = np.concatenate((self.x, self.y))
        ex_k_out = expected_out_degree_dcm(sol)
        ex_k_in = expected_in_degree_dcm(sol)
        ex_k = np.concatenate((ex_k_out, ex_k_in))
        k = np.concatenate((self.dseq_out, self.dseq_in))
        # print(k, ex_k)
        self.expected_dseq = ex_k
        self.error = np.linalg.norm(ex_k - k)


    def _set_initial_guess_CReAMa(self, model, method):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if self.initial_guess is None:
            self.b_out = (self.out_strength>0).astype(int) / self.out_strength.sum()  # This +1 increases the stability of the solutions.
            self.b_in = (self.r_in_strength>0).astype(int) / self.r_in_strength.sum()
        elif self.initial_guess == 'strengths':
            self.b_out = (self.out_strength>0).astype(int) / (self.out_strength + 1)
            self.b_in = (self.in_strength>0).astype(int) / (self.in_strength + 1)


    def _initialize_problem_CReAMa(self, model, method):
        if ~self.is_reduced:
            self.degree_reduction()
        
        if model=='CReAMa':
            self._set_initial_guess_CReAMa(model, method)
        else:
            self._set_initial_guess(model,method)
        if method in ['newton', 'quasinewton']:
            d_fun = {
                    'dcm': loglikelihood_prime_dcm,
                    'CReAMa': loglikelihood_prime_CReAMa,
                    }

            d_fun_jac = {
                    'dcm': loglikelihood_hessian_diag_dcm,
                    'CReAMa': loglikelihood_hessian_diag_CReAMa,
                    }

            d_fun_stop = {
                    'dcm': loglikelihood_dcm,
                    'CReAMa': loglikelihood_CReAMa,
                    }
            if model=='CReAMa':
                self.args = (self.out_strength, self.in_strength, self.adjacency, self.nz_index_sout, self.nz_index_sin)
            else:
                self.args = (self.rnz_dseq_out, self.rnz_dseq_in, self.nz_index_out, self.nz_index_in, self.r_multiplicity)
            self.fun = lambda x: -d_fun[model](x, self.args)
            self.fun_jac = lambda x: -d_fun_jac[model](x, self.args)
            self.stop_fun = lambda x: -d_fun_stop[model](x, self.args)
        d_pmatrix = {
                    'dcm': pmatrix_dcm
                    }
        if model != 'CReAMa':
            self.args_p = (self.n_nodes, np.nonzero(self.dseq_out)[0], np.nonzero(self.dseq_in)[0])
            self.fun_pmatrix = lambda x: d_pmatrix[model](x,self.args_p)


    def _solve_problem_CReAMa(self, initial_guess=None, model='CReAMa', adjacency='dcm', method='quasinewton', max_steps=100, full_return=False, verbose=False):
        if model == 'CReAMa':
            if not isinstance(adjacency,(list,np.ndarray,str)):
                raise ValueError('adjacency must be a matrix or a method')
            elif isinstance(adjacency,str):
                self.initial_guess = initial_guess
                self._initialize_problem_CReAMa(adjacency, method)
                x0 = np.concatenate((self.r_x, self.r_y))
                sol =  solver(x0, fun=self.fun, fun_jac=self.fun_jac, g=self.stop_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return)
                self._set_solved_problem(sol)
                adjacency = self.fun_pmatrix(np.concatenate([self.x,self.y]))
            elif isinstance(adjacency,list):
                adjacency = np.array(adjacency)

            if adjacency.shape[0] != adjacency.shape[1]:
                raise ValueError(r'adjacency matrix must be $n \times n$')
            
            self.initial_guess = 'strengths'
            self._initialize_problem_CReAMa(model,method)
            x0 = np.concatenate((self.b_out, self.b_in))
            
            sol = solver(x0, fun=self.fun, fun_jac=self.fun_jac, g=self.stop_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return)

            self.beta_out = sol[:self.n_nodes]
            self.beta_in = sol[self.n_nodes:]
            
        else:
            
            self.initial_guess = initial_guess
            self._initialize_problem(model, method)
            x0 = np.concatenate((self.r_x, self.r_y))
            # print('x0', x0)
            # print('index',self.nz_index_out, self.nz_index_in)

            sol =  solver(x0, fun=self.fun, fun_jac=self.fun_jac, g=self.stop_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return)

            self._set_solved_problem(sol)


    def solve_tool(self, model, method, initial_guess=None, adjacency=None, max_steps=100, full_return=False, verbose=False):
        """ function to switch around the various problems
        """
        if model in ['dcm']:
            self._solve_problem(initial_guess=initial_guess, model=model, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
        elif model in ['CReAMa']:
            self._solve_problem_CReAMa(initial_guess=initial_guess, model=model, adjacency=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)


