import numpy as np
import scipy.sparse
from numba import jit
import time


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
        return np.sum(a , 1).A1


def pmatrix_cm(x, args):
	n = args[0]
	nz_index = args[1]
	f = np.zeros(shape=(n, n), dtype=np.float64)
	for i in np.arange(n):
		for j in np.arange(i+1, n):
			aux = x[i]*x[j]
			aux1 = aux/(1+aux)
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
                fx += (c[j]-1) * (x[j]/(1+x[j]*x[i]))
            else:
                fx += (c[j]) * (x[j]/(1+x[j]*x[i]))
        if fx:
            f[i] = k[i]/fx
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
                f -= (c[i]*(c[i]-1)*np.log(1+(x[i])**2))/2
            else:
                f -= (c[i]*c[j]*np.log(1+x[i]*x[j]))/2
    return f


@jit(nopython=True)
def loglikelihood_prime_cm(x, args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        f[i] += k[i]/x[i]
        for j in np.arange(n):
            if i == j:
                f[i] -= (c[j]-1) * (x[j]/(1+(x[j]**2)))
            else:
                f[i] -= c[j] * (x[j]/(1+x[i]*x[j]))
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
                aux_f = - k[i]/(x[i]*x[i])
                for h in range(n):
                    if i == h:
                        aux = 1 + x[h]*x[h]
                        aux_f += ((x[h]*x[h])/(aux*aux))*(c[h]-1)
                    else:
                        aux = 1 + x[i]*x[h]
                        aux_f += ((x[h]*x[h])/(aux*aux))*c[h]
            else:
                aux = 1+x[i]*x[j]
                aux_f = ((x[j]*x[j]-aux)/(aux*aux))*c[j]

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
        f[i] - k[i]/(x[i]*x[i])
        for j in np.arange(n):
            if i == j:
                aux = 1 + x[j]*x[j]
                f[i] += ((x[j]*x[j])/(aux*aux))*(c[j]-1)
            else:
                aux = 1 + x[i]*x[j]
                f[i] += ((x[j]*x[j])/(aux*aux))*c[j]
    return i


@jit(nopython=True)
def iterative_CReAMa(beta, args):
    s = args[0]
    adj = args[1]
    nz_index = args[2]

    n = len(s)

    f = np.zeros_like(s, dtype=np.float64)

    for i in np.arange(n):
        for j in np.arange(n):
            if (i != j) and (adj[i, j] != 0):
                f[i] -= (adj[i, j] / (1+(beta[j]/beta[i])))/s[i]
    return f


@jit(nopython=True)
def loglikelihood_CReAMa(beta, args):
	s = args[0]
	adj = args[1]
	nz_index = args[2]

	n = len(s)

	f = 0.0
	for i in np.arange(n):
		f -= s[i] * beta[i]
		for j in np.arange(0, i):
			if adj[i, j] != 0:
				f += adj[i, j] * np.log(beta[i]+beta[j])
	return f


@jit(nopython=True)
def loglikelihood_prime_CReAMa(beta, args):
	s = args[0]
	adj = args[1]
	nz_index = args[2]

	n = len(s)

	f = np.zeros_like(s, dtype=np.float64)
	for i in np.arange(n):
		f[i] -= s[i]
		for j in np.arange(n):
			if (i != j) and adj[i, j] != 0:
				f[i] += adj[i, j] / (beta[i] + beta[j])
	return f


@jit(nopython=True)
def loglikelihood_hessian_CReAMa(beta, args):
    s = args[0]
    adj = args[1]
    nz_index = args[2]

    n = len(s)

    f = np.zeros(shape=(n, n), dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(n):
            if i !=j:
                aux_f = 0.0
                if adj[i, j] != 0: 
                    aux = beta[i] + beta[j]
                    aux_f = - adj[i, j]/(aux*aux)
                    f[i,j] = aux_f
    for i in np.arange(n):
        for j in np.arange(n):
            if (i!=j) and (adj[i,j]!=0):
                f[i,i] -= adj[i,j]/(beta[i]+beta[j])**2
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_CReAMa(beta,args):
	s = args[0]
	adj = args[1]
	nz_index = args[2]

	n = len(s)

	f = np.zeros_like(s, dtype=np.float64)
	for i in np.arange(n):

		for j in np.arange(n):
			if (i!=j) and (adj[i,j]!=0):
				f[i] -= adj[i,j]/(beta[i]+beta[j])**2
	return f


@jit(nopython=True)
def iterative_ecm(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]

    f = np.zeros(2*n, dtype=np.float64)
    for i in np.arange(n):
        fx = 0.0
        fy = 0.0
        for j in np.arange(n):
            if i!=j:
                aux1 = x[i] * x[j]
                aux2 = y[i] * y[j]
                fx += (x[i] * aux2)/(1-aux2+aux1*aux2)
                fy += (aux1 * y[j])/((1-aux2)*(1-aux2+aux1*aux2))
        if fx:
            f[i] = k[i] / fx
        if fy:
            f[i+n] = s[i] / fy 

    return f



@jit(nopython=True)
def loglikelihood_ecm(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = 0.0
    for i in np.arange(n):
        f += k[i] * np.log(x[i]) + s[i] * np.log(y[i])
        for j in np.arange(i+1,n):
            aux = y[i] * y[j]
            f += np.log((1-aux)/(1-aux+x[i]*x[j]*aux))
    return f


@jit(nopython=True)
def loglikelihood_prime_ecm(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(2*n,dtype = np.float64)
    for i in np.arange(n):
        f[i] += k[i]/x[i]
        f[i+n] += s[i]/y[i]
        for j in np.arange(n):
            if (i!=j):
                aux1 = x[i]*x[j]
                aux2 = y[i]*y[j]
                f[i] -= (x[j]*aux2)/(1-aux2+aux1*aux2)
                f[i+n] -= (aux1*y[j])/((1-aux2)*(1-aux2+aux1*aux2))
    return f


@jit(nopython=True)
def loglikelihood_hessian_ecm(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(shape=(n,n),dtype=np.float64)

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_ecm(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(2*n,dtype=np.float64)

    for i in np.arange(n):
        f[i] -= k[i]/(x[i]*x[i])
        f[i+n] -= s[i]/(y[i]*y[i])
        for j in np.arange(n):
            if j!=i:
                aux1 = x[i]*x[j]
                aux2 = y[i]*y[j]
                aux3 = (1 - aux2)**2
                aux4 = (1- aux2 + aux1 * aux2)**2
                f[i] += ((x[j]*aux2)**2)/aux4
                f[i+n] += (aux1*y[j] * (aux1*y[j] * (1-2*aux2) - 2*y[j]*(1-aux2)))/(aux3*aux4)
    return f


def solver(x0, fun, step_fun, fun_jac=None, tol=1e-6, eps=1e-3, max_steps=100, method='newton', verbose=False, regularise=True, full_return = False, linsearch = True):
    """Find roots of eq. f = 0, using newton, quasinewton or dianati.
    """

    tic_all = time.time()
    toc_init = 0
    tic = time.time()

    # algorithm
    beta = .5  # to compute alpha
    n_steps = 0
    x = x0  # initial point

    f = fun(x)
    norm = np.linalg.norm(f)
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
            dx = np.linalg.solve(B, - f)
        elif method == 'quasinewton':
            dx = - f/B
        elif method == 'fixed-point':
            dx = f - x
        toc_dx += time.time() - tic

        # backtraking line search
        tic = time.time()
        if linsearch:
            alfa = 1 
            i = 0
            # TODO: fun(x) non e' il graident di step_funx
            # TODO: check dianati fornisce una direzione di discesa 

            """
            s_new = np.linalg.norm(fun(x+alfa*dx)-x-alfa*dx)
            s_old = np.linalg.norm(fun(x)-x)
            while sufficient_decrease_condition(s_old, \
                s_new, alfa, fun(x), dx) == False and i<50:
            """
            s_old = step_fun(x)
            while sufficient_decrease_condition(s_old, \
                step_fun(x + alfa*dx), alfa, f, dx) == False and i<50:
                alfa *= beta
                i +=1
        else:
            """
            if True:
                alfa = 0.1
                eps2=1e-2
                alfa0 = (eps2-1)*x/dx
                for a in alfa0:
                    if a>=0:
                        alfa = min(alfa, a)
            """
            alfa = 1


        toc_alfa += time.time() - tic

        tic = time.time()
        # solution update
        # direction= dx@fun(x).T
        x = x + alfa*dx
        toc_update += time.time() - tic

        f = fun(x)

        # stopping condition computation
        norm = np.linalg.norm(f)
        diff = np.linalg.norm(x - x_old)

        if full_return:
            norm_seq.append(norm)

        # step update
        n_steps += 1

        if verbose == True:
            print('step {}'.format(n_steps))
            print('alpha = {}'.format(alfa))
            print('fun = {}'.format(f))
            print('dx = {}'.format(dx))
            print('x = {}'.format(x))
            print('|f(x)| = {}'.format(norm))

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
    # c1 = 0
    
    # print ('f_old',f_old)
    # print ('c1',c1)
    # print('alpha',alpha)
    # print ('grad_f',grad_f)
    # print('p.T',p.T)

    sup = f_old + c1 *alpha*grad_f@p.T
    # print(alpha, f_new, sup)
    return bool(f_new < sup)


def hessian_regulariser_function(B, eps):
    """Trasform input matrix in a positive defined matrix
    input matrix should be numpy.array
    """
    eps = 1e-8
    B = (B + B.transpose())*0.5  # symmetrization
    l, e = np.linalg.eigh(B)
    ll = np.array([0 if li>eps else eps-li for li in l])
    Bf = e @ (np.diag(ll) + np.diag(l)) @ e.transpose()
    # lll, eee = np.linalg.eigh(Bf)
    # debug check
    # print('B regularised eigenvalues =\n {}'.format(lll))
    return Bf


@jit(nopython=True)
def expected_degree_cm(sol):
	ex_k = np.zeros_like(sol, dtype=np.float64)
	n = len(sol)
	for i in np.arange(n):
		for j in np.arange(n):
			if i!=j:
				aux = sol[i]*sol[j]
				ex_k[i] += aux/(1+aux)
	return ex_k


@jit(nopython=True)
def expected_strength_CReAMa(sol,adj):
	ex_s = np.zeros_like(sol, dtype=np.float64)
	n = len(sol)
	for i in np.arange(n):
		for j in np.arange(n):
			if (i!=j) and (adj[i,j]!=0):
				ex_s[i] += adj[i,j]/(sol[i]+sol[j])
	return ex_s


@jit(nopython=True)
def expected_ecm(sol):
    ex_ks = np.zeros_like(sol, dtype=np.float64)
    n = int(len(sol)/2)
    for i in np.arange(n):
        for j in np.arange(n):
            aux1 = x[i]*x[j]
            aux2 = y[i]*y[j]
            aux3 = (aux1*aux2)/(1-aux2+aux1*aux2)
            ex_ks[i] += aux3
            ex_ks[i+n] += aux3/(1-aux2)
    return ex_ks

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
    unique_nodes, degree_seq = np.unique(np.concatenate(
        (edgelist['source'], edgelist['target'])), return_counts=True)
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
    if len(edgelist[0]) == 3:
       	aux_edgelist = np.concatenate((edgelist_new['source'],edgelist_new['target']))
        aux_weights = np.concatenate((edgelist_new['weigth'],edgelist_new['weigth']))
        strength_seq = np.array(
        						   [aux_weights[aux_edgelist == i].sum() for i in unique_nodes])
        return edgelist_new, degree_seq, strength_seq, nodes_dict
    return edgelist_new, degree_seq, nodes_dict


class UndirectedGraph:
    def __init__(self, adjacency=None, edgelist=None, degree_sequence=None, strength_sequence=None):
        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.sparse_adjacency = None
        self.edgelist = None
        self.dseq = None
        self.strength_sequence = None
        self.nodes_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_weighted = False
        self._initialize_graph(adjacency=adjacency, edgelist=edgelist,
                               degree_sequence=degree_sequence, strength_sequence=strength_sequence)
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
        self.error_strength = None
        self.relative_error_strength = None
        self.full_return = False
        self.last_model = None

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
                if np.sum(adjacency<0):
                    raise TypeError('The adjacency matrix entries must be positive.')
                if isinstance(adjacency, list): # Cast it to a numpy array: if it is given as a list it should not be too large
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.adjacency = adjacency
                    self.is_sparse = True
                if np.sum(adjacency)==np.sum(adjacency>0):
                    self.dseq = degree(adjacency)
                else:
                    self.dseq = degree(adjacency)
                    self.strength_sequence = strength(adjacency)
                    self.nz_index = np.nonzero(self.strength_sequence)[0]
                    self.is_weighted = True
                    
                # self.edgelist, self.deg_seq = edgelist_from_adjacency(adjacency)
                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError('The edgelist must be passed as a list or numpy array.')
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        'This is not an edgelist. An edgelist must be a list or array of couples of nodes with optional weights. Is this an adjacency matrix?')
                elif len(edgelist[0])==2:
                    self.edgelist, self.dseq, self.nodes_dict = edgelist_from_edgelist(edgelist)
                else:
                    self.edgelist, self.dseq, self.strength_sequence, self.nodes_dict = edgelist_from_edgelist(edgelist)
                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)
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
                    self.n_nodes = int(len(degree_sequence))
                    self.dseq = degree_sequence
                    self.n_edges = np.sum(self.dseq)
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
                            if len(strength_sequence) != len(degree_sequence):
                                raise ValueError('Degrees and strengths arrays must have same length.')
                            self.n_nodes = int(len(strength_sequence))
                            self.strength_sequence = strength_sequence
                            self.nz_index = np.nonzero(self.strength_sequence)[0]
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
                    self.n_nodes = int(len(strength_sequence))
                    self.strength_sequence = strength_sequence
                    self.nz_index = np.nonzero(self.strength_sequence)[0]
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


    def _solve_problem(self, initial_guess=None, model='cm', method='quasinewton', max_steps=100, full_return=False, verbose=False, linsearch=True):
        
        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self._initialize_problem(model, method)
        x0 = self.x0 

        sol =  solver(x0, fun=self.fun, fun_jac=self.fun_jac, step_fun=self.step_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return, linsearch=linsearch)

        self._set_solved_problem(sol)


    def _set_solved_problem_cm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
        else:
            self.r_xy = solution 
            
        self.r_x = self.r_xy

        self.x = self.r_x[self.r_invert_dseq]
 

    def _set_solved_problem(self, solution):
        model = self.last_model
        if model == 'cm':
            self._set_solved_problem_cm(solution)
        elif model == 'ecm':
            self._set_solved_problem_ecm(solution)
        elif model == 'CReAMa':
            self._set_solved_problem_CReAMa(solution)

        
    def degree_reduction(self):
        self.r_dseq, self.r_invert_dseq, self.r_multiplicity = np.unique(self.dseq, return_index=False, return_inverse=True, return_counts=True, axis=0)
        self.rnz_n = self.r_dseq.size
        self.is_reduced = True


    def _set_initial_guess(self, model):

        if model == 'cm':
            self._set_initial_guess_cm()
        elif model == 'ecm':
            self._set_initial_guess_ecm()
        elif model == 'CReAMa':
            self._set_initial_guess_CReAMa()


    def _set_initial_guess_cm(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.

        if ~self.is_reduced:
            self.degree_reduction()

        if self.initial_guess is None:
            self.r_x = self.r_dseq / (np.sqrt(self.n_edges) + 1)  # This +1 increases the stability of the solutions.
        elif self.initial_guess == 'random':
            self.r_x = np.random.rand(self.rnz_n).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.r_x = 0.5*np.ones(self.rnz_n, dtype=np.float64)  # All probabilities will be 1/2 initially
        elif self.initial_guess == 'degrees':
            self.r_x = self.r_dseq.astype(np.float64)

        self.r_x[self.r_dseq == 0] = 0

        self.x0 = self.r_x


    def _set_initial_guess_CReAMa(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if self.initial_guess is None:
            self.beta = (self.strength_sequence>0).astype(float) / self.strength_sequence.sum()
        elif self.initial_guess == 'strengths':
            self.beta = (self.strength_sequence>0).astype(float) / (self.strength_sequence + 1)

        self.x0 = self.beta


    def _set_initial_guess_ecm(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if self.initial_guess is None:
            self.x = self.dseq.astype(float) / (self.n_edges + 1)
            self.y = self.strength_sequence.astype(float) / self.strength_sequence.sum()  # This +1 increases the stability of the solutions.
        elif self.initial_guess == 'strengths':
            self.x = self.dseq.astype(float) / (self.dseq + 1)
            self.y = self.strength_sequence.astype(float) / (self.strength_sequence + 1)
        elif self.initial_guess == 'random':
            self.x = np.random.rand(self.n_nodes).astype(np.float64)
            self.y = np.random.rand(self.n_nodes).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.x = 0.9*np.ones(self.n_nodes, dtype=np.float64)  # All probabilities will be 1/2 initially
            self.y = 0.9*np.ones(self.n_nodes, dtype=np.float64)


    # DA SISTEMARE
    def solution_error(self):
        if self.last_model in ['cm','CReAMa']:
            if (self.x is not None):
                ex_k = expected_degree_cm(self.x)
                # print(k, ex_k)
                self.expected_dseq = ex_k
                self.error = np.linalg.norm(ex_k - self.dseq)
            if (self.beta is not None):
                ex_s = expected_strength_CReAMa(self.beta,self.adjacency_CReAMa)
                self.expected_stregth_seq = ex_s
                self.error_strength = np.linalg.norm(ex_s - self.strength_sequence)
                self.relative_error_strength = self.error_strength/self.strength_sequence.sum()
        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ['ecm']:
                sol = np.concatenate((self.x, self.y))
                ex = expected_ecm(sol)
                k = np.concatenate((self.dseq, self.strength_sequence))
                self.expected_dseq = ex[:self.n_nodes]
                self.expected_stregth_seq = ex[self.n_nodes:]
                self.error = np.linalg.norm(ex - k)
                self.relative_error_strength = self.error/self.out_strength.sum()
    

    def _set_args(self, model):

        if model=='CReAMa':
            self.args = (self.strength_sequence, self.adjacency_CReAMa, self.nz_index)
        elif model == 'cm':
            self.args = (self.r_dseq, self.r_multiplicity)
        elif model == 'ecm':
            self.args = (self.dseq, self.strength_sequence) 


    def _initialize_problem(self, model, method):
        
        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = '-'
        mod_met = mod_met.join([model,method])

        d_fun = {
                'cm-newton': lambda x: -loglikelihood_prime_cm(x,self.args),
                'cm-quasinewton': lambda x: -loglikelihood_prime_cm(x,self.args),
                'cm-fixed-point': lambda x: iterative_cm(x,self.args),


                'CReAMa-newton': lambda x: -loglikelihood_prime_CReAMa(x,self.args),
                'CReAMa-quasinewton': lambda x: -loglikelihood_prime_CReAMa(x,self.args),
                'CReAMa-fixed-point': lambda x: -iterative_CReAMa(x,self.args),

                'ecm-newton': lambda x: -loglikelihood_prime_ecm(x,self.args),
                'ecm-quasinewton': lambda x: -loglikelihood_prime_ecm(x,self.args),
                'ecm-fixed-point': lambda x: iterative_ecm(x,self.args),
                }

        d_fun_jac = {
                    'cm-newton': lambda x: -loglikelihood_hessian_cm(x,self.args),
                    'cm-quasinewton': lambda x: -loglikelihood_hessian_diag_cm(x,self.args),
                    'cm-fixed-point': None,

                    'CReAMa-newton': lambda x: -loglikelihood_hessian_CReAMa(x,self.args),
                    'CReAMa-quasinewton': lambda x: -loglikelihood_hessian_diag_CReAMa(x,self.args),
                    'CReAMa-fixed-point': None,

                    'ecm-newton': lambda x: -loglikelihood_hessian_ecm(x,self.args),
                    'ecm-quasinewton': lambda x: -loglikelihood_hessian_diag_ecm(x,self.args),
                    'ecm-fixed-point': None,
                    }
        d_fun_stop = {
                     'cm-newton': lambda x: -loglikelihood_cm(x,self.args),
                     'cm-quasinewton': lambda x: -loglikelihood_cm(x,self.args),
                     'cm-fixed-point': lambda x: -loglikelihood_cm(x,self.args),

                     'CReAMa-newton': lambda x: -loglikelihood_CReAMa(x,self.args),
                     'CReAMa-quasinewton': lambda x: -loglikelihood_CReAMa(x,self.args),
                     'CReAMa-fixed-point': lambda x: -loglikelihood_CReAMa(x,self.args),

                     'ecm-newton': lambda x: -loglikelihood_ecm(x,self.args),
                     'ecm-quasinewton': lambda x: -loglikelihood_ecm(x,self.args),
                     'ecm-fixed-point': lambda x: -loglikelihood_ecm(x,self.args),
                     }
        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.step_fun = d_fun_stop[mod_met]
        except:    
            raise ValueError('Method must be "newton","quasi-newton", or "fixed-point".')
        
        # TODO: mancano metodi
        d_pmatrix = {
                    'cm': pmatrix_cm
                    }

        if model in ['cm']:
            self.args_p = (self.n_nodes, np.nonzero(self.dseq)[0])
            self.fun_pmatrix = lambda x: d_pmatrix[model](x,self.args_p)
    
    
    def _solve_problem_CReAMa(self, initial_guess=None, model='CReAMa', adjacency='cm', method='quasinewton', max_steps=100, full_return=False, verbose=False):
        if not isinstance(adjacency,(list,np.ndarray,str)):
            raise ValueError('adjacency must be a matrix or a method')
        elif isinstance(adjacency,str):
            self._solve_problem(initial_guess=initial_guess, model=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
            if self.is_sparse:
                self.adjacency_CReAMa = self.x
            else:
                self.adjacency_CReAMa = self.fun_pmatrix(self.x)
        elif isinstance(adjacency,list):
            self.adjacency_CReAMa = np.array(adjacency)
        elif isinstance(adjacency,np.ndarray):
            self.adjacency_CReAMa = adjacency

        #if self.adjacency_CReAMa.shape[0] != self.adjacency_CReAMa.shape[1]:
        #    raise ValueError(r'adjacency matrix must be $n \times n$')

        self.last_model = model
        self.full_return = full_return
        self.initial_guess = 'strengths'
        self._initialize_problem(model,method)
        x0 = self.x0 
            
        sol = solver(x0, fun=self.fun, fun_jac=self.fun_jac, step_fun=self.step_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return)
            
        self._set_solved_problem_CReAMa(sol)


    def _set_solved_problem_CReAMa(self, solution):
        if self.full_return:
            self.beta = solution[0]
            self.comput_time_creama = solution[1]
            self.n_steps_creama = solution[2]
            self.norm_seq_creama = solution[3]
        
        else:
            self.beta = solution


    def solve_tool(self, model, method, initial_guess=None, adjacency=None, max_steps=100, full_return=False, verbose=False):
        """ function to switch around the various problems
        """
        # TODO: aggiungere tutti i metodi
        if model in ['cm', 'ecm']:
            self._solve_problem(initial_guess=initial_guess, model=model, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
        elif model in ['CReAMa']:
            self._solve_problem_CReAMa(initial_guess=initial_guess, model=model, adjacency=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)

