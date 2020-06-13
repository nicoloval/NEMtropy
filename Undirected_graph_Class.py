import numpy as np
import scipy.sparse
from numba import jit
import time


def pmatrix_cm(x,args):
	n = args[0]
	f = np.ones(shape=(n,n),dtype=np.float64)
	for i in np.arange(n):
		for j in np.arange(i+1,n):
			aux = x[i]*x[j]
			aux1 = aux/(1+aux)
			f[i,j] = aux1
			f[j,i] = aux1
	return f


@jit(nopython=True)
def iterative_cm(x,args):
	k = args[0]
    c = args[1]
    n = len(k)
    f = np.ones_like(k,dtype=np.float64)
    for i in np.arange(n):
    	fx=0
    	for j in np.arange(n):
    		if i==j:
    			fx += c[j] * (x[j]/(1+x[j]*x[i]))
    		else:
    			fx += (c[j]-1) * (x[j]/(1+x[j]*x[i]))

    	f[i] = k[i]/fx
    return f


@jit(nopython=True)
def loglikelihood_cm(x,args):
    k = args[0]
    c = args[1]
    n = len(k)
    f=0.0
    for i in np.arange(n):
        f += c[i] * k[i] * np.log(x[i])
        for j in np.arange(n):
            if i == j:
                f -= (c[i]*(c[i]-1)*np.log(1+(x[i])**2))/2
            else:
                f -= (c[i]*c[j]*np.log(1+x[i]*x[j]))/2
    return f


@jit(nopython=True)
def loglikelihood_prime_cm(x,args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.ones_like(k,dtype=np.float64)
    for i in np.arange(n):
        f[i] += k[i]/x[i]
        for j in np.arange(n):
            if i==j:
                f[i] -= (c[j]-1) * (x[j]/(1+(x[j]**2)))
            else:
                f[i] -= c[j] * (x[j]/(1+x[i]*x[j]))
    return f


@jit(nopython=True)
def loglikelihood_hessian_cm(x,args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.ones(shape=(n,n),dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(i,n):
            if i==j:
                aux_f = - k[i]/(x[i]*x[i])
                for h in range(n):
                    if i==h:
                        aux = 1 + x[h]*x[h]
                        aux_f += ((x[h]*x[h])/(aux*aux))*(c[h]-1)
                    else:
                        aux = 1 + x[i]*x[h]
                        aux_f += ((x[h]*x[h])/(aux*aux))*c[h]
            else:
                aux = 1+x[i]*x[j]
                aux_f = ((x[j]*x[j]-aux)/(aux*aux))*c[j]
            
            f[i,j] = aux_f
            f[j,i] = aux_f
    return f

@jit(nopython=True)
def loglikelihood_hessian_diag_cm(x,args):
	k = args[0]
    c = args[1]
    n = len(k)
    f = np.ones(n,dtype=np.float64)
    for i in np.arange(n):
    	f[i] - k[i]/(x[i]*x[i])
    	for j in np.arange(n):
    		if i==j:
    			aux = 1 + x[j]*x[j]
                f[i] += ((x[j]*x[j])/(aux*aux))*(c[j]-1)
    		else:
    			aux = 1 + x[i]*x[j]
                f[i] += ((x[j]*x[j])/(aux*aux))*c[j]
    return i


@jit(nopython=True)
def loglikelihood_CReAMa(x,args):


@jit(nopython=True)
def loglikelihood_prime_CReAMa(x,args):


@jit(nopython=True)
def loglikelihood_hessian_CReAMa(x,args):


@jit(nopython=True)
def loglikelihood_hessian_diag_CReAMa(x,args):


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
                if (adjacency<0).any():
                    raise TypeError('The adjacency matrix entries must be positive.')
                if isinstance(adjacency, list): # Cast it to a numpy array: if it is given as a list it should not be too large
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.sparse_adjacency = adjacency
                if np.sum(adjacency)==np.sum(adjacency>0):
                    self.dseq = np.sum(adjacency, axis=0)
                else:
                    self.dseq = np.sum(adjacency>0, axis=0)
                    self.strength_sequence = np.sum(adjacency, axis=0)
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

        sol =  solver(x0, fun=self.fun, fun_jac=self.fun_jac, g=self.stop_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return, linsearch=linsearch)

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
            self._set_solved_problem_decm(solution)
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

    ##### DA SISTEMARE
    def solution_error(self):
        if self.last_model in ['cm','CReAMa']:
            if (self.x is not None):
                sol = np.concatenate((self.x, self.y))
                ex_k = expected__cm(sol)
                # print(k, ex_k)
                self.expected_dseq = ex_k
                self.error = np.linalg.norm(ex_k - self.dseq)
            if (self.b_out is not None) and (self.b_in is not None):
                sol = np.concatenate([self.b_out,self.b_in])
                ex_s_out = expected_out_strength_CReAMa(sol,self.adjacency)
                ex_s_in = expected_in_stregth_CReAMa(sol,self.adjacency)
                ex_s = np.concatenate([ex_s_out,ex_s_in])
                s = np.concatenate([self.out_strength,self.in_strength])
                self.expected_stregth_seq = ex_s
                self.error_strength = np.linalg.norm(ex_s - s)
                self.relative_error_strength = self.error_strength/self.out_strength.sum()
        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ['ecm']:
                sol = np.concatenate((self.x, self.y, self.b_out, self.b_in))
                ex = expected_decm(sol)
                k = np.concatenate((self.dseq_out, self.dseq_in, self.out_strength, self.in_strength))
                self.expected_dseq = ex[:2*self.n_nodes]
                self.expected_stregth_seq = ex[2*self.n_nodes:]
                self.error = np.linalg.norm(ex - k)
                self.relative_error_strength = self.error/self.out_strength.sum()
    

    def _set_args(self, model):

        if model=='CReAMa':
            self.args = (self.out_strength, self.in_strength, self.adjacency, self.nz_index_sout, self.nz_index_sin)
        elif model == 'cm':
            self.args = (self.dseq, self.r_multiplicity)
        elif model == 'ecm':
            self.args = (self.dseq_out, self.dseq_in, self.out_strength, self.in_strength) 


    def _initialize_problem(self, model, method):
        
        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = '-'
        mod_met = mod_met.join([model,method])

        d_fun = {
                'cm-newton': lambda x: -loglikelihood_prime_cm(x,self.args),
                'cm-quasinewton': lambda x: -loglikelihood_prime_cm(x,self.args),
                'cm-fixed-point': lambda x: -iterative_cm(x,self.args),


                'CReAMa-newton': lambda x: -loglikelihood_prime_CReAMa(x,self.args),
                'CReAMa-quasinewton': lambda x: -loglikelihood_prime_CReAMa(x,self.args),
                'CReAMa-fixed-point': lambda x: -iterative_CReAMa(x,self.args),

                'ecm-newton': lambda x: -loglikelihood_prime_decm(x,self.args),
                'ecm-quasinewton': lambda x: -loglikelihood_prime_decm(x,self.args),
                'ecm-fixed-point': lambda x: iterative_decm(x,self.args),
                }

        d_fun_jac = {
                    'cm-newton': lambda x: -loglikelihood_hessian_cm(x,self.args),
                    'cm-quasinewton': lambda x: -loglikelihood_hessian_diag_cm(x,self.args),
                    'cm-fixed-point': None,

                    'CReAMa-newton': lambda x: -loglikelihood_hessian_CReAMa(x,self.args),
                    'CReAMa-quasinewton': lambda x: -loglikelihood_hessian_diag_CReAMa(x,self.args),
                    'CReAMa-fixed-point': None,

                    'ecm-newton': lambda x: -loglikelihood_hessian_decm(x,self.args),
                    'ecm-quasinewton': lambda x: -loglikelihood_hessian_diag_decm(x,self.args),
                    'ecm-fixed-point': None,
                    }
        d_fun_stop = {
                     'cm-newton': lambda x: -loglikelihood_cm(x,self.args),
                     'cm-quasinewton': lambda x: -loglikelihood_cm(x,self.args),
                     'cm-fixed-point': lambda x: -loglikelihood_cm(x,self.args),

                     'CReAMa-newton': lambda x: -loglikelihood_CReAMa(x,self.args),
                     'CReAMa-quasinewton': lambda x: -loglikelihood_CReAMa(x,self.args),
                     'CReAMa-fixed-point': lambda x: -loglikelihood_CReAMa(x,self.args),

                     'ecm-newton': lambda x: -loglikelihood_decm(x,self.args),
                     'ecm-quasinewton': lambda x: -loglikelihood_decm(x,self.args),
                     'ecm-fixed-point': lambda x: -loglikelihood_decm(x,self.args),
                     }
        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.stop_fun = d_fun_stop[mod_met]
        except:    
            raise ValueError('Method must be "newton","quasi-newton", or "fixed-point".')
            
        #TODO: mancano metodi
        d_pmatrix = {
                    'cm': pmatrix_cm
                    }
        
        # Così basta aggiungere il decm e funziona tutto
        if model in ['cm']:
            self.args_p = (self.n_nodes, np.nonzero(self.dseq_out)[0], np.nonzero(self.dseq_in)[0])
            self.fun_pmatrix = lambda x: d_pmatrix[model](x,self.args_p)
    
    
    def _solve_problem_CReAMa(self, initial_guess=None, model='CReAMa', adjacency='dcm', method='quasinewton', max_steps=100, full_return=False, verbose=False):
        self.last_model = model
        if not isinstance(adjacency,(list,np.ndarray,str)):
            raise ValueError('adjacency must be a matrix or a method')
        elif isinstance(adjacency,str):
            self._solve_problem(initial_guess=initial_guess, model=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
            self.adjacency = self.fun_pmatrix(np.concatenate([self.x,self.y]))
        elif isinstance(adjacency,list):
            self.adjacency = np.array(adjacency)
        elif isinstance(adjacency,np.ndarray):
            self.adjacency = adjacency

        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError(r'adjacency matrix must be $n \times n$')

        self.full_return = full_return
        self.initial_guess = 'strengths'
        self._initialize_problem(model,method)
        x0 = self.x0 
            
        sol = solver(x0, fun=self.fun, fun_jac=self.fun_jac, g=self.stop_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return)
            
        self._set_solved_problem_CReAMa(sol)
    


    def solve_tool(self, model, method, initial_guess=None, adjacency=None, max_steps=100, full_return=False, verbose=False):
        """ function to switch around the various problems
        """
        #TODO: aggiungere tutti i metodi
        if model in ['cm', 'ecm']:
            self._solve_problem(initial_guess=initial_guess, model=model, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
        elif model in ['CReAMa']:
            self._solve_problem_CReAMa(initial_guess=initial_guess, model=model, adjacency=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)

