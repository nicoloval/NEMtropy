import numpy as np
import scipy.sparse
from numba import jit
import time


def out_degree(a):
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


def in_degree(a):
    """returns matrix A in degrees

    :param a: np.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 0).A1


def out_strength(a):
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


def in_strength(a):
    """returns matrix A out degrees

    :param a: numpy.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 0)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a , 0).A1


@jit(nopython=True)
def pmatrix_dcm(x,args):
    """
    Function evaluating the P-matrix of DCM given the solution of the underlying model
    """
    n = args[0]
    index_out = args[1]
    index_in = args[2]
    P = np.zeros((n,n),dtype=np.float64)
    xout = x[:n]
    yin = x[n:]
    for i in index_out:
        for j in index_in:
            aux = xout[i]*yin[j]
            P[i,j] = aux/(1+aux)
    return P


@jit(nopython=True)
def weighted_adjacency(x, adj, is_sparse):
    n = adj.shape[0]
    beta_out = x[:n]
    beta_in = x[n:]
    if self.is_sparse:
        weighted = scipy.sparse.csr_matrix((n,n),dtype=np.float64)
    else:
        weighted_adj = np.zeros_like(adj,dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(n):
            if adj[i,j]>0:
                weighted_adj[i,j] = adj[i,j]/(beta_out[i]+beta_in[j])
    return weighted_adj


@jit(nopython=True)
def iterative_CReAMa(beta,args):
    """Return the next iterative step for the CReAMa Model.

    :param numpy.ndarray v: old iteration step 
    :param numpy.ndarray par: constant parameters of the cm function
    :return: next iteration step 
    :rtype: numpy.ndarray
    """
    s_out = args[0]
    s_in = args[1]
    aux_adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    is_sparse = args[5]
    
    aux_n = len(s_out)
    
    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]
    
    xd = np.zeros(aux_n,dtype=np.float64)
    yd = np.zeros(aux_n,dtype=np.float64)
    
    if (aux_adj.size==2*aux_n) and (not is_sparse):
        x = aux_adj[:aux_n]
        y = aux_adj[aux_n:]

        for i in nz_index_out:
            for j in nz_index_in:
                if i!=j:
                    aux = x[i]*y[j]
                    aux_entry = aux/(1+aux)
                    if aux_entry>0:
                        aux = aux_entry/(1+beta_in[j]/beta_out[i])
                        xd[i] -= aux/s_out[i]

        for i in nz_index_in:
            for j in nz_index_out:
                if i!=j:
                    aux = x[j]*y[i]
                    aux_entry = aux/(1+aux)
                    if aux_entry>0:
                        aux = aux_entry/(1+beta_out[j]/beta_in[i])
                        yd[i] -= aux/s_in[i]
        
        return(np.concatenate((xd,yd)))

    else:
        adj = aux_adj
        for i in nz_index_out:
            for j in nz_index_in:
                if adj[i,j]>0:
                    aux = adj[i,j]/(1+beta_in[j]/beta_out[i])
                    xd[i] -= aux/s_out[i]

        for i in nz_index_in:
            for j in nz_index_out:
                if adj[j,i]>0:
                    aux = adj[j,i]/(1+beta_out[j]/beta_in[i])
                    yd[i] -= aux/s_in[i]
        
        return(np.concatenate((xd,yd)))


@jit(nopython=True)
def loglikelihood_CReAMa(beta,args):
    s_out = args[0]
    s_in = args[1]
    aux_adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    is_sparse = args[5]
    
    aux_n = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]
    
    f=0.0
    
    if (aux_adj.size==2*aux_n) and (not is_sparse):
        x = aux_adj[:aux_n]
        y = aux_adj[aux_n:]

        for i in nz_index_out:
            f -= s_out[i] * beta_out[i] 
            for j in nz_index_in:
                if i!=j:
                    aux = x[i]*y[j]
                    aux_entry = aux/(1+aux)
                    if (aux_entry>0):
                        f += aux_entry * np.log(beta_out[i] + beta_in[j])
        
        for i in nz_index_in:
            f -=  s_in[i] * beta_in[i]
        
        return f

    else:
        adj = aux_adj
        for i in nz_index_out:
            f -= s_out[i] * beta_out[i] 
            for j in nz_index_in:
                if (i!=j) and (adj[i,j]!=0):
                    f += adj[i,j] * np.log(beta_out[i] + beta_in[j])
        
        for i in nz_index_in:
            f -=  s_in[i] * beta_in[i]
        
        return f


@jit(nopython=True)
def loglikelihood_prime_CReAMa(beta, args):
    s_out = args[0]
    s_in = args[1]
    aux_adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    is_sparse = args[5]
    
    aux_n = len(s_out)

    beta_out = beta[0:aux_n]
    beta_in = beta[aux_n:2*aux_n]

    aux_F_out = np.zeros_like(beta_out,dtype=np.float64)
    aux_F_in = np.zeros_like(beta_in,dtype=np.float64)

    if (aux_adj.size==2*aux_n) and (not is_sparse):
        x = aux_adj[:aux_n]
        y = aux_adj[aux_n:]
        
        for i in nz_index_out:
            aux_F_out[i] -= s_out[i]
            for j in nz_index_in:
                if (i!=j):
                    aux = x[i]*y[j]
                    aux_entry = aux/(1+aux)
                    if (aux_entry > 0):
                        aux_F_out[i] += aux_entry/(beta_out[i]+beta_in[j])

        for i in nz_index_in:
            aux_F_in[i] -= s_in[i]
            for j in nz_index_out:
                if (i!=j):
                    aux = x[j]*y[i]
                    aux_entry = aux/(1+aux)
                    if (aux_entry > 0):
                        aux_F_in[i] += aux_entry/(beta_out[j]+beta_in[i])

        return (np.concatenate((aux_F_out,aux_F_in)))

    else:
        adj = aux_adj
        for i in nz_index_out:
            aux_F_out[i] -= s_out[i]
            for j in nz_index_in:
                if (adj[i, j] > 0) and (i!=j):
                    aux_F_out[i] += adj[i, j]/(beta_out[i]+beta_in[j])

        for i in nz_index_in:
            aux_F_in[i] -= s_in[i]
            for j in nz_index_out:
                if (adj[j, i] > 0) and (i!=j):
                    aux_F_in[i] += adj[j, i]/(beta_out[j]+beta_in[i])

        return (np.concatenate((aux_F_out,aux_F_in)))


@jit(nopython=True)
def loglikelihood_hessian_CReAMa(beta, args):
    s_out = args[0]
    s_in = args[1]
    aux_adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    is_sparse = args[5]

    aux_n  = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(shape=(2*aux_n, 2*aux_n),dtype=np.float64)

    if (aux_adj.size==2*aux_n) and (not is_sparse):
        x = aux_adj[:aux_n]
        y = aux_adj[aux_n:]
    
        for i in nz_index_out:
            for j in nz_index_in:
                if (i!=j):
                    aux = x[i]*y[j]
                    aux_entry = aux/(1+aux)
                    if (aux_entry>0):
                        aux = aux_entry/((beta_out[i]+beta_in[j])**2)
                        f[i,i] += -aux
                        f[i,j+aux_n] = - aux
                        f[j+aux_n,i] = -aux
        for i in nz_index_in:
            for j in nz_index_out:
                if (i!=j):
                    aux = x[j]*y[i]
                    aux_entry = aux/(1+aux)
                    if (aux_entry>0):
                        aux = aux_entry/((beta_out[j]+beta_in[i])**2)
                        f[i+aux_n,i+aux_n] += -aux

            return f
    else:
        adj = aux_adj
        for i in nz_index_out:
            for j in nz_index_in:
                if (adj[i,j]>0) and (i!=j):
                    aux = adj[i,j]/((beta_out[i]+beta_in[j])**2)
                    f[i,i] += -aux
                    f[i,j+aux_n] = - aux
                    f[j+aux_n,i] = -aux
        for i in nz_index_in:
            for j in nz_index_out:
                if (adj[j,i]>0) and (i!=j):
                    aux = adj[j,i]/((beta_out[j]+beta_in[i])**2)
                    f[i+aux_n,i+aux_n] += -aux

            return f


@jit(nopython=True)
def loglikelihood_hessian_diag_CReAMa(beta, args):
    s_out = args[0]
    s_in = args[1]
    aux_adj = args[2]
    nz_index_out = args[3]
    nz_index_in = args[4]
    is_sparse = args[5]

    aux_n  = len(s_out)

    beta_out = beta[:aux_n]
    beta_in = beta[aux_n:]

    f = np.zeros(2*aux_n,dtype=np.float64)

    if (aux_adj.size==2*aux_n) and (not is_sparse):
        x = aux_adj[:aux_n]
        y = aux_adj[aux_n:]

        for i in nz_index_out:
            for j in  nz_index_in:
                if (i!=j):
                    aux = x[i]*y[j]
                    aux_entry = aux/(1+aux)
                    if (aux_entry>0):
                        f[i] -= aux_entry / \
                                 ((beta_out[i]+beta_in[j])**2)

        for i in nz_index_in:
            for j in nz_index_out:
                if (i!=j):
                    aux = x[j]*y[i]
                    aux_entry = aux/(1+aux)
                    if (aux_entry>0):
                        f[i+aux_n] -= aux_entry / \
                                       ((beta_out[j]+beta_in[i])**2)

        return f
    else:
        adj = aux_adj

        for i in nz_index_out:
            for j in  nz_index_in:
                if (adj[i,j]>0) and (i!=j):
                    f[i] -= adj[i, j] / \
                             ((beta_out[i]+beta_in[j])**2)

        for i in nz_index_in:
            for j in nz_index_out:
                if adj[j,i]>0:
                    f[i+aux_n] -= adj[j,i] / \
                                   ((beta_out[j]+beta_in[i])**2)

        return f


@jit(forceobj=True)
def random_binary_matrix_generator_dense(n, sym=False, seed=None):
    if sym == False:
        np.random.seed(seed = seed)
        A = np.random.randint(0, 2, size=(n, n))
        # zeros on the diagonal
        for i in range(n):
            A[i, i] = 0
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]),k_in,k_out):
            if (ki==0)and(ko==0):
                while((np.sum(A[:,ind])==0) and (np.sum(A[ind,:])==0)):
                    if np.random.random() >0.5:
                        A[np.random.randint(A.shape[0]), ind] = 1
                    else:
                        A[ind,np.random.randint(A.shape[0])] = 1
                            
                    A[ind,ind] = 0
        return A


@jit(forceobj=True)
def random_weighted_matrix_generator_dense(n, sup_ext = 10, sym=False, seed=None, dtype = np.float64):
    if sym==False:
        np.random.seed(seed = seed)
        A = np.random.random(size=(n, n)) * sup_ext
        np.fill_diagonal(A,0)
        
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]),k_in,k_out):
            if (ki==0) and (ko==0):
                while((np.sum(A[:,ind])==0) and (np.sum(A[ind,:])==0)):
                    if np.random.random() >0.5:
                        A[np.random.randint(A.shape[0]), ind] = np.random.random() * sup_ext
                    else:
                        A[ind,np.random.randint(A.shape[0])] = np.random.random() * sup_ext
                            
                    A[ind,ind] = 0
        return A.astype(dtype)
    else:
        np.random.seed(seed = seed)
        b = np.random.random(size=(n, n)) * sup_ext
        A = (b + b.T)/2
        np.fill_diagonal(A,0)
        
        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k==0:
                while(np.sum(A[:,ind])==0):
                    indices = np.random.randint(A.shape[0])
                    if indices!= ind:
                        A[0, indices] = np.random.random() * sup_ext
                        A[indices, 0] = A[0, indices]
        return A.astype(dtype)


jit(forceobj=True)
def random_binary_matrix_generator_custom_density(n, p=0.1 , sym=False, seed=None):
    if sym==False:
        np.random.seed(seed = seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if np.random.random()<=p:
                    A[i,j] = 1
        np.fill_diagonal(A,0)
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]),k_in,k_out):
            if (ki==0) and (ko==0):
                while((np.sum(A[:,ind])==0) and (np.sum(A[ind,:])==0)):
                    if np.random.random() >0.5:
                        A[np.random.randint(A.shape[0]), ind] = 1
                    else:
                        A[ind,np.random.randint(A.shape[0])] = 1
                            
                    A[ind,ind] = 0
        return A
    else:
        np.random.seed(seed = seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i+1,n):
                if np.random.random()<=p:
                    A[i,j] = 1
                    A[j,i] = A[i,j]
        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k==0:
                while(np.sum(A[:,ind])==0):
                    indices = np.random.randint(A.shape[0])
                    if indices!= ind:
                        A[ind, indices] = 1
                        A[indices, ind] = A[ind, indices]
                    A[ind,ind] = 0
        return A
  

jit(forceobj=True)
def random_weighted_matrix_generator_custom_density(n, p=0.1 ,sup_ext = 10, sym=False, seed=None, dtype = np.float64):
    if sym==False:
        np.random.seed(seed = seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if np.random.random()<=p:
                    A[i,j] = np.random.random()*sup_ext
        np.fill_diagonal(A,0)
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]),k_in,k_out):
            if (ki==0) and (ko==0):
                while((np.sum(A[:,ind])==0) and (np.sum(A[ind,:])==0)):
                    if np.random.random() >0.5:
                        A[np.random.randint(A.shape[0]), ind] = np.random.random() * sup_ext
                    else:
                        A[ind,np.random.randint(A.shape[0])] = np.random.random() * sup_ext
                            
                    A[ind,ind] = 0
        return A.astype(dtype)
    else:
        np.random.seed(seed = seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i+1,n):
                if np.random.random()<=p:
                    A[i,j] = np.random.random()*sup_ext
                    A[j,i] = A[i,j]
        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k==0:
                while(np.sum(A[:,ind])==0):
                    indices = np.random.randint(A.shape[0])
                    if indices!= ind:
                        A[0, indices] = np.random.random() * sup_ext
                        A[indices, 0] = A[0, indices]
        return A.astype(dtype)


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
def loglikelihood_hessian_dcm(x, args):
    """
    :param x: np.array
    :param args: list
    :return: np.array

    x = [a_in, a_out] where a^{in}_i = e^{-\theta^{in}_i} for rd class i
    par = [k_in, k_out, c]
        c is the cardinality of each class

    log-likelihood hessian: Directed Configuration Model reduced.

    """
    k_out = args[0]
    k_in = args[1]
    nz_out_index = args[2]
    nz_in_index = args[3]
    c = args[4]
    n = len(k_out)

    out = np.zeros((2*n, 2*n))  # hessian matrix

    for h in nz_out_index:
        out[h, h] = -c[h]*k_out[h]/(x[h])**2
        # out[h+n, h+n] = -c[h]*k_in[h]/(x[h+n])**2
        for i in nz_in_index:
            if i == h:
                # const = c[h]*(c[h] - 1)
                const = (c[h] - 1)
            else:
                # const = c[h]*c[i]
                const = c[i]

            out[h, h] += const*(x[i+n]/(1 + x[h]*x[i+n]))**2
            out[h, i+n] = -const/(1 + x[i+n]*x[h])**2

    for i in nz_in_index:
        out[i+n, i+n] = -c[i]*k_in[i]/(x[i+n])**2
        for h in nz_out_index:
            if i == h:
                # const = c[h]*(c[h] - 1)
                const = (c[h] - 1)
            else:
                # const = c[h]*c[i]
                const = c[i]

            out[i+n, i+n] += const*(x[h]**2)/(1 + x[i+n]*x[h])**2
            out[i+n, h] = -const/(1 + x[i+n]*x[h])**2
    
    return out


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
            
            tmp = (1 + x[i]*x[j+n])
            fx += const*x[j+n]*x[j+n]/(tmp*tmp)
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
            
            tmp = (1 + x[j+n]*x[i])*(1 + x[j+n]*x[i])
            fy += const*x[i]*x[i]/(tmp)
        # original prime
        f[j+n] = fy - k_in[j]/(x[j+n]*x[j+n])

    # f[f == 0] = 1

    return f


@jit(nopython=True)
def loglikelihood_decm(x, args):
    """not reduced
    """
    # problem fixed parameters
    k_out = args[0] 
    k_in = args[1] 
    s_out = args[2] 
    s_in = args[3] 
    n = len(k_out) 

    f = 0
    for i in range(n):
        f += k_out[i]*np.log(x[i]) \
            + k_in[i]*np.log(x[i+n]) \
            + s_out[i]*np.log(x[i+2*n]) \
            + s_in[i]*np.log(x[i+3*n])
        for j in range(n):
            if i != j:
                tmp = x[i+2*n]*x[j+3*n]
                f += np.log(1 - tmp)
                f -= np.log(1 - tmp \
                     + tmp*x[i]*x[j+n])
    return f


@jit(nopython=True)
def loglikelihood_prime_decm(x, args):
    """not reduced
    """
    # problem fixed parameters
    k_out = args[0] 
    k_in = args[1] 
    s_out = args[2] 
    s_in = args[3] 
    n = len(k_out) 

    f = np.zeros(4*n) 
    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp = x[i+2*n]*x[j+3*n]
                fa_out += x[j+n]*tmp\
                          /(1 - tmp\
                          + x[i]*x[j+n]*tmp)
                tmp = x[j+2*n]*x[i+3*n] 
                fa_in += x[j]*tmp\
                         /(1 - tmp\
                         + x[j]*x[i+n]*tmp)
                tmp = x[j+3*n]*x[i+2*n]
                fb_out += x[j +3*n]/(1 - tmp)\
                          + (x[j+n]*x[i] - 1)*x[j+3*n]\
                          /(1 - tmp\
                          + x[i]*x[j+n]*tmp)
                tmp = x[i+3*n]*x[j+2*n]
                fb_in += x[j +2*n]/(1 - tmp)\
                         + (x[i+n]*x[j] - 1)*x[j+2*n]\
                         /(1 - tmp\
                         + x[j]*x[i+n]*tmp)

        f[i] = k_out[i]/x[i] - fa_out
        f[i+n] = k_in[i]/x[i+n] - fa_in
        f[i+2*n] = s_out[i]/x[i+2*n] - fb_out
        f[i+3*n] = s_in[i]/x[i+3*n] - fb_in

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_decm(x, args):
    """not reduced
    """
    # problem fixed parameters
    k_out = args[0] 
    k_in = args[1] 
    s_out = args[2] 
    s_in = args[3] 
    n = len(k_out) 

    f = np.zeros(4*n) 
    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                tmp0 = x[i+2*n]*x[j+3*n]
                tmp = (x[j+n]*tmp0)/(1 - tmp0 + x[i]*x[j+n]*tmp0)
                fa_out += tmp*tmp

                tmp0 = x[j+2*n]*x[i+3*n]
                tmp = (x[j]*tmp0)/(1 - tmp0 + x[j]*x[i+n]*tmp0)
                fa_in += tmp*tmp 

                tmp0 = x[j+3*n]*x[i+2*n]
                tmp1 = x[j+3*n]/(1 - tmp0)
                tmp2 = ((x[j+n]*x[i] - 1)*x[j+3*n])/(1 - tmp0 + x[i]*x[j+n]*tmp0)
                fb_out += tmp1*tmp1 - tmp2*tmp2

                tmp0 = x[i+3*n]*x[j+2*n]
                tmp1 = x[j +2*n]**2/(1 - tmp0)
                tmp2 = ((x[i+n]*x[j] - 1)*x[j+2*n])/(1 - tmp0 + x[j]*x[i+n]*tmp0)
                fb_in += tmp1*tmp1 - tmp2*tmp2 

        f[i] = -k_out[i]/x[i]**2 + fa_out
        f[i+n] = -k_in[i]/x[i+n]**2 + fa_in
        f[i+2*n] = -s_out[i]/x[i+2*n]**2 - fb_out
        f[i+3*n] = -s_in[i]/x[i+3*n]**2 - fb_in

    return f


@jit(nopython=True)
def loglikelihood_hessian_decm(x, args):
    # MINUS log-likelihood function Hessian : DECM case
    # x : where the function is computed : np.array
    # par : real data constant parameters : np.array
    # rem: size(x)=size(par)

    # REMARK:
    # the ll fun is to max but the algorithm minimize,
    # so the fun return -H

    k_out = args[0]
    k_in = args[1] 
    s_out = args[2] 
    s_in = args[3] 

    n = len(k_out)
    f = np.zeros((n * 4, n * 4))

    a_out = x[:n]
    a_in = x[n:2*n]
    b_out = x[2*n:3*n]
    b_in = x[3*n:]

    for h in range(n):
        for l in range(n):
            if h == l:
                # dll / da^in da^in
                f[h+n, l+n] = -k_in[h] / a_in[h] ** 2
                # dll / da^out da^out
                f[h, l] = -k_out[h] / a_out[h] ** 2
                # dll / db^in db^in
                f[h+3*n, l+3*n] = -s_in[h] / b_in[h] ** 2
                # dll / db^out db^out
                f[h+2*n, l+2*n] = -s_out[h] / b_out[h] ** 2

                for j in range(n):
                    if j != h:
                        # dll / da^in da^in
                        f[h+n, l+n] = (f[h+n, l+n]
                                   + (a_out[j] * b_in[h] * b_out[j]
                                      / (1 - b_in[h] * b_out[j]
                                         + a_in[h] * a_out[j]
                                         * b_in[h] * b_out[j])) ** 2)
                        # dll / da^in db^in
                        f[h+n, l+3*n] = (f[h+n, l+3*n]
                                       - a_out[j]*b_out[j]
                                       / (1-b_in[h]*b_out[j]
                                          + a_in[h]*a_out[j]*b_in[h]*b_out[j])**2)
                        # dll / da^out da^out
                        f[h, l] = (f[h, l]
                                           + (a_in[j] * b_in[j] * b_out[h]) ** 2
                                           / (1 - b_in[j] * b_out[h] +
                                              a_in[j] * a_out[h] *
                                              b_in[j] * b_out[h]) ** 2)
                        # dll / da^out db^out
                        f[h, l+2*n] = (f[h, l+2*n]
                                               - a_in[j] * b_in[j]
                                               / (1 - b_in[j] * b_out[h]
                                                  + a_in[j] * a_out[h]
                                                  * b_in[j] * b_out[h])**2)
                        # dll / db^in da^in
                        f[h+3*n,l+n] = (f[h+3*n, l+n] - a_out[j] * b_out[j]
                                           / (1 - b_in[h] * b_out[j] + a_in[h]
                                           * a_out[j] * b_in[h] * b_out[j]) ** 2)
                        # dll / db_in db_in
                        f[h+3*n, l+3*n] = (f[h+3*n, l+3*n]
                                           - (b_out[j]/(1-b_in[h]*b_out[j]))**2
                                           + (b_out[j]*(a_in[h]*a_out[j]-1)
                                              / (1-b_in[h]*b_out[j]
                                                 + a_in[h]*a_out[j]
                                                 * b_in[h]*b_out[j]))**2)
                        # dll / db_out da_out
                        f[h+2*n, l] = (f[h+2*n, l]
                                               - a_in[j] * b_in[j]
                                               / (1 - b_in[j] * b_out[h]
                                                  + a_in[j] * a_out[h]
                                                  * b_in[j] * b_out[h]) ** 2)
                        # dll / db^out db^out
                        f[h+2*n, l+2*n] = (f[h+2*n, l+2*n]
                                           - (b_in[j]/(1-b_in[j]*b_out[h]))**2
                                           + ((a_in[j]*a_out[h]-1)*b_in[j]
                                              / (1-b_in[j]*b_out[h]
                                                 + a_in[j]*a_out[h]
                                                 * b_in[j]*b_out[h]))**2)

            else:
                # dll / da_in da_out
                f[h+n, l] = (- b_in[h] * b_out[l] * (1 - b_in[h] * b_out[l])
                               / (1 - b_in[h] * b_out[l]
                                  + a_in[h] * a_out[l]
                                  * b_in[h] * b_out[l]) ** 2)
                # dll / da_in db_out
                f[h+n, l+2*n] = (- a_out[l] * b_in[h]
                                   / (1 - b_in[h] * b_out[l]
                                      + a_in[h] * a_out[l]
                                      * b_in[h] * b_out[l]) ** 2)
                # dll / da_out da_in
                f[h, l+n] = (- b_in[l] * b_out[h]*(1 - b_in[l] * b_out[h])
                               / (1 - b_in[l] * b_out[h]
                                  + a_in[l] * a_out[h]
                                  * b_in[l] * b_out[h]) ** 2)
                # dll / da_out db_in
                f[h, l+3*n] = (-a_in[l] * b_out[h]
                                       / (1 - b_in[l] * b_out[h]
                                       + a_in[l] * a_out[h]
                                       * b_in[l] * b_out[h]) ** 2)
                # dll / db_in da_out
                f[h+3*n, l] = (- a_in[h] * b_out[l]
                                       / (1 - b_in[h] * b_out[l]
                                          + a_in[h] * a_out[l]
                                          * b_in[h] * b_out[l]) ** 2)
                # dll / db_in db_out
                f[h+3*n, l+2*n] = (-1 / (1 - b_in[h] * b_out[l])**2
                                           - (a_out[l] * a_in[h] - 1)
                                           / (1 - b_in[h] * b_out[l]
                                              + a_in[h] * a_out[l]
                                              * b_in[h] * b_out[l]) ** 2)
                # dll / db_out da_in
                f[h+2*n, l+n] = (- a_out[h] * b_in[l]
                                   / (1 - b_in[l] * b_out[h]
                                      + a_in[l] * a_out[h]
                                      * b_in[l] * b_out[h]) ** 2)
                # dll / db_out db_in
                f[h+2*n, l+3*n] = (-1 / (1 - b_in[l] * b_out[h]) ** 2
                                           - (a_in[l] * a_out[h] - 1)
                                           / (1 - b_in[l] * b_out[h]
                                              + a_in[l] * a_out[h]
                                              * b_in[l] * b_out[h]) ** 2)

    return f


@jit(nopython=True)
def iterative_decm(x, args):
    """iterative function for decm
    """
    # problem fixed parameters
    k_out = args[0] 
    k_in = args[1] 
    s_out = args[2] 
    s_in = args[3] 
    n = len(k_out) 

    f = np.zeros(4*n) 

    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        b = 0
        for j in range(n):
            if i != j:
                tmp = x[i+2*n]*x[j+3*n]
                fa_out += x[j+n]*tmp/(1 - tmp + x[i]*x[j+n]*tmp)

                tmp = x[j+2*n]*x[i+3*n]
                fa_in += x[j]*tmp/(1 - tmp + x[j]*x[i+n]*tmp)

                tmp = x[j+3*n]*x[i+2*n]
                tmp0 = x[j+n]*x[i]
                fb_out += x[j+3*n]/(1 - tmp) + ( tmp0- 1)*x[j+3*n]\
                          /(1 - tmp + tmp0*tmp)
                tmp = x[i+3*n]*x[j+2*n]
                tmp0 = x[i+n]*x[j]
                fb_in += x[j+2*n]/(1 - tmp)+ (tmp0 - 1)*x[j+2*n]\
                         /(1 - tmp + tmp0*tmp)
            
        """
        if k_out[i] != 0:
            f[i] = x[i] - k_out[i]/fa_out
        else:
            f[i] = x[i] 
        if k_in[i] != 0:
            f[i+n] = x[i+n] - k_in[i]/fa_in
        else:
            f[i+n] = x[i+n]
        if s_out[i] != 0:
            f[i+2*n] = x[i+2*n] - s_out[i]/fb_out
        else:
            f[i+2*n] = 0
        if s_in[i] != 0:
            f[i+3*n] = x[i+3*n] - s_in[i]/fb_in
        else:
            f[i+3*n] = x[i+3*n]
        """
        if k_out[i] != 0:
            f[i] = k_out[i]/fa_out
        else:
            f[i] = 0 
        if k_in[i] != 0:
            f[i+n] = k_in[i]/fa_in
        else:
            f[i+n] = 0 
        if s_out[i] != 0:
            f[i+2*n] = s_out[i]/fb_out
        else:
            f[i+2*n] = 0
        if s_in[i] != 0:
            f[i+3*n] = s_in[i]/fb_in
        else:
            f[i+3*n] = 0 
 
 
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


@jit(nopython=True)
def expected_decm(x):
    """
    """
    # casadi MX function calculation
    n = int(len(x)/4)
    f = np.zeros(len(x))

    for i in range(n):
        fa_out = 0
        fa_in = 0
        fb_out = 0
        fb_in = 0
        for j in range(n):
            if i != j:
                fa_out += x[j+n]*x[i+2*n]*x[j+3*n]/(1 - x[i+2*n]*x[j+3*n] + x[i]*x[j+n]*x[i+2*n]*x[j+3*n])
                fa_in += x[j]*x[j+2*n]*x[i+3*n]/(1 - x[j+2*n]*x[i+3*n] + x[j]*x[i+n]*x[j+2*n]*x[i+3*n])
                fb_out += x[j +3*n]/(1 - x[j+3*n]*x[i+2*n]) + (x[j+n]*x[i] - 1)*x[j+3*n]/(1 - x[i+2*n]*x[j+3*n] + x[i]*x[j+n]*x[i+2*n]*x[j+3*n])
                fb_in += x[j +2*n]/(1 - x[i+3*n]*x[j+2*n]) + (x[i+n]*x[j] - 1)*x[j+2*n]/(1 - x[j+2*n]*x[i+3*n] + x[j]*x[i+n]*x[j+2*n]*x[i+3*n])
        f[i] = x[i]*fa_out
        f[i+n] = x[i+n]*fa_in
        f[i+2*n] = x[i+2*n]*fb_out
        f[i+3*n] = x[i+3*n]*fb_in

    return f


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


@jit(forceobj=True)
def expected_out_strength_CReAMa(sol,adj):
    n = int(sol.size/2)
    b_out = sol[:n]
    b_in = sol[n:]
    s = np.zeros(n)
    if (adj.size==2*n) and (not scipy.sparse.isspmatrix(adj)):
        x = adj[:n]
        y = adj[n:]
        for i in range(n):
            for j in range(n):
                if i!=j:
                    aux = x[i] * y[j]
                    aux_entry = aux/(1+aux)
                    if aux_entry>0:
                        s[i] += aux_entry/(b_out[i]+b_in[j])
        return s

    else:
        for i in range(n):
            for j in range(n):
                if (adj[i,j]>0)and(i!=j):
                    s[i] += adj[i,j]/(b_out[i]+b_in[j])
        return s


@jit(forceobj=True)
def expected_in_stregth_CReAMa(sol,adj):  
    n = int(sol.size/2)
    b_out = sol[:n]
    b_in = sol[n:]
    s = np.zeros(n)
    if (adj.size==2*n) and (not scipy.sparse.isspmatrix(adj)):
        x = adj[:n]
        y = adj[n:]
        for i in range(n):
            for j in range(n):
                if i!=j:
                    aux = x[j] * y[i]
                    aux_entry = aux/(1+aux)
                    if aux_entry>0:
                        s[i] += aux_entry/(b_out[j]+b_in[i])
        return s

    else:
        for i in range(n):
            for j in range(n):
                if (adj[j,i]>0)and(i!=j):
                    s[i] += adj[j,i]/(b_out[j]+b_in[i])
        return s





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
            #TODO: fun(x) non e' il graident di step_funx
            #TODO: check dianati fornisce una direzione di discesa 

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
    
    #print ('f_old',f_old)
    #print ('c1',c1)
    #print('alpha',alpha)
    #print ('grad_f',grad_f)
    #print('p.T',p.T)

    sup = f_old + c1 *alpha*grad_f@p.T
    # print(alpha, f_new, sup)
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
        self.x = None
        self.y = None
        self.xy = None
        self.b_out = None
        self.b_in = None

        # reduced solutions
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
        self.x0 = None
        self.error = None
        self.error_strength = None
        self.relative_error_strength = None
        self.full_return = False
        self.last_model = None
        self.is_sparse = False

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
                    self.dseq_in = in_degree(adjacency)
                    self.dseq_out = out_degree(adjacency)
                else:
                    self.dseq_in = in_degree(adjacency)
                    self.dseq_out = out_degree(adjacency)
                    self.in_strength = in_strength(adjacency)
                    self.out_strength = out_strength(adjacency)
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


    def _solve_problem(self, initial_guess=None, model='dcm', method='quasinewton', max_steps=100, full_return=False, verbose=False, linsearch=True):
        
        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self._initialize_problem(model, method)
        x0 = self.x0 

        sol =  solver(x0, fun=self.fun, fun_jac=self.fun_jac, step_fun=self.step_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return, linsearch=linsearch)

        self._set_solved_problem(sol)


    def _set_solved_problem_dcm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
        else:
            self.r_xy = solution 
            
        self.r_x = self.r_xy[:self.rnz_n_out]
        self.r_y = self.r_xy[self.rnz_n_out:]

        self.x = self.r_x[self.r_invert_dseq]
        self.y = self.r_y[self.r_invert_dseq]
        

    def _set_solved_problem_decm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
        else:
            self.r_xy = solution 

        self.x = self.r_xy[:self.n_nodes]
        self.y = self.r_xy[self.n_nodes: 2*self.n_nodes]
        self.b_out = self.r_xy[2*self.n_nodes:3*self.n_nodes]
        self.b_in = self.r_xy[3*self.n_nodes:]
 

    def _set_solved_problem(self, solution):
        model = self.last_model
        if model == 'dcm':
            self._set_solved_problem_dcm(solution)
        elif model == 'decm':
            self._set_solved_problem_decm(solution)
        elif model == 'CReAMa':
            self._set_solved_problem_CReAMa(solution)

        
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


    def _set_initial_guess(self, model):

        if model == 'dcm':
            self._set_initial_guess_dcm()
        elif model == 'decm':
            self._set_initial_guess_decm()
        elif model == 'CReAMa':
            self._set_initial_guess_CReAMa()


    def _set_initial_guess_dcm(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.

        if ~self.is_reduced:
            self.degree_reduction()

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

        self.x0 = np.concatenate((self.r_x, self.r_y))


    def _set_initial_guess_CReAMa(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        #TODO: mettere un self.is_weighted bool 
        if self.initial_guess is None:
            self.b_out = (self.out_strength>0).astype(float) / self.out_strength.sum()  # This +1 increases the stability of the solutions.
            self.b_in = (self.in_strength>0).astype(float) / self.in_strength.sum()
        elif self.initial_guess == 'strengths':
            self.b_out = (self.out_strength>0).astype(float) / (self.out_strength + 1)
            self.b_in = (self.in_strength>0).astype(float) / (self.in_strength + 1)

        self.x0 = np.concatenate((self.b_out, self.b_in))


    def _set_initial_guess_decm(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if self.initial_guess is None:
            self.x = self.dseq_out.astype(float) / (self.n_edges + 1)
            self.y = self.dseq_in_in.astype(float) / (self.n_edges + 1)
            self.b_out = self.out_strength.astype(float) / self.out_strength.sum()  # This +1 increases the stability of the solutions.
            self.b_in = self.in_strength.astype(float) / self.in_strength.sum()
        elif self.initial_guess == 'strengths':
            self.x = self.dseq_out.astype(float) / (self.dseq_out + 1)
            self.y = self.dseq_in.astype(float) / (self.dseq_in + 1)
            self.b_out = self.out_strength.astype(float) / (self.out_strength + 1)
            self.b_in = self.in_strength.astype(float) / (self.in_strength + 1)
        elif self.initial_guess == 'random':
            self.x = np.random.rand(self.n_nodes).astype(np.float64)
            self.y = np.random.rand(self.n_nodes).astype(np.float64)
            self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
            self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.x = 0.9*np.ones(self.n_nodes, dtype=np.float64)  # All probabilities will be 1/2 initially
            self.y = 0.9*np.ones(self.n_nodes, dtype=np.float64)
            self.b_out = 0.9*np.ones(self.n_nodes, dtype=np.float64) 
            self.b_in = 0.9*np.ones(self.n_nodes, dtype=np.float64)
 
        
        self.x[self.rnz_dseq_out == 0] = 0
        self.y[self.rnz_dseq_in == 0] = 0

        self.x0 = np.concatenate((self.x, self.y, self.b_out, self.b_in))


    def solution_error(self):
        if self.last_model in ['dcm','CReAMa']:
            if (self.x is not None) and (self.y is not None):
                sol = np.concatenate((self.x, self.y))
                ex_k_out = expected_out_degree_dcm(sol)
                ex_k_in = expected_in_degree_dcm(sol)
                ex_k = np.concatenate((ex_k_out, ex_k_in))
                k = np.concatenate((self.dseq_out, self.dseq_in))
                # print(k, ex_k)
                self.expected_dseq = ex_k
                self.error = np.linalg.norm(ex_k - k)
            if (self.b_out is not None) and (self.b_in is not None):
                sol = np.concatenate([self.b_out,self.b_in])
                ex_s_out = expected_out_strength_CReAMa(sol,self.adjacency_CReAMa)
                ex_s_in = expected_in_stregth_CReAMa(sol,self.adjacency_CReAMa)
                ex_s = np.concatenate([ex_s_out,ex_s_in])
                s = np.concatenate([self.out_strength,self.in_strength])
                self.expected_stregth_seq = ex_s
                self.error_strength = np.linalg.norm(ex_s - s)
                self.relative_error_strength = self.error_strength/self.out_strength.sum()
        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ['decm']:
                sol = np.concatenate((self.x, self.y, self.b_out, self.b_in))
                ex = expected_decm(sol)
                k = np.concatenate((self.dseq_out, self.dseq_in, self.out_strength, self.in_strength))
                self.expected_dseq = ex[:2*self.n_nodes]
                self.expected_stregth_seq = ex[2*self.n_nodes:]
                self.error = np.linalg.norm(ex - k)
                self.relative_error_strength = self.error/self.out_strength.sum()
    

    def _set_args(self, model):

        if model=='CReAMa':
            self.args = (self.out_strength, self.in_strength, self.adjacency_CReAMa, self.nz_index_sout, self.nz_index_sin, scipy.sparse.isspmatrix(self.adjacency_CReAMa))
        elif model == 'dcm':
            self.args = (self.rnz_dseq_out, self.rnz_dseq_in, self.nz_index_out, self.nz_index_in, self.r_multiplicity)
        elif model == 'decm':
            self.args = (self.dseq_out, self.dseq_in, self.out_strength, self.in_strength) 


    def _initialize_problem(self, model, method):
        
        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = '-'
        mod_met = mod_met.join([model,method])

        d_fun = {
                'dcm-newton': lambda x: -loglikelihood_prime_dcm(x,self.args),
                'dcm-quasinewton': lambda x: -loglikelihood_prime_dcm(x,self.args),
                'dcm-fixed-point': lambda x: iterative_dcm(x,self.args),


                'CReAMa-newton': lambda x: -loglikelihood_prime_CReAMa(x,self.args),
                'CReAMa-quasinewton': lambda x: -loglikelihood_prime_CReAMa(x,self.args),
                'CReAMa-fixed-point': lambda x: -iterative_CReAMa(x,self.args),

                'decm-newton': lambda x: -loglikelihood_prime_decm(x,self.args),
                'decm-quasinewton': lambda x: -loglikelihood_prime_decm(x,self.args),
                'decm-fixed-point': lambda x: iterative_decm(x,self.args),
                }

        d_fun_jac = {
                    'dcm-newton': lambda x: -loglikelihood_hessian_dcm(x,self.args),
                    'dcm-quasinewton': lambda x: -loglikelihood_hessian_diag_dcm(x,self.args),
                    'dcm-fixed-point': None,

                    'CReAMa-newton': lambda x: -loglikelihood_hessian_CReAMa(x,self.args),
                    'CReAMa-quasinewton': lambda x: -loglikelihood_hessian_diag_CReAMa(x,self.args),
                    'CReAMa-fixed-point': None,

                    'decm-newton': lambda x: -loglikelihood_hessian_decm(x,self.args),
                    'decm-quasinewton': lambda x: -loglikelihood_hessian_diag_decm(x,self.args),
                    'decm-fixed-point': None,
                    }
        d_fun_step = {
                     'dcm-newton': lambda x: -loglikelihood_dcm(x,self.args),
                     'dcm-quasinewton': lambda x: -loglikelihood_dcm(x,self.args),
                     'dcm-fixed-point': lambda x: -loglikelihood_dcm(x,self.args),

                     'CReAMa-newton': lambda x: -loglikelihood_CReAMa(x,self.args),
                     'CReAMa-quasinewton': lambda x: -loglikelihood_CReAMa(x,self.args),
                     'CReAMa-fixed-point': lambda x: -loglikelihood_CReAMa(x,self.args),

                     'decm-newton': lambda x: -loglikelihood_decm(x,self.args),
                     'decm-quasinewton': lambda x: -loglikelihood_decm(x,self.args),
                     'decm-fixed-point': lambda x: -loglikelihood_decm(x,self.args),
                     }
        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.step_fun = d_fun_step[mod_met]
        except:    
            raise ValueError('Method must be "newton","quasi-newton", or "fixed-point".')
            
        #TODO: mancano metodi
        d_pmatrix = {
                    'dcm': pmatrix_dcm
                    }
        
        # Così basta aggiungere il decm e funziona tutto
        if model in ['dcm']:
            self.args_p = (self.n_nodes, np.nonzero(self.dseq_out)[0], np.nonzero(self.dseq_in)[0])
            self.fun_pmatrix = lambda x: d_pmatrix[model](x,self.args_p)
    
    
    def _solve_problem_CReAMa(self, initial_guess=None, model='CReAMa', adjacency='dcm', method='quasinewton', max_steps=100, full_return=False, verbose=False, linsearch=True):
        if not isinstance(adjacency,(list,np.ndarray,str)):
            raise ValueError('adjacency must be a matrix or a method')
        elif isinstance(adjacency,str):
            self._solve_problem(initial_guess=initial_guess, model=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
            if self.is_sparse:
                self.adjacency_CReAMa = np.concatenate([self.x,self.y])
            else:
                self.adjacency_CReAMa = self.fun_pmatrix(np.concatenate([self.x,self.y]))
        elif isinstance(adjacency,list):
            self.adjacency_CReAMa = np.array(adjacency).astype(float)
        elif isinstance(adjacency,np.ndarray):
            self.adjacency_CReAMa = adjacency.astype(float)

        #if self.adjacency_CReAMa.shape[0] != self.adjacadjacency_CReAMaency.shape[1]:
        #    raise ValueError(r'adjacency matrix must be $n \times n$')

        self.last_model = model
        self.full_return = full_return
        self.initial_guess = 'strengths'
        self._initialize_problem(model,method)
        x0 = self.x0 

        sol = solver(x0, fun=self.fun, fun_jac=self.fun_jac, step_fun=self.step_fun, tol=1e-6, eps=1e-10, max_steps=max_steps, method=method, verbose=verbose, regularise=True, full_return = full_return, linsearch=linsearch)

        self._set_solved_problem_CReAMa(sol)
    

    def _set_solved_problem_CReAMa(self, solution):
        if self.full_return:
            self.b_out = solution[0][:self.n_nodes]
            self.b_in = solution[0][self.n_nodes:]
            self.comput_time_creama = solution[1]
            self.n_steps_creama = solution[2]
            self.norm_seq_creama = solution[3]
        
        else:
            self.b_out = solution[:self.n_nodes]
            self.b_in = solution[self.n_nodes:]


    def solve_tool(self, model, method, initial_guess=None, adjacency=None, max_steps=100, full_return=False, verbose=False):
        """ function to switch around the various problems
        """
        #TODO: aggiungere tutti i metodi
        if model in ['dcm', 'decm']:
            self._solve_problem(initial_guess=initial_guess, model=model, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)
        elif model in ['CReAMa']:
            self._solve_problem_CReAMa(initial_guess=initial_guess, model=model, adjacency=adjacency, method=method, max_steps=max_steps, full_return=full_return, verbose=verbose)


    def _weighted_realisation(self):
        weighted_realisation = weighted_adjacency(np.concatenate((self.b_out,self.b_in)), self.adjacency_CReAMa, self.is_sparse)
        
        return(weighted_realisation)
