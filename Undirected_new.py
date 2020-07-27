import numpy as np
import scipy.sparse
from numba import jit
import time


@jit(nopython=True)
def iterative_cm_new(x, args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        fx = 0
        for j in np.arange(n):
            if i == j:
                fx += (c[j]-1) * (np.exp(-x[j])/(1+np.exp(-x[j])*np.exp(-x[i])))
            else:
                fx += (c[j]) * (np.exp(-x[j])/(1+np.exp(-x[j])*np.exp(-x[i])))
        if fx:
            f[i] = -np.log(k[i]/fx)

    return f


@jit(nopython=True)
def loglikelihood_cm_new(x, args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = 0.0
    for i in np.arange(n):
        f -= c[i] * k[i] * x[i]
        for j in np.arange(n):
            if i == j:
                f -= (c[i]*(c[i]-1)*np.log(1+(np.exp(-x[i]))**2))/2
            else:
                f -= (c[i]*c[j]*np.log(1+np.exp(-x[i])*np.exp(-x[j])))/2
    return f


@jit(nopython=True)
def loglikelihood_prime_cm_new(x, args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros_like(k, dtype=np.float64)
    for i in np.arange(n):
        f[i] -= k[i]
        for j in np.arange(n):
            if i == j:
                aux = np.exp(-x[i])**2
                f[i] += (c[i]-1) * (aux/(1+aux))
            else:
                aux = np.exp(-x[i])* np.exp(-x[j])
                f[i] += c[j] * (aux/(1+aux))
    return f


@jit(nopython=True)
def loglikelihood_hessian_cm_new(x, args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros(shape=(n, n), dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(i, n):
            if i == j:
                aux_f = 0
                for h in range(n):
                    if i == h:
                        aux = np.exp(-x[h])**2
                        aux_f -= (aux/(1+aux)**2)*(c[h]-1)
                    else:
                        aux = np.exp(-x[i])*np.exp(-x[h])
                        aux_f -= ((aux)/(1+aux)**2)*c[h]
            else:
                aux = np.exp(-x[i])*np.exp(-x[j])
                aux_f = -((aux)/(1+aux)**2)*c[j]

            f[i, j] = aux_f
            f[j, i] = aux_f
    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_cm_new(x, args):
    k = args[0]
    c = args[1]
    n = len(k)
    f = np.zeros(n, dtype=np.float64)
    for i in np.arange(n):
        for j in np.arange(n):
            if i == j:
                aux = np.exp(-x[j])**2
                f[i] -= (aux/(1+aux))*(c[j]-1)
            else:
                aux = np.exp(-x[i])*np.exp(-x[j])
                f[i] -= (aux/(1+aux))*c[j]
    return f


@jit(nopython=True)
def iterative_ecm_new(sol,args):
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
                aux1 = np.exp(-x[i]) * np.exp(-x[j])
                aux2 = np.exp(-y[i]) * np.exp(-y[j])
                fx += ((np.exp(-x[j])*aux2)/(1-aux2+aux1*aux2))
                fy += (aux1*np.exp(-y[j])/((1-aux2)*(1-aux2+aux1*aux2)))
        if fx:
            f[i] = -np.log(k[i] / fx)
        else:
            f[i] = 0.0
        if fy:
            f[i+n] = -np.log(s[i] / fy )
        else:
            f[i+n] = 0.0
    return f


@jit(nopython=True)
def loglikelihood_ecm_new(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = 0.0
    for i in np.arange(n):
        f -= k[i] * x[i] + s[i] * y[i]
        for j in np.arange(0,i):
            aux1 = np.exp(-x[i]) * np.exp(-x[j])
            aux2 = np.exp(-y[i]) * np.exp(-y[j])
            f -= np.log(1 + (aux1 * (aux2/(1-aux2))))
    return f


@jit(nopython=True)
def loglikelihood_prime_ecm_new(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(2*n,dtype = np.float64)
    for i in np.arange(n):
        f[i] -= k[i]
        f[i+n] -= s[i]
        for j in np.arange(n):
            if (i!=j):
                aux1 = np.exp(-x[i])*np.exp(-x[j])
                aux2 = np.exp(-y[i])*np.exp(-y[j])
                f[i] += (aux1*aux2)/(1-aux2+aux1*aux2)
                f[i+n] += (aux1*aux2)/((1-aux2)*(1-aux2+aux1*aux2))
    return f


@jit(nopython=True)
def loglikelihood_hessian_ecm_new(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(shape=(2*n,2*n),dtype=np.float64)
    for i in np.arange(n):

        for j in np.arange(i,n):
            if i==j:
                f1 = - k[i]/(x[i]**2)
                f2 =  - s[i]/((y[i])**2)
                f3 = 0.0
                for h in np.arange(n):
                    if h!=i:
                        aux1 = x[i]*x[h]
                        aux2 = y[i]*y[h]
                        aux3 = (1-aux2)**2
                        aux4 = (1- aux2 + aux1*aux2)**2
                        f1 += ((x[h] * aux2)**2)/aux4
                        f2 += ((aux1*y[h] * (aux1*y[h] * (1-2*aux2) - 2*y[h]*(1-aux2))))/(aux3*aux4)
                        f3 -= (x[h]*y[h])/aux4
                f[i,i] = f1
                f[i+n,i+n] = f2
                f[i+n,i] = f3
                f[i,i+n] = f3
            else:
                aux1 = x[i] * x[j]
                aux2 =  y[i] * y[j] 
                aux3 = (1 - aux2)**2
                aux4 = (1- aux2 + aux1*aux2)**2

                aux = - (aux2*(1-aux2))/aux4
                f[i,j] = aux
                f[j,i] = aux

                aux = -(x[j]*y[i])/aux4
                f[i,j+n] = aux
                f[j+n,i] = aux

                aux = - (aux1 * (1 - aux2**2 + aux1 * (aux2**2)))/(aux3*aux4)
                f[i+n,j+n] = aux
                f[j+n,i+n] = aux

                aux = - (x[i]*y[j])/aux4
                f[i+n,j] = aux
                f[j,i+n] = aux

    return f


@jit(nopython=True)
def loglikelihood_hessian_diag_ecm_new(sol,args):
    k = args[0]
    s = args[1]

    n = len(k)

    x = sol[:n]
    y = sol[n:]
    f = np.zeros(2*n,dtype=np.float64)

    for i in np.arange(n):
        for j in np.arange(n):
            if j!=i:
                aux1 = np.exp(-x[i])*np.exp(-x[j])
                aux2 = np.exp(-y[i])*np.exp(-y[j])
                aux3 = aux1*aux2
                f[i] -= ((aux3)*(1-aux2))/((1-aux2+aux3)**2)
                f[i+n] -= (aux1*aux2*(1-(aux2**2)+aux1*(aux2**2)))/(((1-aux2)**2)*((1-aux2+aux3)**2))
    return f


@jit(nopython=True)
def expected_degree_cm_new(sol):
    ex_k = np.zeros_like(sol, dtype=np.float64)
    n = len(sol)
    for i in np.arange(n):
        for j in np.arange(n):
            if i!=j:
                aux = np.exp(-sol[i])*np.exp(-sol[j])
                ex_k[i] += aux/(1+aux)
    return ex_k
