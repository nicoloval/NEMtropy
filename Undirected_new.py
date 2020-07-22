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
                fx += (c[j]-1) * (np.exp(x[j])/(1+np.exp(x[j])*np.exp(x[i])))
            else:
                fx += (c[j]) * (np.exp(x[j])/(1+np.exp(x[j])*np.exp(x[i])))
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
                f -= (c[i]*(c[i]-1)*np.log(1+(np.exp(x[i]))**2))/2
            else:
                f -= (c[i]*c[j]*np.log(1+np.exp(x[i])*np.exp(x[j])))/2
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
                aux = np.exp(x[i])**2
                f[i] += (c[i]-1) * (aux/(1+aux))
            else:
                aux = np.exp(x[i])* np.exp(x[j])
                f[i] += c[j] * (aux/(1+aux))
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
                aux_f = 0
                for h in range(n):
                    if i == h:
                        aux = np.exp(x[h])**2
                        aux_f -= (aux/(1+aux)**2)*(c[h]-1)
                    else:
                        aux = np.exp(x[i])*(x[h])
                        aux_f -= ((aux)/(1+aux)**2)*c[h]
            else:
                aux = np.exp(x[i])*np.exp(x[j])
                aux_f = -((aux)/(1+aux)**2)*c[j]

            f[i, j] = aux_f
            f[j, i] = aux_f
    return f

@jit(forceobj=True)
def linsearch_fun_CM_new(X,args):
    x = X[0]
    dx = X[1]
    beta = X[2]
    alfa = X[3]
    f = X[4]
    step_fun = args[0]
    
    i=0
    s_old = step_fun(x)
    while sufficient_decrease_condition(s_old, \
        step_fun(x + alfa*dx), alfa, f, dx) == False and i<50:
        alfa *= beta
        i +=1
    #print(alfa)
    return alfa

