import numpy as np
from numba import jit, prange
from . import solver_functions as sof


# DBCM functions
# --------------


@jit(nopython=True)
def expected_dyads_dcm(sol):
    """Expected count of dyads after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected dyads count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            temp += x[j]*y[j]/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= x[i]*y[i]/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
        er += x[i]*y[i]*temp
    return er


@jit(nopython=True)
def std_dyads_dcm(sol):
    """ compute the standard deviation of the number of reciprocated links.
    
    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of the dyads count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    temp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                temp += 2*pij*pji*(1 - pij*pji)
    return np.sqrt(temp)


@jit(nopython=True)
def expected_singles_dcm(sol):
     """Expected count of singles after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected singles count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            temp += y[j]*x[i]/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= x[i]*y[i]/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
        er += temp
    return er
    

@jit(nopython=True)
def std_singles_dcm(sol):
    """ compute the standard deviation of the number of reciprocated links.
    
    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of the singles count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    temp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                temp += pij*(1 - pji)*(1 - pij*(1 - pji) - pji*(1 - pij))
    return np.sqrt(temp)



@jit(nopython=True)
def expected_zeros_dcm(sol):
     """Expected count of singles after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected zeros count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    er = 0
    for i in range(n):
        temp = 0
        for j in range(n):
            temp += 1/((1 + x[i]*y[j])*(1 + y[i]*x[j]))
        # i != j should not be accounted
        temp -= 1/((1 + x[i]*y[i])*(1 + y[i]*x[i]))
        er += temp
    return er
    
    
@jit(nopython=True)
def std_zeros_dcm(sol):
    """ compute the standard deviation of the number of zeros couples.
    
    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of the zeros count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    temp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                temp += 2*(1 - pij)*(1 - pji)*(1 - (1 - pij)*(1 - pji))
    return np.sqrt(temp)
