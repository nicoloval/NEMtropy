import numpy as np
from numba import jit
from . import network_functions as nef

# DBCM functions
# --------------

# 2-nodes motifs
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


def dyads_zscore_dcm(sol, a):
    count = nef.dyads_count(a)
    exp = expected_dyads_dcm(sol)
    std = std_dyads_dcm(sol)
    return (count - exp)/std


def singles_zscore_dcm(sol, a):
    count = nef.singles_count(a)
    exp = expected_singles_dcm(sol)
    std = std_singles_dcm(sol)
    return (count - exp)/std


def zeros_zscore_dcm(sol, a):
    count = nef.zeros_count(a)
    exp = expected_zeros_dcm(sol)
    std = std_zeros_dcm(sol)
    return (count - exp)/std

# 3-nodes motifs
# --------------

@jit(nopython=True)
def expected_motif2_dcm(sol):
    """Expected number of 3-nodes motif 2 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 2 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                for k in range(n):
                    if k is not j and k is not i:
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pij*(1 - pji)*(1 - pik)*(1 - pki)*pjk*(1 - pkj)
    return s



@jit(nopython=True)
def expected_motif13_dcm(sol):
    """Expected number of 3-nodes motif 13 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 13 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    y = sol[:n]
    x = sol[n:]
    s = 0
    for i in range(n):
        for j in range(n):
            if j is not i:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                for k in range(n):
                    if k is not j and k is not i:
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pij*pji*pik*pki*pjk*pkj
    return s


@jit(nopython=True)
def std_motif2_dcm(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 2.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 2 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            pij = x[i]*y[j]/(1 + x[i]*y[j])
            pji = x[j]*y[i]/(1 + x[j]*y[i])
            s = 0
            for k in range(n):
                if (i != k) and (j != k):
                    pik = x[i]*y[k]/(1 + x[i]*y[k])
                    pki = x[k]*y[i]/(1 + x[k]*y[i])
                    pjk = x[j]*y[k]/(1 + x[j]*y[k])
                    pkj = x[k]*y[j]/(1 + x[k]*y[j])
                    s -= ((1-pjk)*pkj*(1-pki)*pik + (1-pkj)*pjk*(1-pik)*pki*(1-pij))
                    s += ((1-pkj)*pki*(1-pjk)*(-pij) + pkj*(1-pki)*(1-pik)*(1-pjk)*(1-pij))
                    s += ((1-pik)*(1-pki)*(1-pkj)*pjk*(-pij) + (1 - pjk)*(1-pkj)*(1-pki)*pik*(1-pij))
            tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_motif13_dcm(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 13.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 13 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            pij = x[i]*y[j]/(1 + x[i]*y[j])
            pji = x[j]*y[i]/(1 + x[j]*y[i])
            s = 0
            for k in range(n):
                pik = x[i]*y[k]/(1 + x[i]*y[k])
                pki = x[k]*y[i]/(1 + x[k]*y[i])
                pjk = x[j]*y[k]/(1 + x[j]*y[k])
                pkj = x[k]*y[j]/(1 + x[k]*y[j])
                s += pij*pjk*pkj*pki*pik*(1 - int(j == k))*(1 - int(k == i))
            tmp += (1 - pji)*pji*((6*s)**2)
    return np.sqrt(tmp)


def motif13_zscore_dcm(sol, a):
    count = nef.motif13_count(a)
    exp = expected_motif13_dcm(sol)
    std = std_motif13_dcm(sol)
    return (count - exp)/std


def motif2_zscore_dcm(sol, a):
    count = nef.motif2_count(a)
    exp = expected_motif2_dcm(sol)
    std = std_motif13_dcm(sol)
    return (count - exp)/std
