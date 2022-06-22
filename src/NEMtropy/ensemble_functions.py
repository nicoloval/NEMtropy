import numpy as np
from numba import jit
from . import network_functions as nef

# DBCM functions
# --------------

# 2-nodes motifs
# --------------


@jit(nopython=True)
def expected_dcm_2motif_2(sol):
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
def std_dcm_2motif_2(sol):
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
def expected_dcm_2motif_1(sol):
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
def std_dcm_2motif_1(sol):
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
def expected_dcm_2motif_0(sol):
    """Expected count of zeros after the DBCM.

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
def std_dcm_2motif_0(sol):
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


def zscore_dcm_2motif_2(sol, a):
    count = nef.count_2motif_2(a)
    exp = expected_dcm_2motif_2(sol)
    std = std_dcm_2motif_2(sol)
    return (count - exp)/std


def zscore_dcm_2motif_1(sol, a):
    count = nef.count_2motif_1(a)
    exp = expected_dcm_2motif_1(sol)
    std = std_dcm_2motif_1(sol)
    return (count - exp)/std


def zscore_dcm_2motif_0(sol, a):
    count = nef.count_2motif_0(a)
    exp = expected_dcm_2motif_0(sol)
    std = std_dcm_2motif_0(sol)
    return (count - exp)/std

# 3-nodes motifs
# --------------


@jit(nopython=True)
def expected_dcm_3motif_1(sol):
    """Expected number of 3-nodes motif 1 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 1 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += (1 - pij)*pji*pjk*(1 - pkj)*(1 - pik)*(1 - pki)
                        # s += (1 - pij)*pkj*pki*(1 - pjk)*(1 - pik)*(1 - pji)
    return s


@jit(nopython=True)
def expected_dcm_3motif_2(sol):
    """Expected number of 3-nodes motif 2 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 2 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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

#TODO: write std motif3
@jit(nopython=True)
def expected_dcm_3motif_3(sol):
    """Expected number of 3-nodes motif 3 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 3 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += pij*pji*(1 - pik)*(1 - pki)*pjk*(1 - pkj)
    return s

#TODO: write std motif4
@jit(nopython=True)
def expected_dcm_3motif_4(sol):
    """Expected number of 3-nodes motif 4 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 4 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += (1-pij)*(1-pji)*(1 - pki)*pik*pjk*(1 - pkj)
    return s

@jit(nopython=True)
def expected_dcm_3motif_5(sol):
    """Expected number of 3-nodes motif 5 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 5 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += (1-pij)*pji*pik*(1 - pki)*pjk*(1 - pkj)
    return s


@jit(nopython=True)
def expected_dcm_3motif_6(sol):
    """Expected number of 3-nodes motif 6 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 6 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += pij*pji*pik*(1 - pki)*pjk*(1 - pkj)
    return s


@jit(nopython=True)
def expected_dcm_3motif_7(sol):
    """Expected number of 3-nodes motif 7 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 7 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += pij*pji*(1 - pik)*(1 - pki)*(1 - pjk)*pkj
    return s


@jit(nopython=True)
def expected_dcm_3motif_8(sol):
    """Expected number of 3-nodes motif 8 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 8 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += pij*pji*(1 - pik)*(1 - pki)*pjk*pkj
    return s


@jit(nopython=True)
def expected_dcm_3motif_9(sol):
    """Expected number of 3-nodes motif 9 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 9 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += (1 - pij)*pji*pik*(1 - pki)*(1 - pjk)*pkj
    return s


@jit(nopython=True)
def expected_dcm_3motif_10(sol):
    """Expected number of 3-nodes motif 10 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 10 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += (1 - pij)*pji*pik*(1 - pki)*pjk*pkj
    return s


@jit(nopython=True)
def expected_dcm_3motif_11(sol):
    """Expected number of 3-nodes motif 11 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 11 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += pij*(1 - pji)*pik*(1 - pki)*pjk*pkj
    return s


@jit(nopython=True)
def expected_dcm_3motif_12(sol):
    """Expected number of 3-nodes motif 12 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 12 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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
                        s += pij*pji*pik*(1 - pki)*pjk*pkj
    return s


@jit(nopython=True)
def expected_dcm_3motif_13(sol):
    """Expected number of 3-nodes motif 13 after the DBCM.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Expected motif 13 count.
    :rtype: numpy.float
    """
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
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

## standard deviation 3-motifs

@jit(nopython=True)
def std_dcm_3motif_1(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 1.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 1 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s -= 2*(1-pjk)*pkj*(1-pik)*pki*(1-pij)
                        s += -(1-pkj)*(1-pjk)*(1-pki)*pik*pij + (1-pkj)*pjk*(1-pki)*(1-pik)*(1-pij)
                        s += -pik*(1-pki)*(1-pkj)*(1-pjk)*pij + pjk*(1-pkj)*(1-pki)*(1-pik)*(1-pij)
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)



@jit(nopython=True)
def std_dcm_3motif_2(sol):
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
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s -= ((1-pjk)*pkj*(1-pki)*pik + (1-pkj)*pjk*(1-pik)*pki)*(1-pij)
                        s += (-(1-pkj)*(1-pjk)*(1-pik)*pki*pij + (1-pki)*(1-pik)*(1-pjk)*pkj*(1-pij))
                        s += (-(1-pik)*(1-pki)*(1-pkj)*pjk*pij + (1 - pjk)*(1-pkj)*(1-pki)*pik*(1-pij))
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)

@jit(nopython=True)
def std_dcm_3motif_3(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 3.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 3 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s -= (pjk*pkj*pki*(1 - pik) + (1-pjk)*pkj*pik*pki)*(1 - pij)
                        s += (-(1-pkj)*(1-pjk)*pik*pki*pij + (1-pki)*(1-pik)*pjk*pkj*(1-pij))
                        s += ((1-pik)*(1-pki)*(1-pkj)*pjk*pij + (1 - pjk)*(1-pkj)*(1-pki)*pik*pij)
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)

@jit(nopython=True)
def std_dcm_3motif_4(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 4.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 4 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += ((1 - pjk)*(1 - pkj)*(1 - pik)*pki*(1 - pij) - (1 - pjk)*pkj*(1 - pik)*(1 - pki)*pij)
                        s += -pkj*(1-pjk)*(1 - pik)*(1 - pki)*pij + pki*(1-pik)*(1 - pjk)*(1 - pkj)*(1-pij)
                        s -= 2*(1 - pij)*pik*(1 - pki)*(1 - pkj)*pjk
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_5(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 5.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 5 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += (1 - pjk)*pkj*(1-pik)*pki*(1 - 2*pij)
                        s += (-pkj*(1 - pjk)*pik*(1 - pki)*pij + pki*(1-pik)*pjk*(1 - pkj)*(1-pij))
                        s += (1 - 2*pij)*pik*(1 - pki)*(1 - pkj)*pjk
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_6(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 6.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 6 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pjk*pkj*(1-pik)*pki*(1 - pij) - (1 - pjk)*pkj*pki*pik*pij
                        s += (-pkj*(1 - pjk)*pik*pki*pij + pki*(1-pik)*pjk*pkj*(1 - pij))
                        s += 2*pij*pik*(1 - pki)*(1 - pkj)*pjk
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_7(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 7.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 7 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s -= pjk*pkj*pik*(1 - pki)*(1 - pij) + pjk*(1 - pkj)*pki*pik*(1 - pij)
                        s += ((1 - pkj)*(1 - pjk)*pik*pki*(1 - pij) - (1 - pki)*(1-pik)*pjk*pkj*pij)
                        s += ((1 - pik)*pki*(1 - pkj)*(1 - pjk) + (1 - pik)*(1 - pki)*pkj*(1 - pjk))*pij
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_8(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 8.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 8 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s -= 2*pkj*pjk*pik*pki*(1 - pij)
                        s += pij*((1 - pjk)*(1 - pkj)*pik*pki + pjk*pkj*(1 - pki)*(1 - pik))
                        s += (1 - pik)*(1 - pki)*pkj*pjk*pij + pik*pki*(1 - pkj)*(1 - pjk)*pij
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_9(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 9.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 9 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pik*(1 - pki)*pkj*(1 - pjk)*(1 - pij) - pki*(1 - pik)*pjk*(1 - pkj)*pij
                        s += (1 - pij)*(1 - pjk)*pkj*pik*(1 - pki) - pjk*(1 - pkj)*pki*(1 - pik)*pij
                        s += pik*(1 - pki)*pkj*(1 - pjk)*(1 - pij) - (1 - pik)*pki*(1 - pkj)*pjk*pij
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_10(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 10.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 10 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pkj*(1 - pjk)*pik*pki*(1 - pij) - pjk*pkj*pki*(1 - pik)*pij
                        s += pij*((1 - pjk)*pkj*pik*(1 - pki) + pjk*(1 - pkj)*pki*(1 - pik))
                        s += pik*(1 - pki)*pkj*pjk*(1 - pij) - pik*pki*(1 - pkj)*pjk*pij
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


def std_dcm_3motif_11(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 11.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 11 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += (1 - pkj)*pjk*pik*pki*(1 - pij) - pjk*pkj*(1 - pki)*pik*pij
                        s += (1 - pij)*pjk*(1 - pkj)*pik*pki - pjk*pkj*(1 - pki)*pik*pij
                        s += 2*(1 - pik)*pki*pkj*(1 - pjk)*pij
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


def std_dcm_3motif_12(sol):
    """ compute the standard deviation of the number of 3-nodes motifs 12.

    :param sol: DBCM solution.
    :type sol: numpy.ndarray
    :return: Standard deviation of motif 12 count.
    :rtype: numpy.float
    """
    # edges
    n = int(len(sol)/2)
    x = sol[:n]
    y = sol[n:]
    tmp = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (i != k) and (j != k):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pkj*pjk*pik*pki*(1 - 2*pij)
                        s += pij*(pjk*(1 - pkj)*pik*pki + pjk*pkj*(1 - pki)*pik)
                        s += pij*(pik*pki*pkj*(1 - pjk) + pki*(1 - pik)*pkj*pjk)
                tmp += (1 - pji)*pji*((s)**2)
    return np.sqrt(tmp)


@jit(nopython=True)
def std_dcm_3motif_13(sol):
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
            if i != j:
                pij = x[i]*y[j]/(1 + x[i]*y[j])
                pji = x[j]*y[i]/(1 + x[j]*y[i])
                s = 0
                for k in range(n):
                    if (k != i) and (k != j):
                        pik = x[i]*y[k]/(1 + x[i]*y[k])
                        pki = x[k]*y[i]/(1 + x[k]*y[i])
                        pjk = x[j]*y[k]/(1 + x[j]*y[k])
                        pkj = x[k]*y[j]/(1 + x[k]*y[j])
                        s += pij*pjk*pkj*pki*pik
                tmp += (1 - pji)*pji*((6*s)**2)
    return np.sqrt(tmp)

# z-score
def zscore_dcm_3motif_1(sol, a):
    count = nef.count_3motif_1(a)
    exp = expected_dcm_3motif_1(sol)
    std = std_dcm_3motif_1(sol)
    return (count - exp)/std

def zscore_dcm_3motif_2(sol, a):
    count = nef.count_3motif_2(a)
    exp = expected_dcm_3motif_2(sol)
    std = std_dcm_3motif_2(sol)
    return (count - exp)/std

def zscore_dcm_3motif_3(sol, a):
    count = nef.count_3motif_3(a)
    exp = expected_dcm_3motif_3(sol)
    std = std_dcm_3motif_3(sol)
    return (count - exp)/std

def zscore_dcm_3motif_4(sol, a):
    count = nef.count_3motif_4(a)
    exp = expected_dcm_3motif_4(sol)
    std = std_dcm_3motif_4(sol)
    return (count - exp)/std

def zscore_dcm_3motif_5(sol, a):
    count = nef.count_3motif_5(a)
    exp = expected_dcm_3motif_5(sol)
    std = std_dcm_3motif_5(sol)
    return (count - exp)/std

def zscore_dcm_3motif_6(sol, a):
    count = nef.count_3motif_6(a)
    exp = expected_dcm_3motif_6(sol)
    std = std_dcm_3motif_6(sol)
    return (count - exp)/std

def zscore_dcm_3motif_7(sol, a):
    count = nef.count_3motif_7(a)
    exp = expected_dcm_3motif_7(sol)
    std = std_dcm_3motif_7(sol)
    return (count - exp)/std

def zscore_dcm_3motif_8(sol, a):
    count = nef.count_3motif_8(a)
    exp = expected_dcm_3motif_8(sol)
    std = std_dcm_3motif_8(sol)
    return (count - exp)/std

def zscore_dcm_3motif_9(sol, a):
    count = nef.count_3motif_9(a)
    exp = expected_dcm_3motif_9(sol)
    std = std_dcm_3motif_9(sol)
    return (count - exp)/std

def zscore_dcm_3motif_10(sol, a):
    count = nef.count_3motif_10(a)
    exp = expected_dcm_3motif_10(sol)
    std = std_dcm_3motif_10(sol)
    return (count - exp)/std

def zscore_dcm_3motif_11(sol, a):
    count = nef.count_3motif_11(a)
    exp = expected_dcm_3motif_11(sol)
    std = std_dcm_3motif_11(sol)
    return (count - exp)/std

def zscore_dcm_3motif_12(sol, a):
    count = nef.count_3motif_12(a)
    exp = expected_dcm_3motif_12(sol)
    std = std_dcm_3motif_12(sol)
    return (count - exp)/std

def zscore_dcm_3motif_13(sol, a):
    count = nef.count_3motif_13(a)
    exp = expected_dcm_3motif_13(sol)
    std = std_dcm_3motif_13(sol)
    return (count - exp)/std
