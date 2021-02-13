import multiprocessing as mp
import numpy as np
# import warnings
# import sys


def ensemble_sampler_cm_graph(outfile_name, x, cpu_n=2, seed=None):
    """Produce a single undirected binary graph after UBCM.
        The graph is saved as an edges list on a file.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param x: UBCM solution
    :type x: numpy.ndarray
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(x))

    # put together inputs for pool
    iter_ = iter(
        ((i, xi), (j, xj), np.random.randint(0, 1000000))
        for i, xi in zip(inds, x)
        for j, xj in zip(inds, x)
        if i < j)

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_cm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {}\n".format(str(i), str(j))
                for (i, j) in edges_list)
            )

    return outfile_name


def ensemble_sampler_ecm_graph(outfile_name, x, y, cpu_n=2, seed=None):
    """Produce a single undirected weighted graph after ECM.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param x: ECM degrees solution
    :type x: numpy.ndarray
    :param y: ECM strengths solution
    :type y: numpy.ndarray
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(x))

    # put together inputs for pool
    iter_ = iter(
        ((i, xi, yi), (j, xj, yj), np.random.randint(0, 1000000))
        for i, xi, yi in zip(inds, x, y)
        for j, xj, yj in zip(inds, x, y)
        if i < j)

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_ecm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_dcm_graph(outfile_name, x, y, cpu_n=2, seed=None):
    """Produce a single directed binary graph after DBCM.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param x: DBCM out-degrees parameters solution
    :type x: numpy.ndarray
    :param y: DBCM in-degrees parameters solution
    :type y: numpy.ndarray
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(x))

    # put together inputs for pool
    iter_ = iter(((i, xi), (j, yj), np.random.randint(0, 1000000))
                 for i, xi in zip(inds, x)
                 for j, yj in zip(inds, y)
                 if i != j)

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_dcm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {}\n".format(str(i), str(j))
                for (i, j) in edges_list)
            )

    return outfile_name


def ensemble_sampler_decm_graph(
        outfile_name,
        a_out, a_in,
        b_out, b_in,
        cpu_n=2,
        seed=None):
    """Produce a single directed weighted graph after DECM.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param a_out: DECM out-degrees parameters solution
    :type a_out: numpy.ndarray
    :param a_in: DECM in-degrees parameters solution
    :type a_in: numpy.ndarray
    :param b_out: DECM out-strengths parameters solution
    :type b_out: numpy.ndarray
    :param b_in: DECM in-strengths parameters solution
    :type b_in: numpy.ndarray
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(a_out))

    # put together inputs for pool
    iter_ = iter((
                    (i, a_out_i, b_out_i),
                    (j, a_in_j, b_in_j),
                    np.random.randint(0, 1000000))
                 for i, a_out_i, b_out_i in zip(inds, a_out, b_out)
                 for j, a_in_j, b_in_j in zip(inds, a_in, b_in)
                 if i != j)

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_decm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_crema_ecm_det_graph(
        outfile_name,
        beta,
        adj,
        cpu_n=2,
        seed=None):
    """Produce a single undirected weighted graph after CReMa,
        given a fixed, complete network topology.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param beta: CReMa solution
    :type beta: numpy.ndarray
    :param adj: Edges list: out-nodes array, in-nodes array and links weights.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    (row_inds, col_inds, weigths_value) = adj
    del weigths_value

    # put together inputs for pool
    iter_ = iter(
        ((i, beta[i]), (j, beta[j]), np.random.randint(0, 1000000))
        for i, j in zip(row_inds, col_inds)
    )

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_crema_ecm_det, iter_)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_crema_ecm_prob_graph(
        outfile_name,
        beta,
        adj,
        cpu_n=2,
        seed=None):
    """Produce a single undirected weighted graph after CReMa,
        given a probabilistic network topology.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param beta: CReMa solution
    :type beta: numpy.ndarray
    :param adj: Edges list:
         out-nodes array, in-nodes array and links probabilities.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    (row_inds, col_inds, weigths_value) = adj

    # put together inputs for pool
    iter_ = iter(
        ((i, beta[i]), (j, beta[j]), w_prob, np.random.randint(0, 1000000))
        for i, j, w_prob in zip(row_inds, col_inds, weigths_value)
    )

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_crema_ecm_prob, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_crema_sparse_ecm_prob_graph(
        outfile_name,
        beta,
        adj,
        cpu_n=2,
        seed=None):
    """Produce a single undirected weighted graph after CReMa,
        given a probabilistic network topology.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param beta: CReMa solution
    :type beta: numpy.ndarray
    :param adj: Edges list:
         out-nodes array, in-nodes array and links probabilities.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    x = adj[0]
    n = len(x)

    # put together inputs for pool
    iter_ = iter(
        ((i, beta[i], x[i]), (j, beta[j], x[j]), np.random.randint(0, 1000000))
        for i in range(n)
        for j in range(n)
        if i != j
    )

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_crema_sparse_ecm_prob, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # debug
    # print(edges_list)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_crema_decm_prob_graph(
        outfile_name,
        beta,
        adj,
        cpu_n=2,
        seed=None):
    """Produce a single directed weighted graph after CReMa,
        given a probabilistic network topology.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param beta: CReMa solution
    :type beta: numpy.ndarray
    :param adj: Edges list:
         out-nodes array, in-nodes array and links probabilities.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    b_out, b_in = beta

    (row_inds, col_inds, weigths_value) = adj

    # put together inputs for pool
    iter_ = iter(
        ((i, b_out[i]), (j, b_in[j]), w_prob, np.random.randint(0, 1000000))
        for i, j, w_prob in zip(row_inds, col_inds, weigths_value)
    )

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_crema_decm_prob, iter_)

    # debug
    # print(edges_list)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_crema_decm_det_graph(
        outfile_name,
        beta,
        adj,
        cpu_n=2,
        seed=None):
    """Produce a single directed weighted graph after CReMa,
        given a fixed, complete network topology.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param beta: CReMa solution
    :type beta: numpy.ndarray
    :param adj: Edges list:
         out-nodes array, in-nodes array and links weights.
    :type adj: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    b_out, b_in = beta

    (row_inds, col_inds, weigths_value) = adj
    del weigths_value

    # put together inputs for pool
    iter_ = iter(
        ((i, b_out[i]), (j, b_in[j]), np.random.randint(0, 1000000))
        for i, j in zip(row_inds, col_inds)
    )

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_crema_decm_det, iter_)

    # debug
    # print(edges_list)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def ensemble_sampler_crema_sparse_decm_prob_graph(
        outfile_name,
        beta,
        adj,
        cpu_n=2,
        seed=None):
    """Produce a single directed weighted graph after the
        sparse version of CReMa, given a probabilistic network topology.
        The graph is saved as an edges list on a file.
        The function is parallelyzed.

    :param outfile_name: Name of the file where to save the graph on.
    :type outfile_name: str
    :param beta: CReMa solution
    :type beta: numpy.ndarray
    :param adj: Tuple of DBCM out-degrees parameters and
        in-degrees parameters solution.
    :type adj: (numpy.ndarray, numpy.ndarray)
    :param cpu_n: Number of cpus to use, defaults to 2
    :type cpu_n: int, optional
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: The `outfile_name` input parameter
    :rtype: str
    """
    if seed is not None:
        np.random.seed(seed)

    b_out, b_in = beta
    x, y = adj
    n = len(x)

    # put together inputs for pool
    iter_ = iter(
        (
            (i, b_out[i], x[i]),
            (j, b_in[j], y[j]),
            np.random.randint(0, 1000000))
        for i in range(n)
        for j in range(n)
        if i != j
    )

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_crema_sparse_decm_prob, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w))
                for (i, j, w) in edges_list)
            )

    return outfile_name


def is_a_link_cm(args_1, args_2, seed=None):
    """The function randomly returns an undirected link
        between two given nodes after the UBCM.

    :param args_1: Tuple containing node 1 and related parameter solution.
    :type args_1: (int, float)
    :param args_2: Tuple containing node 2 and related parameter solution.
    :type args_2: (int, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the couple of nodes,
        otherwise returns None
    :rtype: (int, int)
    """
    if seed is not None:
        np.random.seed(seed)
    (i, xi) = args_1
    (j, xj) = args_2
    p = np.random.random_sample()
    xij = xi*xj
    p_ensemble = xij/(1 + xij)
    if p < p_ensemble:
        return (i, j)


def is_a_link_ecm(args_1, args_2, seed=None):
    """The function randomly returns an undirected link with related weight
        between two given nodes after the ECM.

    :param args_1: Tuple containing node 1, degree parameter solution
        and strength parameter solution.
    :type args_1: (int, float, float)
    :param args_2: Tuple containing node 2, degree parameter solution
        and strength parameter solution.
    :type args_2: (int, float, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # q-ensemble source: "Unbiased sampling of network ensembles", WUNs
    if seed is not None:
        np.random.seed(seed)
    (i, xi, yi) = args_1
    (j, xj, yj) = args_2
    p = np.random.random_sample()
    xij = xi*xj
    yij = yi*yj
    p_ensemble = xij*yij/(1 - yij + xij*yij)
    if p < p_ensemble:
        q_ensemble = yij
        w = np.random.geometric(1-q_ensemble)
        return (i, j, w)


def is_a_link_dcm(args_1, args_2, seed=None):
    """The function randomly returns a directed link
        between two given nodes after the DBCM.

    :param args_1: Tuple containing out node and out-degree parameter solution.
    :type args_1: (int, float)
    :param args_2: Tuple containing in node and in-degree parameter solution.
    :type args_2: (int, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the couple of nodes,
        otherwise returns None
    :rtype: (int, int)
    """
    if seed is not None:
        np.random.seed(seed)
    (i, xi) = args_1
    (j, yj) = args_2
    p = np.random.random_sample()
    tmp = xi*yj
    p_ensemble = tmp/(1 + tmp)
    if p < p_ensemble:
        return (i, j)


def is_a_link_decm(args_1, args_2, seed=None):
    """The function randomly returns an directed link with related weight
        between two given nodes after the DECM.

    :param args_1: Tuple containing out node, out-degree parameter solution
        and out-strength parameter solution.
    :type args_1: (int, float, float)
    :param args_2: Tuple containing in node, in-degree parameter solution
        and in-strength parameter solution.
    :type args_2: (int, float, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # q-ensemble source:
    # "Fast and scalable resolution of the likelihood maximization problem
    # for Exponential Random Graph models"
    if seed is not None:
        np.random.seed(seed)
    (i, a_out_i, b_out_i) = args_1
    (j, a_in_j, b_in_j) = args_2
    p = np.random.random_sample()
    aij = a_out_i * a_in_j
    bij = b_out_i * b_in_j
    p_ensemble = aij*bij/(1 - bij + aij*bij)
    if p < p_ensemble:
        q_ensemble = bij
        w = np.random.geometric(1-q_ensemble)
        return (i, j, w)


def is_a_link_crema_ecm_det(args_1, args_2, seed=None):
    """The function randomly returns an undirected link with related weight
        between two given nodes after the CReMa undirected
        for a deterministic topology.

    :param args_1: Tuple containing node 1, and strength parameter solution.
    :type args_1: (int, float)
    :param args_2: Tuple containing node 2, and strength parameter solution.
    :type args_2: (int, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # Q-ensemble source: "A faster Horse on a safer trail".
    if seed is not None:
        np.random.seed(seed)
    (i, beta_i) = args_1
    (j, beta_j) = args_2

    q_ensemble = 1/(beta_i + beta_j)
    w_link = np.random.exponential(q_ensemble)
    return (i, j, w_link)


def is_a_link_crema_ecm_prob(args_1, args_2, p_ensemble, seed=None):
    """The function randomly returns an undirected link with related weight
        between two given nodes after the CReMa undirected
        for probabilistic o topology.

    :param args_1: Tuple containing node 1, and strength parameter solution.
    :type args_1: (int, float)
    :param args_2: Tuple containing node 2, and strength parameter solution.
    :type args_2: (int, float)
    :param p_ensemble: Probability the link exists
    :type p_ensemble: float
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # Q-ensemble source: "A faster Horse on a safer trail".
    if seed is not None:
        np.random.seed(seed)
    (i, beta_i) = args_1
    (j, beta_j) = args_2

    p = np.random.random_sample()
    if p < p_ensemble:
        q_ensemble = 1/(beta_i + beta_j)
        w_link = np.random.exponential(q_ensemble)
        return (i, j, w_link)


def is_a_link_crema_sparse_ecm_prob(args_1, args_2, seed=None):
    """The function randomly returns an undirected link with related weight
        between two given nodes after the CReMa undirected
        for probabilistic o topology.
        Function for the sparse version of CReMa.

    :param args_1: Tuple containing node 1, strength parameter solution and
        degree parameter solution from the probabilistic model.
    :type args_1: (int, float)
    :param args_2: Tuple containing node 2, strength parameter solution and
        degree parameter solution from the probabilistic model
    :type args_2: (int, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # Q-ensemble source: "A faster Horse on a safer trail".
    if seed is not None:
        np.random.seed(seed)
    (i, beta_i, x_i) = args_1
    (j, beta_j, x_j) = args_2

    p = np.random.random_sample()
    p_ensemble = x_i*x_j/(1 + x_j*x_i)
    if p < p_ensemble:
        q_ensemble = 1/(beta_i + beta_j)
        w_link = np.random.exponential(q_ensemble)
        return (i, j, w_link)


def is_a_link_crema_decm_det(args_1, args_2, seed=None):
    """The function randomly returns a directed link with related weight
        between two given nodes after the CReMa undirected for a deterministic
         topology.

    :param args_1: Tuple containing out node and out-strength parameter
        solution.
    :type args_1: (int, float)
    :param args_2: Tuple containing in node and in-strength parameter solution.
    :type args_2: (int, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # Q-ensemble source: "A faster Horse on a safer trail".
    if seed is not None:
        np.random.seed(seed)
    (i, b_out_i) = args_1
    (j, b_in_j) = args_2

    q_ensemble = 1/(b_out_i + b_in_j)
    w_link = np.random.exponential(q_ensemble)
    return (i, j, w_link)


def is_a_link_crema_decm_prob(args_1, args_2, p_ensemble, seed=None):
    """The function randomly returns a directed link with related weight
        between two given nodes after the CReMa directed
        for a probabilistic topology.

    :param args_1: Tuple containing out node and out-strength parameter
        solution.
    :type args_1: (int, float)
    :param args_2: Tuple containing in node and in-strength parameter solution.
    :type args_2: (int, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # Q-ensemble source: "A faster Horse on a safer trail".
    if seed is not None:
        np.random.seed(seed)
    (i, b_out_i) = args_1
    (j, b_in_j) = args_2

    p = np.random.random_sample()
    if p < p_ensemble:
        q_ensemble = 1/(b_out_i + b_in_j)
        w_link = np.random.exponential(q_ensemble)
        return (i, j, w_link)


def is_a_link_crema_sparse_decm_prob(args_1, args_2, seed=None):
    """The function randomly returns a directed link with related weight
        between two given nodes after the CReMa directed
        for a probabilistic topology.
        Function for the sparse version of CReMa.

    :param args_1: Tuple containing out node, out-strength parameter solution
        and out-degree parameter solution from the probabilistic model
    :type args_1: (int, float, float)
    :param args_2: Tuple containing in node and in-strength parameter solution
        and in-degree parameter solution from the probabilistic model
    :type args_2: (int, float, float)
    :param seed: Random seed, defaults to None
    :type seed: int, optional
    :return: If the links exists returns the triplet of nodes and weight,
        otherwise returns None
    :rtype: (int, int, float)
    """
    # Q-ensemble source: "A faster Horse on a safer trail".
    if seed is not None:
        np.random.seed(seed)
    (i, b_out_i, x_i) = args_1
    (j, b_in_j, x_j) = args_2

    p = np.random.random_sample()
    p_ensemble = x_i*x_j/(1 + x_j*x_i)
    if p < p_ensemble:
        q_ensemble = 1/(b_out_i + b_in_j)
        w_link = np.random.exponential(q_ensemble)
        return (i, j, w_link)
