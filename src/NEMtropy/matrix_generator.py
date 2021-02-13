import numpy as np
import scipy.sparse
import scipy
import networkx as nx
import powerlaw as plw


def random_binary_matrix_generator_dense(n, sym=False, seed=None):
    """Generates a densely connected random binary adjacency matrix.

    :param n: Number of nodes.
    :type n: int
    :param sym: If True the adjacency matrix is simmetric, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :return: Binary adjacency matrix.
    :rtype: numpy.ndarray
    """
    if not sym:
        np.random.seed(seed=seed)
        A = np.random.randint(0, 2, size=(n, n))
        # zeros on the diagonal
        np.fill_diagonal(A, 0)
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]), k_in, k_out):
            if (ki == 0) and (ko == 0):
                while (np.sum(A[:, ind]) == 0) and (np.sum(A[ind, :]) == 0):
                    if np.random.random() > 0.5:
                        A[np.random.randint(A.shape[0]), ind] = 1
                    else:
                        A[ind, np.random.randint(A.shape[0])] = 1

                    A[ind, ind] = 0
        return A
    else:
        np.random.seed(seed=seed)
        A = np.random.randint(0, 2, size=(n, n))
        A = ((A + A.T) / 2).astype(int)
        # zeros on the diagonal
        np.fill_diagonal(A, 0)
        k = np.sum(A, axis=0)
        for ind, kk in zip(np.arange(k.shape[0]), k):
            if kk == 0:
                while np.sum(A[:, ind]) == 0:
                    indices = np.random.randint(A.shape[0])
                    A[indices, ind] = 1
                    A[ind, indices] = 1
                    A[ind, ind] = 0
        return A


def random_binary_matrix_generator_custom_density(
    n, p=0.1, sym=False, seed=None
):
    """Generates a random binary adjacency matrix with density determined by the parameter p.

    :param n: Number of nodes.
    :type n: int
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sym: If True the adjacency matrix is simmetric, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :return: Binary adjacency matrix.
    :rtype: numpy.ndarray
    """
    if not sym:
        np.random.seed(seed=seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if np.random.random() <= p:
                    A[i, j] = 1
        np.fill_diagonal(A, 0)
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]), k_in, k_out):
            if (ki == 0) and (ko == 0):
                while (np.sum(A[:, ind]) == 0) and (np.sum(A[ind, :]) == 0):
                    if np.random.random() > 0.5:
                        A[np.random.randint(A.shape[0]), ind] = 1
                    else:
                        A[ind, np.random.randint(A.shape[0])] = 1

                    A[ind, ind] = 0
        return A
    else:
        np.random.seed(seed=seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() <= p:
                    A[i, j] = 1
                    A[j, i] = A[i, j]
        np.fill_diagonal(A, 0)
        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k == 0:
                while np.sum(A[:, ind]) == 0:
                    indices = np.random.randint(A.shape[0])
                    if indices != ind:
                        A[ind, indices] = 1
                        A[indices, ind] = A[ind, indices]
                    A[ind, ind] = 0
        return A


def random_weighted_matrix_generator_dense(
    n, sup_ext=10, sym=False, seed=None, intweights=False
):
    """Generates a densely connected random weighted adjacency matrix.

    :param n: Number of nodes.
    :type n: int
    :param sup_ext: Maximum strength, defaults to 10.
    :type sup_ext: int, float, optional
    :param sym: If True the adjacency matrix is simmetric, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param intweights: If True weights are integers, defaults to False.
    :type intweights: bool, optional
    :return: Weighted adjacency matrix.
    :rtype: numpy.ndarray
    """
    if not sym:
        np.random.seed(seed=seed)
        A = np.random.random(size=(n, n)) * sup_ext
        np.fill_diagonal(A, 0)

        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]), k_in, k_out):
            if (ki == 0) and (ko == 0):
                while (np.sum(A[:, ind]) == 0) and (np.sum(A[ind, :]) == 0):
                    if np.random.random() > 0.5:
                        A[np.random.randint(A.shape[0]), ind] = (
                            np.random.random() * sup_ext
                        )
                    else:
                        A[ind, np.random.randint(A.shape[0])] = (
                            np.random.random() * sup_ext
                        )

                    A[ind, ind] = 0
        if intweights:
            return np.ceil(A)
        else:
            return A
    else:
        np.random.seed(seed=seed)
        b = np.random.random(size=(n, n)) * sup_ext
        A = (b + b.T) / 2
        np.fill_diagonal(A, 0)

        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k == 0:
                while np.sum(A[:, ind]) == 0:
                    indices = np.random.randint(A.shape[0])
                    if indices != ind:
                        A[ind, indices] = np.random.random() * sup_ext
                        A[indices, ind] = A[ind, indices]
        if intweights:
            return np.ceil(A)
        else:
            return A


def random_weighted_matrix_generator_uniform_custom_density(
    n, p=0.1, sup_ext=10, sym=False, seed=None, intweights=False
):
    """Generates a random weighted adjacency matrix with density determined
    by the parameter p and weights extracted from a uniform distribution.

    :param n: Number of nodes.
    :type n: int
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sup_ext: Maximum strength, defaults to 10.
    :type sup_ext: int or float, optional
    :param sym: If True the adjacency matrix is simmetric, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param intweights: If True weights are integers, defaults to False.
    :type intweights: bool, optional
    :return: Weighted adjacency matrix.
    :rtype: numpy.ndarray
    """
    if not sym:
        np.random.seed(seed=seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if np.random.random() <= p:
                    A[i, j] = np.random.random() * sup_ext
        np.fill_diagonal(A, 0)
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]), k_in, k_out):
            if (ki == 0) and (ko == 0):
                while (np.sum(A[:, ind]) == 0) and (np.sum(A[ind, :]) == 0):
                    if np.random.random() > 0.5:
                        A[np.random.randint(A.shape[0]), ind] = (
                            np.random.random() * sup_ext
                        )
                    else:
                        A[ind, np.random.randint(A.shape[0])] = (
                            np.random.random() * sup_ext
                        )

                    A[ind, ind] = 0
        if intweights:
            return np.ceil(A)
        else:
            return A
    else:
        np.random.seed(seed=seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() <= p:
                    A[i, j] = np.random.random() * sup_ext
                    A[j, i] = A[i, j]
        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k == 0:
                while np.sum(A[:, ind]) == 0:
                    indices = np.random.randint(A.shape[0])
                    if indices != ind:
                        A[ind, indices] = np.random.random() * sup_ext
                        A[indices, ind] = A[ind, indices]
        if intweights:
            return np.ceil(A)
        else:
            return A


def random_weighted_matrix_generator_gaussian_custom_density(
    n, mean, sigma, p=0.1, sym=False, seed=None, intweights=False
):
    """Generates a random weighted adjacency matrix with density determined
    by the parameter p and weights extracted from a gaussian distribution.

    :param n: Number of nodes.
    :type n: int
    :param mean: Mean of the gaussian.
    :type mean: int, float
    :param sigma: Sigma of the gaussian.
    :type sigma: int, float
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sym: If True the adjacency matrix is simmetric, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param intweights: If True weights are integers, defaults to False.
    :type intweights: bool, optional
    :return: Weighted adjacency matrix.
    :rtype: numpy.ndarray
    """
    if not sym:
        np.random.seed(seed=seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if np.random.random() <= p:
                    A[i, j] = np.random.normal(loc=mean, scale=sigma)
        np.fill_diagonal(A, 0)
        k_in = np.sum(A, axis=0)
        k_out = np.sum(A, axis=1)
        for ind, ki, ko in zip(np.arange(k_in.shape[0]), k_in, k_out):
            if (ki == 0) and (ko == 0):
                while (np.sum(A[:, ind]) == 0) and (np.sum(A[ind, :]) == 0):
                    if np.random.random() > 0.5:
                        A[
                            np.random.randint(A.shape[0]), ind
                        ] = np.random.normal(loc=mean, scale=sigma)
                    else:
                        A[
                            ind, np.random.randint(A.shape[0])
                        ] = np.random.normal(loc=mean, scale=sigma)

                    A[ind, ind] = 0
        if intweights:
            return np.ceil(A)
        else:
            return A
    else:
        np.random.seed(seed=seed)
        A = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() <= p:
                    A[i, j] = np.random.normal(loc=mean, scale=sigma)
                    A[j, i] = A[i, j]
        np.fill_diagonal(A, 0)
        degree = np.sum(A, axis=0)
        for ind, k in enumerate(degree):
            if k == 0:
                while np.sum(A[:, ind]) == 0:
                    indices = np.random.randint(A.shape[0])
                    if indices != ind:
                        A[indices, ind] = np.random.normal(
                            loc=mean, scale=sigma
                        )
                        A[ind, indices] = A[indices, ind]
        if intweights:
            return np.ceil(A)
        else:
            return A


def random_binary_matrix_generator_custom_density_sparse(
    n, p=0.1, sym=False, seed=None
):
    """Generates a random binary adjacency sparse matrix
    with density determined by the parameter p.

    :param n: Number of nodes.
    :type n: int
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sym: If True the adjacency matrix is simmetric, defaults to False.
    :type sym: bool, optional
    :param seed: seed of the random process, defaults to None.
    :type seed: int, optional
    :return: a binary sparse adjacency matrix.
    :rtype: scipy.sparse_matrix
    """
    if sym:
        nx_graph = nx.fast_gnp_random_graph(n, p=p, seed=seed)
        largest_cc = max(nx.connected_components(nx_graph), key=len)
        nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()
        adj_sparse = nx.to_scipy_sparse_matrix(nx_graph_lcc)
        return adj_sparse

    else:
        nx_graph = nx.fast_gnp_random_graph(n, p=p, seed=seed, directed=True)
        largest_cc = max(nx.weakly_connected_components(nx_graph), key=len)
        nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()
        adj_sparse = nx.to_scipy_sparse_matrix(nx_graph_lcc)
        return adj_sparse


def random_uniform_weighted_matrix_generator_custom_density_sparse(
    n, sup_ext, p=0.1, sym=False, seed=None, intweights=False
):
    """Generates a random weighted adjacency sparse matrix with density
    determined by the parameter p and weights extracted
    from a uniform distribution.

    :param n: Number of nodes.
    :type n: int
    :param sup_ext: Maximum strength, defaults to 10.
    :type sup_ext: int, float, optional
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sym: If True the adjacency matrix is simmetrix, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param intweights: If True weights are integers, defaults to False.
    :type intweights: bool, optional
    :return: Weighted sparse adjacency matrix.
    :rtype: scipy.sparse_matrix
    """
    if sym:
        nx_graph = nx.fast_gnp_random_graph(n, p=p, seed=seed)
        largest_cc = max(nx.connected_components(nx_graph), key=len)
        nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()
        adj_sparse = nx.to_scipy_sparse_matrix(nx_graph_lcc)
        adj_sparse = adj_sparse.astype(np.float64)

        # Weigths
        raw_ind, col_ind = scipy.sparse.triu(adj_sparse).nonzero()
        if intweights:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.ceil(np.random.random() * sup_ext)
                adj_sparse[j, i] = adj_sparse[i, j]
        else:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.random.random() * sup_ext
                adj_sparse[j, i] = adj_sparse[i, j]

        return adj_sparse

    else:
        nx_graph = nx.fast_gnp_random_graph(n, p=p, seed=seed, directed=True)
        largest_cc = max(nx.weakly_connected_components(nx_graph), key=len)
        nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()
        adj_sparse = nx.to_scipy_sparse_matrix(nx_graph_lcc)
        adj_sparse = adj_sparse.astype(np.float64)

        # Weigths
        raw_ind, col_ind = adj_sparse.nonzero()
        if intweights:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.ceil(np.random.random() * sup_ext)
        else:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.random.random() * sup_ext

        return adj_sparse


def random_gaussian_weighted_matrix_generator_custom_density_sparse(
    n, mean, sigma, p=0.1, sym=False, seed=None, intweights=False
):
    """Generates a random weighted adjacency sparse matrix with density
    determined by the parameter p and weights
    extracted from a gaussian distribution.

    :param n: Number of nodes.
    :type n: int
    :param mean: Mean of the gaussian.
    :type mean: int, float
    :param sigma: Sigma of the gaussian.
    :type sigma: int, float
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sym: If True the adjacency matrix is simmetrix, defaults to False.
    :type sym: bool, optional
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param intweights: If True weights are integers, defaults to False.
    :type intweights: bool, optional
    :return: Weighted sparse adjacency matrix.
    :rtype: scipy.sparse_matrix
    """
    if sym:
        nx_graph = nx.fast_gnp_random_graph(n, p=p, seed=seed)
        largest_cc = max(nx.connected_components(nx_graph), key=len)
        nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()
        adj_sparse = nx.to_scipy_sparse_matrix(nx_graph_lcc)
        adj_sparse = adj_sparse.astype(np.float64)

        # Weigths
        raw_ind, col_ind = scipy.sparse.triu(adj_sparse).nonzero()
        if intweights:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.ceil(
                    np.random.normal(loc=mean, scale=sigma)
                )
                adj_sparse[j, i] = adj_sparse[i, j]
        else:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.random.normal(loc=mean, scale=sigma)
                adj_sparse[j, i] = adj_sparse[i, j]

        return adj_sparse

    else:
        nx_graph = nx.fast_gnp_random_graph(n, p=p, seed=seed, directed=True)
        largest_cc = max(nx.weakly_connected_components(nx_graph), key=len)
        nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()
        adj_sparse = nx.to_scipy_sparse_matrix(nx_graph_lcc)
        adj_sparse = adj_sparse.astype(np.float64)

        # Weigths
        raw_ind, col_ind = adj_sparse.nonzero()
        if intweights:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.ceil(
                    np.random.normal(loc=mean, scale=sigma)
                )
        else:
            for i, j in zip(raw_ind, col_ind):
                adj_sparse[i, j] = np.random.normal(loc=mean, scale=sigma)

        return adj_sparse


def random_graph_nx(
    n, p, sup_ext, alpha, seed=None, is_weighted=None, is_sparse=False
):
    """Retunrs an undirected weighted adjacency matrix using erdos-renyi model
    to generate the binary adjacency and assigning weights extracted'
    from a uniform, gaussian or powerlaw distribution.

    :param n: umber of nodes.
    :type n: int
    :param p: Probability of observing a link between a couple of nodes,
     defaults to 0.1.
    :type p: float, optional
    :param sup_ext: Maximum strength, defaults to 10.
    :type sup_ext: int, float, optional
    :param alpha: Powerlaw exponent.
    :type alpha: float
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param is_weighted: If True the adjacency matrix is weighted,
     defaults to None.
    :type is_weighted: bool, optional
    :param is_sparse: If True the adjacency matrix is sparse,
     defaults to False.
    :type is_sparse: bool, optional
    :return: Adjacency matrix.
    :rtype: numpy.ndarray, scipy.sparse_matrix
    """
    if seed is None:
        seed = np.random.randint(0, n ** 2)
    nx_graph = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()

    np.random.seed(seed=seed)
    if is_weighted == "uniform":
        for e in nx_graph_lcc.edges:
            nx_graph_lcc[e[0]][e[1]]["weight"] = np.random.randint(0, sup_ext)
    elif is_weighted == "gaussian":
        for e in nx_graph_lcc.edges:
            nx_graph_lcc[e[0]][e[1]]["weight"] = np.random.normal(
                loc=sup_ext, scale=sup_ext / 5.5
            )
    elif is_weighted == "powerlaw":
        for e in nx_graph_lcc.edges:
            nx_graph_lcc[e[0]][e[1]]["weight"] = plw.Power_Law(
                xmin=1, xmax=sup_ext, parameters=[alpha], discrete=True
            ).generate_random(1)

    if is_sparse:
        adjacency = nx.to_scipy_sparse_matrix(nx_graph_lcc)
    else:
        adjacency = nx.to_numpy_array(nx_graph_lcc)

    return adjacency


def barabasi_albert_graph_nx(
    n, m, sup_ext, alpha, seed=None, is_weighted=None, is_sparse=False
):
    """Generates a undirected weighted adjacency matrix using barabasi-albert model
    for the binary part and assigning weights extracted from
    a uniform, gaussian or powerlaw distribution.

    :param n: Number of nodes.
    :type n: int
    :param m: Number of edges to attach from a new node to existing nodes.
    :type m: int
    :param sup_ext: Maximum strength, defaults to 10.
    :type sup_ext: int, float, optional
    :param alpha: Powerlaw exponent.
    :type alpha: float
    :param seed: Seed of the random process, defaults to None.
    :type seed: int, optional
    :param is_weighted: If True the adjacency matrix is weighted,
     defaults to None.
    :type is_weighted: bool, optional
    :param is_sparse: If True the adjacency matrix is sparse,
     defaults to False.
    :type is_sparse: bool, optional
    :return: Adjacency matrix.
    :rtype: numpy.ndarray, scipy.sparse_matrix
    """
    if seed is None:
        seed = np.random.randint(0, n ** 2)
    nx_graph = nx.barabasi_albert_graph(n, m, seed=seed)
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    nx_graph_lcc = nx_graph.subgraph(largest_cc).copy()

    np.random.seed(seed=seed)
    if is_weighted == "uniform":
        for e in nx_graph_lcc.edges:
            nx_graph_lcc[e[0]][e[1]]["weight"] = np.random.randint(0, sup_ext)
    elif is_weighted == "gaussian":
        for e in nx_graph_lcc.edges:
            nx_graph_lcc[e[0]][e[1]]["weight"] = np.random.normal(
                loc=sup_ext, scale=sup_ext / 5.5
            )
    elif is_weighted == "powerlaw":
        for e in nx_graph_lcc.edges:
            nx_graph_lcc[e[0]][e[1]]["weight"] = plw.Power_Law(
                xmin=1, xmax=sup_ext, parameters=[alpha], discrete=True
            ).generate_random(1)

    if is_sparse:
        adjacency = nx.to_scipy_sparse_matrix(nx_graph_lcc)
    else:
        adjacency = nx.to_numpy_array(nx_graph_lcc)

    return adjacency
