import numpy as np
import multiprocessing as mp


def ensemble_sampler_cm_graph(outfile_name, x, cpu_n=2, seed=None):
    # produce and write a single undirected binary graph
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(x))

    # put together inputs for pool 
    # iter_ = itertools.product(zip(inds,x), zip(inds,x))
    # print(list(zip(inds, x)))
    iter_ = iter(((i, xi),(j, xj), np.random.randint(0,1000000)) for i,xi in zip(inds,x) for j,xj in zip(inds,x) if i<j) 

    # debug
    """
    s=0
    for c in iter_:
        i=c[0][0]
        j=c[1][0]
        print(i,j)
        s += 1
    print(s)
    """

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_cm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # debug
    # print(edges_list)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {}\n".format(str(i),str(j)) for (i,j) in edges_list
                )
            )

    return outfile_name 
            

def ensemble_sampler_ecm_graph(outfile_name, x, y, cpu_n=2, seed=None):
    # produce and write a single undirected weighted graph
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(x))

    # put together inputs for pool 
    # iter_ = itertools.product(zip(inds,x), zip(inds,x))
    # print(list(zip(inds, x)))
    iter_ = iter(((i, xi,yi),(j, xj,yj), np.random.randint(0,1000000)) for i,xi,yi in zip(inds,x,y) for j,xj,yj in zip(inds,x,y) if i<j) 

    # debug
    """
    s=0
    for c in iter_:
        i=c[0][0]
        j=c[1][0]
        print(i,j)
        s += 1
    print(s)
    """

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_ecm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # debug
    # print(edges_list)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {} {}\n".format(str(i), str(j), str(w)) for (i,j,w) in edges_list
                )
            )

    return outfile_name 
 

def ensemble_sampler_dcm_graph(outfile_name, x, y, cpu_n=2, seed=None):
    # produce and write a single directed binary graph
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(x))

    # put together inputs for pool 
    iter_ = iter(((i, xi),(j, yj), np.random.randint(0,1000000)) for i,xi in zip(inds, x) for j,yj in zip(inds, y) if i != j) 

    # debug
    """
    s=0
    for c in iter_:
        i=c[0][0]
        j=c[1][0]
        print(i,j)
        s += 1
    print(s)
    """

    # compute existing edges
    with mp.Pool(processes=cpu_n) as pool:
        edges_list = pool.starmap(is_a_link_dcm, iter_)

    # removing None
    edges_list[:] = (value for value in edges_list if value is not None)

    # debug
    # print(edges_list)

    # edgelist writing
    with open(outfile_name, "w") as outfile:
        outfile.write(
            "".join(
                "{} {}\n".format(str(i), str(j)) for (i,j) in edges_list
                )
            )

    return outfile_name 
 

def is_a_link_cm(args_1, args_2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    (i, xi) = args_1
    (j, xj) = args_2
    p = np.random.random()
    xij = xi*xj
    p_ensemble =  xij/(1 + xij)
    if p < p_ensemble:
        return (i, j)
    else:
        return None


def is_a_link_ecm(args_1, args_2, seed=None):
    # q-ensemble source: "Unbiased sampling of network ensembles", WUNs
    if seed is not None:
        np.random.seed(seed)
    (i, xi, yi) = args_1
    (j, xj, yj) = args_2
    p = np.random.random()
    xij = xi*xj
    yij = yi*yj
    p_ensemble =  xij*yij/(1 - yij + xij*yij)
    if p < p_ensemble:
        q_ensemble = yij
        w = np.random.geometric(1-q_ensemble)
        return (i, j, w)
    else:
        return None


def is_a_link_dcm(args_1, args_2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    (i, xi) = args_1
    (j, yj) = args_2
    p = np.random.random()
    tmp = xi*yj
    p_ensemble =  tmp/(1 + tmp)
    if p < p_ensemble:
        return (i, j)
    else:
        return None


