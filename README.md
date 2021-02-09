![PyPI](https://img.shields.io/pypi/v/nemtropy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nemtropy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![ArXiv](https://img.shields.io/badge/ArXiv-2101.12625-red)](https://arxiv.org/abs/2101.12625)

NEMtropy: Network Entropy Maximization, a Toolbox Running On PYthon
-------------------------------------------------------------------

NEMtropy is a Maximum-Entropy toolbox for networks, released as a python3 module. 

NEMtropy provides the user with a state of the art solver for a range variety of Maximum Entropy Networks models derived from the ERGM family.
This module allows you to solve the desired model and generate a number of randomized graphs from the original one: the so-called _graphs ensemble_.

NEMtropy builds on the current literature on the matter, improving both in speed of convergence and in the scale of the feasible networks.
To explore Maximum-Entropy modeling on networks, checkout [Maximum Entropy Hub](https://meh.imtlucca.it/).

The models implemented in NEMtropy are presented in a forthcoming [paper](https://arxiv.org/abs/2101.12625) on arXiv.
If you use the module for your scientific research, please consider citing us:

```
    @misc{vallarano2021fast,
          title={Fast and scalable likelihood maximization for Exponential Random Graph Models}, 
          author={Nicolò Vallarano and Matteo Bruno and Emiliano Marchese and Giuseppe Trapani and Fabio Saracco and Tiziano Squartini and Giulio Cimini and Mario Zanon},
          year={2021},
          eprint={2101.12625},
          archivePrefix={arXiv},
          primaryClass={physics.data-an}
    }
```

#### Table Of Contents
- [Currently Implemented Models](#currently-implemented-models)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [A Simple Example](#simple-example)
- [Development](#development)
- [Testing](#testing)
- [Credits](#credits)

## Currently Implemented Models
The main feature of NEMtropy is (but not limited to) *network randomization*. 
The specific kind of network to randomize and property to preserve defines the model you need:

* **UBCM** *Undirected Binary Configuration Model* [[1]](#1)
* **UECM** *Undirected Enhanced Configuration Model* [[1]](#1)
* **DBCM** *Directed Binary Configuration Model* [[1]](#1)
* **DECM** *Directed Enhanced Configuration Model* [[1]](#1)
* **CReMa** [[2]](#2)

The following table may helps you identify the model that fits your needs in function of the type of network you are working with;
for in-depth discussion please see the references.

[...] | Undirected Graph | Directed Graph
----- | ---------------- | -------------- 
**Binary Graph** | *UBCM* | *DBCM* 
**Weighted Graph** | *UECM*, *CReMa*  | *DECM*, *CReMa*

_References_

* <a id="1">[1]</a>
    Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
    "Unbiased sampling of network ensembles."
    New Journal of Physics 17.2 (2015): 023052.
    https://arxiv.org/abs/1406.1197
* <a id="2">[2]</a>
    Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli.
    "A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks."
    New Journal of Physics 22.5 (2020): 053053.
    https://arxiv.org/abs/1811.09829


Installation
------------

NEMtropy can be installed via pip. You can get it from your terminal:

```
    $ pip install NEMtropy
```

If you already install the package and wish to upgrade it,
you can simply type from your terminal:

```
    $ pip install NEMtropy --upgrade
```

Dependencies
------------

NEMtropy uses <code>numba</code> and <code>powerlaw</code> libraries. They can be installed via pip by running in your terminal the following command:

```
    $ pip install numba
    $ pip install powerlaw
```

For <code>python3.5</code> users the correct command is the following:

```
    $ pip install --prefer-binary numba
```

It avoids an error during the installation of <code>llvmlite</code> due to 
the absence of its wheel in <code>python3.5</code>.

Simple Example
--------------
As an example we solve the UBCM for zachary karate club network.

```
    import networkx as nx
    from NEMtropy import UndirectedGraph

    G = nx.karate_club_graph()
    adj_kar = nx.to_numpy_array(G)
    graph = UndirectedGraph(adj_kar)

    graph.solve_tool(model="cm_exp",
                 method="newton",
                 initial_guess="random")
```

Given the UBCM model, we can generate ten random copies of zachary's karate club.

```
    graph.ensemble_sampler(10, cpu_n=2, output_dir="sample/")
```

These copies are saved as an edgelist, each edgelist can be converted to an
adjacency matrix by running the NEMtropy build graph function.

```
    from NEMtropy.network_functions import build_graph_from_edgelist

    edgelist_ens = np.loadtxt("sample/0.txt")
    ens_adj = build_graph_from_edgelist(edgelist = edgelist_ens,
                                    is_directed = False,
                                    is_sparse = False,
                                    is_weighted = False)
```

These collection of random adjacency matrices can be used as a null model:
it is enough to compute the expected value of the selected network feature 
on the ensemble of matrices and to compare it with its original value.

To learn more, please read the two ipython notebooks in the examples directory:
one is a study case on a [directed graph](https://github.com/nicoloval/NEMtropy/blob/master/examples/Directed%20Graphs.ipynb), 
while the other is on an [undirected graph](https://github.com/nicoloval/NEMtropy/blob/master/examples/Undirected%20Graphs.ipynb).

You can find complete documentation about NEMtropy library in [docs](https://nemtropy.readthedocs.io/en/master/index.html).

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

```
    $ git merge --no-ff myfeature
```

To build a development environment run:

```
    $ python3 -m venv venv 
    $ source venv/bin/activate 
    $ pip install -e '.[dev]'
```

Testing
-------
If you want to test the package integrity, you can run the following 
bash command from the tests directory:

```
    $ bash run_all.sh
```

__P.S.__ _at the moment there may be some problems with the DECM solver functions_

Credits
-------

_Authors_:

[Nicolò Vallarano](http://www.imtlucca.it/en/nicolo.vallarano/)(a.k.a. [nicoloval](https://github.com/nicoloval))

[Emiliano Marchese](https://www.imtlucca.it/en/emiliano.marchese/) (a.k.a. [EmilianoMarchese](https://github.com/EmilianoMarchese))

_Acknowledgements:_

The module was developed under the supervision of [Tiziano Squartini](http://www.imtlucca.it/en/tiziano.squartini/), [Mario Zanon](http://www.imtlucca.it/it/mario.zanon/), and [Giulio Cimini](https://www.fisica.uniroma2.it/elenco-telefonico/ciminigi/).
It was developed at [IMT school of advanced studies Lucca](https://www.imtlucca.it/), and financed by the research project Optimized Reconstruction of Complex networkS - ORCS.
