![PyPI](https://img.shields.io/pypi/v/nemtropy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nemtropy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Scientific Reports](https://media.springernature.com/full/nature-cms/uploads/product/srep/header-d3c533c187c710c1bedbd8e293815d5f.svg)](https://doi.org/10.1038/s41598-021-93830-4)

NEMtropy: Network Entropy Maximization, a Toolbox Running On PYthon
-------------------------------------------------------------------

NEMtropy is a Maximum-Entropy toolbox for networks, released as a python3 module. 

NEMtropy provides the user with a state of the art solver for a range variety of Maximum Entropy Networks models derived from the ERGM family.
This module allows you to solve the desired model and generate a number of randomized graphs from the original one: the so-called _graphs ensemble_.

NEMtropy builds on the current literature on the matter, improving both in speed of convergence and in the scale of the feasible networks.
To explore Maximum-Entropy modeling on networks, checkout [Maximum Entropy Hub](https://meh.imtlucca.it/).

If you use the module for your scientific research, please consider citing us:

```
@article{vallarano2021fast,
  title={Fast and scalable likelihood maximization for exponential random graph models with local constraints},
  author={Vallarano, Nicol{\`o} and Bruno, Matteo and Marchese, Emiliano and Trapani, Giuseppe and Saracco, Fabio and Cimini, Giulio and Zanon, Mario and Squartini, Tiziano},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={15227},
  year={2021},
  publisher={Nature Publishing Group UK London}
}

```

#### Table Of Contents
- [Currently Implemented Models](#currently-implemented-models)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [A Simple Example](#simple-example)
- [Documentation](#documentation)
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
* **BiCM** *Bipartite Configuration Model* [[3]](#3)
* **BiWCM** *Bipartite Weighted Configuration Model* [[4]](#4)

The following table may helps you identify the model that fits your needs in function of the type of network you are working with;
for in-depth discussion please see the references.

[...] | Undirected Graph | Directed Graph | Bipartite Graph
----- | ---------------- | -------------- | --------------
**Binary Graph** | *UBCM* | *DBCM* | *BiCM*
**Weighted Graph** | *UECM*, *CReMa*  | *DECM*, *CReMa* | *BiWCM*

The BiCM module is now (NEMtropy>=3.0.0) imported and it is mantained as a [a standalone package](https://github.com/mat701/BiCM), find its docs [here](https://bipartite-configuration-model.readthedocs.io/en/latest/). 

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
* <a id="3">[3]</a>
    Saracco, Fabio, Riccardo Di Clemente, Andrea Gabrielli, and Tiziano Squartini.
	"Randomizing bipartite networks: the case of the World Trade Web." 
	Scientific reports 5, no. 1 (2015): 1-18.
    https://doi.org/10.1038/srep10595
* <a id="4">[4]</a>
    Bruno, Matteo, Dario Mazzilli, Aurelio Patelli, Tiziano Squartini, and Fabio Saracco.
        "Inferring comparative advantage via entropy maximization."
        Journal of Physics: Complexity, Volume 4, Number 4 (2023).
    https://doi.org/10.1088/2632-072X/ad1411

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

NEMtropy uses <code>numba</code>, <code>powerlaw</code>, <code>tqdm</code>, <code>scipy</code>, <code>networkx</code>, <code>bicm</code> libraries. They can be installed via pip by running in your terminal the following command:

```
    $ pip install numba
    $ pip install powerlaw
    $ pip install networkx
    $ pip install scipy
    $ pip install tqdm
    $ pip install bicm
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
    import numpy as np
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
    from NEMtropy.network_functions import build_adjacency_from_edgelist

    edgelist_ens = np.loadtxt("sample/0.txt")
    ens_adj = build_adjacency_from_edgelist(edgelist = edgelist_ens,
                                            is_directed = False,
                                            is_sparse = False,
                                            is_weighted = False)
```

These collections of random adjacency matrices can be used as a null model:
it is enough to compute the expected value of the selected network feature 
on the ensemble of matrices and to compare it with its original value.

To learn more, please read the ipython notebooks in the examples directory:
- one is a study case on a [directed graph](https://github.com/nicoloval/NEMtropy/blob/master/examples/Directed%20Graphs.ipynb), 
- while the other is on an [undirected graph](https://github.com/nicoloval/NEMtropy/blob/master/examples/Undirected%20Graphs.ipynb).
- There is also one on motifs!!! [motifs](https://github.com/nicoloval/NEMtropy/blob/c4283e6b939274f532278cd84841656b20d819a4/examples/Motifs.ipynb).


Documentation
-------------

You can find complete documentation about NEMtropy library in [docs](https://nemtropy.readthedocs.io/en/master/index.html).

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast-forward:

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
bash command from the tests' directory:

```
    $ bash run_all.sh
```

__P.S.__ _at the moment there may be some problems with the DECM solver functions_

Credits
-------

_Authors_:

[Nicolò Vallarano](https://www.ifi.uzh.ch/en/bdlt/Team/Postdocs/Dr.-Vallarano-Nicol%C3%B2.html) (a.k.a. [nicoloval](https://github.com/nicoloval))

[Emiliano Marchese](https://www.imtlucca.it/en/emiliano.marchese/) (a.k.a. [EmilianoMarchese](https://github.com/EmilianoMarchese))

[Matteo Bruno](https://csl.sony.it/member/matteo-bruno/) (BiCM) (a.k.a. [mat701](https://github.com/mat701))

_Acknowledgements:_

The module was developed under the supervision of [Tiziano Squartini](http://www.imtlucca.it/en/tiziano.squartini/), [Fabio Saracco](http://www.imtlucca.it/en/fabio.saracco/), [Mario Zanon](https://mariozanon.wordpress.com/), and [Giulio Cimini](https://www.fisica.uniroma2.it/elenco-telefonico/ciminigi/).
It was developed at [IMT school of advanced studies Lucca](https://www.imtlucca.it/), and financed by the research project Optimized Reconstruction of Complex networkS - ORCS.
