![PyPI](https://img.shields.io/pypi/v/menet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/menet)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[comment]: # insert arxiv badge

MENet is a Maximum-Entropy toolbox for networks, released as a python3 module. 

MENet provides the user with a state of the art solver for a range variety of Maximum Entropy Networks models derived from the ERGM family.
From a given network, MENet is able to solve the desired model and to generate a number of randomized graphs from the original one: the so-called _graphs ensemble_.

MENet builds on the current literature on the matter, improving both in speed of convergence and in the scale of the feasible networks.

The models implemented in MENet are presented in a forthcoming [paper](arxiv).
If you use the module for your scientific research, consider citing us:

```
    bibtex snippet
```

## Currently Implemented Models
The main feature of MENet is to randomized a given network, following a variety of models from the ERGM family:

* Undirected 
    * **UBCM** *Undirected Binary Configuration* Model[[1]](#1)
    * **UECM** *Undirected Enhanced Configuration* Model[[1]](#1)
    * **CReMa** [[2]](#2)

* Directed
    * **DBCM** *Directed Binary Configuration Model* [[1]](#1)
    * **DECM** *Directed Binary Enhanced Model* [[1]](#1)
    * **CReMa** [[2]](#2)

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

Menet can be installed via pip. You can get it from your terminal:

```
    $ pip install menet
```

If you already install the package and wish to upgrade it,
you can simply type from your terminal:

```
    $ pip install menet --upgrade
```

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

Credits
-------

_authors_:

[Nicolò Vallarano](http://www.imtlucca.it/en/nicolo.vallarano/)(a.k.a. [nicoloval](https://github.com/nicoloval))

[Emiliano Marchese](https://www.imtlucca.it/en/emiliano.marchese/) (a.k.a. [EmilianoMarchese](https://github.com/EmilianoMarchese))

The module was developed under the supervision of [Tiziano Squartini](http://www.imtlucca.it/en/tiziano.squartini/) and  [Mario Zanon](http://www.imtlucca.it/it/mario.zanon/), at [IMT school of advanced studies Lucca](https://www.imtlucca.it/).

Nicolò Vallarano aknoledge funding from _insert stuff_ 
