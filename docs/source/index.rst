NEMtropy documentation
=====================================

NEMtropy is a Maximum-Entropy toolbox for networks, released as a
python3 module.

NEMtropy provides the user with a state of the art solver for a range
variety of Maximum Entropy Networks models derived from the ERGM family.
This module allows you to solve the desired model and generate a number
of randomized graphs from the original one: the so-called *graphs
ensemble*.

NEMtropy builds on the current literature on the matter, improving both
in speed of convergence and in the scale of the feasible networks. To
explore Maximum-Entropy modeling on networks, checkout `Maximum Entropy
Hub <https://meh.imtlucca.it/>`__.

The models implemented in NEMtropy are presented in a forthcoming
`paper <https://arxiv.org/abs/2101.12625>`__ on arXiv. If you use the
module for your scientific research, please consider citing us:

::

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


Currently Implemented Models
============================

The main feature of NEMtropy is (but not limited to) *network
randomization*. The specific kind of network to randomize and property
to preserve defines the model you need:

-  **UBCM** *Undirected Binary Configuration Model* `[1] <#1>`__
-  **UECM** *Undirected Enhanced Configuration Model* `[1] <#1>`__
-  **DBCM** *Directed Binary Configuration Model* `[1] <#1>`__
-  **DECM** *Directed Enhanced Configuration Model* `[1] <#1>`__
-  **CReMa** `[2] <#2>`__
-  **BiCM** *Bipartite Configuration Model* `[3] <#3>`__

The following table may helps you identify the model that fits your
needs in function of the type of network you are working with; for
in-depth discussion please see the references.

+----------------------+--------------------+-------------------+-------------------+
| [...]                | Undirected Graph   | Directed Graph    | Bipartite Graph   |
+======================+====================+===================+===================+
| **Binary Graph**     | *UBCM*             | *DBCM*            | *BiCM*            |
+----------------------+--------------------+-------------------+-------------------+
| **Weighted Graph**   | *UECM*, *CReMa*    | *DECM*, *CReMa*   | -                 |
+----------------------+--------------------+-------------------+-------------------+

The BiCM module is also available as `a standalone package <https://github.com/mat701/BiCM>`__, find its docs `here <https://bipartite-configuration-model.readthedocs.io/en/latest/>`__.

*References*

-  [1] Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
   "Unbiased sampling of network ensembles." New Journal of Physics 17.2
   (2015): 023052. https://arxiv.org/abs/1406.1197
-  [2] Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli. "A
   faster horse on a safer trail: generalized inference for the
   efficient reconstruction of weighted networks." New Journal of
   Physics 22.5 (2020): 053053. https://arxiv.org/abs/1811.09829
-  [3] Saracco, Fabio, Riccardo Di Clemente, Andrea Gabrielli, and Tiziano Squartini.
   "Randomizing bipartite networks: the case of the World Trade Web."
   Scientific reports 5, no. 1 (2015): 1-18. https://doi.org/10.1038/srep10595


Installation
============

NEMtropy can be installed via pip. You can get it from your terminal:

::

        $ pip install NEMtropy

If you already install the package and wish to upgrade it, you can
simply type from your terminal:

::

        $ pip install NEMtropy --upgrade

Dependencies
============

NEMtropy uses <code>numba</code> and <code>powerlaw</code> libraries.
They can be installed via pip by running in your terminal the following command:

::

        $ pip install numba
        $ pip install powerlaw

For python3.5 users the correct command is the following:

::

        $ pip install --prefer-binary numba

It avoids an error during the installation of llvmlite due to the
absence of its wheel in python3.5.


Simple Example
==============

As an example we solve the UBCM for zachary karate club network.

::

        import networkx as nx
        from NEMtropy import UndirectedGraph

        G = nx.karate_club_graph()
        adj_kar = nx.to_numpy_array(G)
        graph = UndirectedGraph(adj_kar)

        graph.solve_tool(model="cm_exp",
                     method="newton",
                     initial_guess="random")

Given the UBCM model, we can generate ten random copies of zachary's
karate club.

::

        graph.ensemble_sampler(10, cpu_n=2, output_dir="sample/")

These copies are saved as an edgelist, each edgelist can be converted to
an adjacency matrix by running the NEMtropy build graph function.

::

        from NEMtropy.network_functions import build_graph_from_edgelist

        edgelist_ens = np.loadtxt("sample/0.txt")
        ens_adj = build_graph_from_edgelist(edgelist = edgelist_ens,
                                        is_directed = False,
                                        is_sparse = False,
                                        is_weighted = False)

These collection of random adjacency matrices can be used as a null
model: it is enough to compute the expected value of the selected
network feature on the ensemble of matrices and to compare it with its
original value.

To learn more, please read the two ipython notebooks in the examples
directory: one is a study case on a `directed
graph <https://github.com/nicoloval/NEMtropy/blob/master/examples/Directed%20Graphs.ipynb>`__,
while the other is on an `undirected
graph <https://github.com/nicoloval/NEMtropy/blob/master/examples/Undirected%20Graphs.ipynb>`__.

Development
===========

Please work on a feature branch and create a pull request to the
development branch. If necessary to merge manually do so without fast
forward:

::

        $ git merge --no-ff myfeature

To build a development environment run:

::

        $ python3 -m venv venv 
        $ source venv/bin/activate 
        $ pip install -e '.[dev]'

Testing
=======

If you want to test the package integrity, you can run the following
bash command from the tests directory:

::

        $ bash run_all.sh

**P.S.** *at the moment there may be some problems with the DECM solver
functions*

Guide
^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   NEMtropy
   license
   contacts
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
