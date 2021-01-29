NEMtropy documentation
=====================================

NEMtropy is a Maximum-Entropy toolbox for networks, released as a python3 module. 

NEMtropy provides the user with a state of the art solver for a range variety of Maximum Entropy Networks models derived from the ERGM family.
This module allows you to solve the desired model and generate a number of randomized graphs from the original one: the so-called *graphs ensemble*.

NEMtropy builds on the current literature on the matter, improving both in speed of convergence and in the scale of the feasible networks.
To explore Maximum-Entropy modeling on networks, checkout [Maximum Entropy Hub](https://meh.imtlucca.it/).


Basic functionalities
=====================================

To install:

.. code-block:: python
    
    pip install NEMtropy

To import the module:

.. code-block:: python
    
    import NEMtropy


Dependencies
============

This package has been developed for Python 3.5 but it should be compatible with other versions. It works on numpy arrays and it uses numba and multiprocessing to speed up the computation. Feel free to send an inquiry (and please do it!) if you find any incompatibility.

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
* :ref:`modindex'
* :ref:`search`
