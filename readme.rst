TODO
----

* write documentation
* finish ensemble generator
* publish on pip


DONE
----

* ~~add sparse matrices compatibility~~
* ~~add model memory, creama interoperability~~
* ~~add possibility of having multiple initial conditions or methods for CReAMa with dcm~~

FUTURE FEATURES
---------------

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

To build a development environment run:

.. code-block:: bash

    python3 -m venv venv 
    source venv/bin/activate 
    pip install -e '.[dev]'

For testing:

.. code-block:: bash

    pytest --cov

Credits
-------
This is a project by `Niccolò Vallarano <http://www.imtlucca.it/en/nicolo.vallarano/>`_ and `Emiliano Marchese <https://www.imtlucca.it/en/emiliano.marchese/>`_, under 
the supervision of `Tiziano Squartini <http://www.imtlucca.it/en/tiziano.squartini/>`_ and  `Mario Zanon <http://www.imtlucca.it/it/mario.zanon/>`_.

