Install pyRSD
=============

pyRSD can be installed using the ``conda`` utility or by installing from
source.

Conda
-----

The easiest installation method uses the ``conda`` utility, as part
of the `Anaconda <https://www.continuum.io/downloads>`_ package
manager. We have pre-built binaries available that are compatible with
Linux and macOS platforms. The package can be installed via::

  conda install -c nickhand -c astropy pyrsd

The package is available for Python versions 2.7, 3.5, and 3.6.

Install From Source
-------------------

pyRSD can also be installed directly from source::

  # clone the github source
  git clone https://github.com/nickhand/pyRSD.git
  cd pyRSD

  # run the install
  pip install .

Test
----

Test whether or not the installation succeeded by importing the module
in IPython:

.. code-block:: python

  import pyRSD
