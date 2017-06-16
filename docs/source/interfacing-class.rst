.. _pygcl:

Interfacing with CLASS
======================

The main power spectrum calculations performed
in pyRSD require the linear matter transfer function. We provide an interface
to the `CLASS <https://class-code.net>`_ CMB Boltzmann code via the
:class:`pyRSD.pygcl` module.

The :mod:`pyRSD.pygcl` module is the **General Cosmology Library**, a
swig-generated Python wrapper of a C++ library that provides useful
cosmological functionality, including an interface to the CLASS code.


.. toctree::
   :maxdepth: 1

   pygcl-overview.rst
   pygcl-api.rst
