pyRSD
=====

*Accurate predictions for the clustering of galaxies in redshift-space in Python*

pyRSD is a Python package for computing the theoretical predictions of
the redshift-space power spectrum of galaxies. The package also includes
functionality for performing Bayesian parameter estimation using
the MCMC sampling technique or Maximum a posteriori estimation using
the LBFGS algorithm to perform the nonlinear optimization.

.. note::

  The theoretical models used in this paper are described in more detail
  in `Hand et al. 2017 <https://arxiv.org/abs/1706.02362>`_. Please cite
  this work if you use this package in your research.


Index
-----

**Getting Started**

* :doc:`install`
* :doc:`overview`
* :doc:`quickstart`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install.rst
   overview.rst
   quickstart.rst

**RSD**

The **RSD** module deals with computing the theoretical power spectrum predictions,
given an input cosmology specified by the user.

* :doc:`cosmo`
* :doc:`interfacing-class`
* :doc:`hzpt`
* :doc:`galaxy-power`
* :doc:`qso-power`

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: RSD

  cosmo.rst
  interfacing-class.rst
  hzpt.rst
  galaxy-power.rst
  qso-power.rst

**RSDFit**

The **RSDFit** module deals with running parameter estimation using the
power spectrum models available in this package and data input by the user.

* :doc:`rsdfit-getting-started`
* :doc:`specifying-data`
* :doc:`specifying-theory`
* :doc:`parameter-fits`
* :doc:`exploring-results`
* :doc:`rsdfit-advanced`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: RSDFit

   rsdfit-getting-started.rst
   specifying-data.rst
   specifying-theory.rst
   parameter-fits.rst
   exploring-results.rst
   rsdfit-advanced.rst

Get in touch
------------

Report bugs, suggest feature ideas, or view the source code `on GitHub <http://github.com/nickhand/pyRSD>`_.
