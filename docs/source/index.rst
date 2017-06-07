pyRSD
===================================================

*Accurate predictions for the clustering of galaxies in redshift-space in Python*


pyRSD is a Python package for computing the theoretical predictions of
the redshift-space power spectrum of galaxies. The package also includes
functionality for fitting data measurements and finding the optimal model parameters,
using both MCMC and nonlinear optimization techniques.

provides a high-level, Python user interface for
interacting with the theoretical models and


Index
-----

**Getting Started**

* :doc:`install`
* :doc:`use-cases`
* :doc:`examples`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install.rst
   use-cases.rst
   examples.rst


**RSD**

* :doc:`cosmo`
* :doc:`hzpt`
* :doc:`galaxy-power`
* :doc:`qso-power`

.. toctree::
  :maxdepth: 1
  :hidden:
  :caption: RSD

  cosmo.rst
  hzpt.rst
  galaxy-power.rst
  qso-power.rst

**RSDFiT**

* :doc:`specifying-data`
* :doc:`specifying-theory`
* :doc:`covariance-matrix`
* :doc:`parameter-fits`
* :doc:`analyzing-results`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: RSDFit

   specifying-data.rst
   specifying-theory.rst
   covariance-matrix.rst
   parameter-fits.rst
   analyzing-results.rst
