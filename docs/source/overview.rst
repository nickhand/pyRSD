Overview
=========

pyRSD provides functionality for computing complex, theoretical models
of the galaxy power spectrum in redshift space. The package has two
main modules:

1. :mod:`pyRSD.rsd`

  The module responsible for evaluating theoretical power spectra and related
  quantities for an input cosmology specified by the user.

2. :mod:`pyRSD.rsdfit`

  The module responsible for performing parameter estimation; it uses
  the theory models in :mod:`pyRSD.rsd` and finds the best-fit parameters
  describing an input data set.


The pyRSD.rsd module
~~~~~~~~~~~~~~~~~~~~~

This module provides the ability to compute several theoretical power
spectrum quantities. These include

1. :class:`pyRSD.rsd.GalaxySpectrum`

  The galaxy power spectrum in redshift space. See :ref:`this section <galaxy-power>` for more details.

2. :class:`pyRSD.rsd.QuasarSpectrum`

  The quasar power spectrum in redshift space. See :ref:`this section <qso-power>` for more details

3. :mod:`pyRSD.rsd.hzpt`

  A module for computing dark matter power spectra using Halo Zel'dovich Perturbation Theory.
  See :ref:`this section <hzpt>` for more details.

4. :mod:`pyRSD.pygcl`

  A module for interfacing with the CLASS Boltzmann code and computing various
  clustering quantities using the CLASS transfer function. This is a swig-wrapped
  C++ module that computes most of the perturbation theory and other numerically
  intensive calculations on which the pyRSD models are based. See
  :ref:`this section <pygcl>` for more details.


The pyRSD.rsdfit module
~~~~~~~~~~~~~~~~~~~~~~~

This module handles parameter estimation, fitting the theoretical models
provided in the :mod:`pyRSD.rsd` module to data provided by the user. There
are two ways parameter estimation can be performed:

1. Monte Carlo Markov Chain (MCMC)

  The full posterior distribution of the model parameters can be found using
  the :mod:`emcee` Python MCMC package.

2. Nonlinear optimization via the LBFGS algorithm

  The best-fit parameters can be found by maximizing the likelihood distribution,
  which is performed using the well-known LBFGS algorithm.
