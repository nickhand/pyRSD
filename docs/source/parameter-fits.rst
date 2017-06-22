.. _parameter-fits:

Running the Fits
================

Once the data, covariance matrix, and theory parameters have be set, the
user can run a likelihood analysis to find the best-fitting
theory parameters. This can be done using the
`MCMC technique <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ or
nonlinear optimization using the
`LBFGS algorithm <https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm>`_.
In this section, we describe how to run both algorithms with
the ``rsdfit`` executable, as well as the configuration options for both algorithms.

.. toctree::
   :maxdepth: 1

   driver-overview.rst
   driver-mcmc.rst
   driver-nlopt.rst
   driver-api.rst
