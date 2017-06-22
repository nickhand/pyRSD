.. currentmodule:: pyRSD.rsdfit

Overview
========

The main class that is responsible for handling the data statistics, as
well as the covariance matrix used during parameter estimation, is
the :class:`pyRSD.rsdfit.FittingDriver` class. This class combines data and
theory to run a Bayesian likelihood analysis.

Information about the parameters that are needed to initialize
the :class:`~pyRSD.rsdfit.FittingDriver` class can be found by using
the :func:`FittingDriver.help` function,

.. ipython:: python

    from pyRSD.rsdfit import FittingDriver

    # print out the help message for the parameters needed
    FittingDriver.help()

These parameters should be specified in the parameter file that is passed
to the ``rsdfit`` executable and the names of the parameters should be
prefixed with the ``driver.`` prefix. In our example parameter file
discussed previously, we the following parameters

.. literalinclude:: ../../pyRSD/data/examples/params.dat
    :lines: 1-14
    :encoding: latin-1


These parameters allow the user to specify which type of data is being
fit, either "galaxy" or "quasar" and to pass options to the solver being used,
either the :mod:`emcee` MCMC solver or the NLOPT
solver. We will detail the MCMC solver (:ref:`mcmc-solver`) and the
LBFGS solver (:ref:`nlopt-solver`) in the next sections.
