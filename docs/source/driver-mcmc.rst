.. _mcmc-solver:

MCMC
====

The pyRSD package wraps the :mod:`emcee` package to provide functionality
for running MCMC chains to sample the posterior distributions of the
model parameters.


Command-line Options
~~~~~~~~~~~~~~~~~~~~

MCMC chains are run by passing the ``mcmc`` sub-command
to the ``rsdfit`` executable. The calling sequence for the ``mcmc`` command is

.. command-output:: rsdfit mcmc -h

The main options are the parameter file, passed by the ``-p`` option,
the directory to save results, passed by the ``-o`` option, and the
name of a model to load, passed by the ``-m`` file. In addition, there
are some MCMC-specific options:

1. **-w, ---walkers**

    The number of :mod:`emcee` walkers to use. These walkers are
    responsible for exploring the relevant parameter space simulataneously.

    .. note::

        The number of walkers must be at least twice the number of model parameters,
        and generally, the more walkers used the better. However, a trade-off exists
        since more walkers can become computationally infeasible when model
        evaluation is slow. In the case of the default model with 13 parameters,
        we recommend using 30-40 walkers.

2. **-i, ---iterations**

    The number of iterations to run in the MCMC chain.

    .. note::

        Generally, the models in the pyRSD package require >1000 iterations to
        achieve convergence, depending on how the MCMC sampler is initialized.

3. **--nchains**

    If the ``rsdfit`` is executed in parallel with multiple processes using
    MPI, it is possible to run multiple chains with MCMC concurrently with
    this option. This is ideal for testing the convergence of the sampler, as the chains
    will be compared statistically to determine if they have converged to
    a similar point in parameter space.

Initializing the MCMC Chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method used to initialize either the MCMC chains can be configured
using the ``driver.init_from`` parameter. The allowed values of this parameter
when using the MCMC solver are:

1. **fiducial** :

    Initialize the parameters in a small ball around the fiducial parameter
    values, specified in the parameter file via the ``fiducial`` keyword
    for each free parameter

2. **prior** :

    Initialize the free parameters by drawing a value from the prior probability
    specified for each parameter, which will be either a uniform or normal
    distribution

3. **result** :

    Initialize the free parameters in a small ball around the best-fit parameters
    loaded from a previous result. In this case, the ``driver.start_from``
    parameter should give the name of a ``.npz``, which
    can be loaded into either a :class:`pyRSD.rsdfit.results.LBFGSResults`
    or :class:`pyRSD.rsdfit.results.EmceeResults` object.

Testing for Convergence
~~~~~~~~~~~~~~~~~~~~~~~

The convergence of the MCMC chain being run will be tested if the
``driver.test_convergence`` parameter is set to ``True``. This test is performed
using the Gelman and Rubin diagnostic, and the tolerance
level for the test is specified via the ``driver.epsilon`` parameter.
The convergence for MCMC chains is performed as


1. Remove the first half of the current chains
2. Calculate the within chain and between chain variances
3. Estimate the variance from the within chain and between chain variance
4. Calculate the potential scale reduction parameter and compare to ``epsilon``

.. note::

    Convergence testing is disabled for MCMC chains by default, as it is often
    easier to run a pre-defined number of iterations (passed by the ``-i`` flag),
    which will be specific to the user's problem at hand.

Recommended Practices
~~~~~~~~~~~~~~~~~~~~~

The :mod:`emcee` package recommends initializing its MCMC sampler in a small
ball around what the user believes to be the area in parameter space close
to the maximum probability. As such, we recommend either of the following
initialization methods for best results:

1. Run the NLOPT solver, starting from a fiducial set of values, and then
inititialize the MCMC solver with the result of that NLOPT run.

2. Initialize the MCMC solver with values drawn from the parameters' prior
distribution and run a set of burn-in iterations (typically ~1000 or so).
Then, initialize a second MCMC chain from the best-fit result of that burn-in period.

For more recommended practices regarding the :mod:`emcee` package, please
see the FAQ on the emcee documentation `here <http://dan.iel.fm/emcee/current/user/faq/>`_.

Finally, it is also useful for convergence reasons to run multiple chains at once
in parallel. Often running two chains, independently initialized, with half the number of iterations
is useful to asses if the chains have truly found the best-fit parameters.
