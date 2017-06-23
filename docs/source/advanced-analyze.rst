Analyzing MCMC Results
======================

The ``rsdfit`` executable includes an ``analyze`` sub-command that can
run some basic analysis of the MCMC chains in a given directory, including
automatically generating nice plot results, removing burn-in steps, and
writing out best-fit parameter files.

The calling sequence is:

.. command-output:: rsdfit analyze -h

The user can specify a results directory as the only positional argument, or
one or more :class:`~pyRSD.rsdfit.results.EmceeResults` file names
as the positional arguments. In the case of a directory, all valid
``.npz`` files in that directory will be analyzed.

The steps performed by the ``analyze`` sub-command are:


1. The first thing the code does is compare the convergence of all
parameters in the result files (both free and constrained) using
the Gelman-Rubin criteria and prints out this convergence. The best results
are achieved when multiple, independent, results files are provided
on the command-line.

2. Next, the code removes automatically trims the chains of the burn-in steps,
removing iterations that are too far away from the maximum probability.
Alternatively, the user can specify the fraction of initial samples to
consider burnin via the ``-b, ---burnin`` flag.

3. After the burn-in steps are removed, a single MCMC chain is created,
and written to the ``info/combined_result.npz`` path. Additionally,
summary files about the best-fit parameters are saved to the ``info``
directory.

4. Several plots are generated, based on the options specified by the user
on the command line. These figures are saved to the ``plots`` directory.
The possible plots include figures of the 1D histograms and triangle
plots showing the 2D correlations between parameters.

The user can specify groupings of parameters to plot by specifying the
``analyze.to_plot_1d`` and ``analyze.to_plot_2d`` parameters in a file and
passing the name of that file via the ``---extra`` command line option.
For example, the file may include:

.. code-block:: bash

    analyze.to_plot_2d = {'biases': ['b1_cA', 'b1_cB', 'b1_sA', 'b1_sB'], \
                          'fractions' : ['fs', 'fsB', 'fcB'], \
                          'cosmo' : ['f', 'sigma8_z', 'fsigma8', 'alpha_par', 'alpha_perp', 'b1sigma8', 'alpha', 'epsilon'], \
                          'sigmas' : ['sigma_c', 'sigma_s', 'sigma_sA', 'sigma_sB'], \
                          'nuisance' : ['Nsat_mult', 'f1h_sBsB', 'f1h_cBs', 'gamma_b1sB', 'gamma_b1sA']}

In this case, 2D triangle plots comparing each of these parameter groupings
will be generated and saved in the ``plots`` directory.

.. note::

    See the documentation of :class:`~pyRSD.rsdfit.analysis.driver.AnalysisDriver`
    below for the accepted parameters that can be specified in the ``extra``
    parameter file passed to the ``rsdfit analyze`` command.

API
---

.. currentmodule:: pyRSD.rsdfit.analysis.driver

.. autoclass:: AnalysisDriver
  :members: __init__
