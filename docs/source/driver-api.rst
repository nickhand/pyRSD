API
===

The :class:`pyRSD.rsdfit.FittingDriver` class is responsible for running parameter fits. It
combines data and theory to run a Bayesian likelihood analysis.

The properties that describe the data and theory configuration are:

.. currentmodule:: pyRSD.rsdfit.FittingDriver

.. autosummary::
    Nb
    Np
    dof
    model
    results

The functions that evaluate the likelihood, its derivatives, and
associated statistics are:

.. autosummary::
    lnprob
    lnlike
    minus_lnlike
    grad_minus_lnlike
    chi2
    reduced_chi2
    fisher
    marginalized_errors
    run

The user can initialize a :class:`FittingDriver` object from a results
directory using

.. autosummary::
    from_directory

The best-fit parameters can be set and visualized using

.. autosummary::
    set_fit_results
    plot
    plot_residuals

.. currentmodule:: pyRSD.rsdfit

.. autoclass:: FittingDriver
  :members: Nb, Np, dof, model, results, lnprob, lnlike, minus_lnlike, grad_minus_lnlike, chi2, reduced_chi2, fisher, marginalized_errors, run, from_directory, set_fit_results, plot, plot_residuals
