.. currentmodule:: pyRSD.rsdfit.results

API
===


MCMC Results
------------

The best-fit parameter vector can be accessed from the MCMC chain using
the following functions of the :class:`~pyRSD.rsdfit.results.EmceeResults`
object:

.. autosummary::
  EmceeResults.values
  EmceeResults.constrained_values
  EmceeResults.max_lnprob_values
  EmceeResults.max_lnprob_constrained_values
  EmceeResults.peak_values
  EmceeResults.peak_constrained_values

The results object can be saved and loaded from file using:

.. autosummary::
  EmceeResults.to_npz
  EmceeResults.from_npz

Correlations between parameters can be analyzed using:

.. autosummary::
  EmceeResults.sorted_1d_corrs
  EmceeResults.corr

The results can be visualized with the following function, which take
advantage of the :mod:`seaborn` Python plotting module:

.. autosummary::
  EmceeResults.jointplot_2d
  EmceeResults.kdeplot_2d
  EmceeResults.plot_correlation
  EmceeResults.plot_timeline
  EmceeResults.plot_triangle

.. autoclass:: EmceeResults
  :members:

.. autoclass:: EmceeParameter
  :members:

NLOPT Results
-------------

.. autoclass:: LBFGSResults
  :members:
