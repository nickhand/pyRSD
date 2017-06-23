Nonlinear Optimization
======================

.. currentmodule:: pyRSD.rsdfit.results

.. ipython:: python
    :suppress:

    import os
    import numpy
    startdir = os.path.abspath('.')
    home = startdir.rsplit('docs' , 1)[0]
    os.chdir(home); os.chdir('docs/data')

The main result of running ``rsdfit nlopt ...`` is a ``*.npz`` file saved to the
output directory for each MCMC chain that was run. These results can be
loaded from file using the :class:`pyRSD.rsdfit.results.LBFGSResults`
class.

The :class:`LBFGSResults` stores the best-fit values for both the
free parameters and the constrained parameters, as computed from the final
iteration of the LBFGS algorithm.

The best-fit parameter values and the corresponding minimum :math:`\chi^2 value`
can be quickly displayed by printing the :class:`LBFGSResults` object.
For example,

.. ipython:: python

    from pyRSD.rsdfit.results import LBFGSResults

    results = LBFGSResults.from_npz('nlopt_result.npz')

    # print out a summary of the best-fit parameters
    print(results)

The minimum :math:`\chi^2` value can be accessed from the :attr:`min_chi2`
attribute of the :class:`LBFGSResults` object.

Parameter Access
----------------

The :class:`LBFGSResults` object provides
access to the parameters via a dictionary-like behavior. When accessing
parameters using the name as the key, the best-fit value of that
parameter is returned.

For example,

.. ipython:: python

      # growth rate
      f = results['f']
      print(f)

      # power spectrum normalization
      sigma8 = results['sigma8_z']
      print(sigma8)

      # this is the product of f and sigma8_z
      fs8 = results['fsigma8']
      print(fs8)
      print(numpy.isclose(fs8, f*sigma8))

The Best-fit Values
-------------------

The user can quickly access the best-fit parameter vector using the
:attr:`min_chi2_values` attribute, which returns the value of each free
parameter at the minimum :math:`\chi^2` of the fit.
Similarly, the :attr:`min_chi2_constrained_values` attribute returns the
corresponding value of each constrained parameter at the minimum :math:`\chi^2`.

.. ipython:: python
    :suppress:

    import os
    os.chdir(startdir)
