.. currentmodule:: pyRSD.rsdfit.results

MCMC
====

.. ipython:: python
    :suppress:

    import os
    startdir = os.path.abspath('.')
    home = startdir.rsplit('docs' , 1)[0]
    os.chdir(home); os.chdir('docs/data')

The main result of running ``rsdfit mcmc ...`` is a ``*.npz`` file saved to the
output directory for each MCMC chain that was run. These results can be
loaded from file using the :class:`pyRSD.rsdfit.results.EmceeResults`
class.

The :class:`EmceeResults` stores the entire MCMC chain
for each free parameter and can return statistics based on this chain for
each parameter, i.e., the median, :math:`1\sigma` and :math:`2\sigma`
errors, etc. This object also computes the corresponding MCMC chain
for the constrained parameter values and can return information
for each constrained parameter.

The median parameter values, as well as the 68% and 95% confidence
intervals can be quickly displayed by printing the
:class:`EmceeResults` object. For example,

.. ipython:: python

    from pyRSD.rsdfit.results import EmceeResults

    results = EmceeResults.from_npz('mcmc_result.npz')

    # print out a summary of the parameters, with mean values and 68% and 95% intervals
    print(results)


The EmceeParameter Object
~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`EmceeResults` object provides
access to the parameters via a dictionary-like behavior. When accessing
parameters using the name as the key, a :class:`EmceeParameter` object
is returned. This object has several useful attributes holding information
about the parameter:

1. **flat_trace** : the flattened MCMC chain for this parameter
2. **median** : the median of the trace
3. **mean**: the average value of the trace
4. **one_sigma** : tuple of the lower and upper :math:`1\sigma` errors
5. **two_sigma** : tuple of the lower and upper :math:`2\sigma` errors
6. **three_sigma** : tuple of the lower and upper :math:`3\sigma` errors
7. **stderr** : the average of the upper and lower :math:`1\sigma` error

For example,

.. ipython:: python

      f = results['f']
      print(f.median, f.one_sigma, f.two_sigma)

      sigma8 = results['sigma8_z']
      print(sigma8.median, sigma8.one_sigma, sigma8.two_sigma)

      # access to constrained parameters too
      fs8 = results['fsigma8']
      print(fs8)

Specifying the Burn-in Period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user can specify a "burn-in" period for a given results object, by
changing the :attr:`burnin` attribute of the :class:`EmceeResults` object.
The value of this attribute specifies the number of steps to ignore, starting
from the beginning of the MCMC chain. The ignored steps are not included when
computing any statistics from the parameter traces.

The attributes of the :class:`EmceeParameter` object automatically take into
account the value of the burnin attribute. Thus, the user just needs to set
the :attr:`burnin` and the parameter values will automatically adjust
accordingly.

For example,

.. ipython:: python

    results.burnin = 0 # ignore 0 steps

    print(results['fsigma8'].median)

    results.burnin = 500 # ignore the first 500 steps

    print(results['fsigma8'].median) # slight change in value

Ideally, if the chain has converged for a given parameter, the user should
see little change to the parameter's value when adjusting the burnin period.

The Best-fit Values
~~~~~~~~~~~~~~~~~~~

The user can quickly access the best-fit parameter vector using the
:attr:`values` attribute, which returns the median value of each free
parameter. Similarly, the :attr:`constrained_values` attribute returns
the median value of each constrained parameter.

Alternatively, the user can access the parameter vector that maximizes
the log probability by accessing the :attr:`max_lnprob_values` and
:attr:`max_lnprob_constrained_values` attributes.

.. ipython:: python
    :suppress:

    import os
    os.chdir(startdir)
