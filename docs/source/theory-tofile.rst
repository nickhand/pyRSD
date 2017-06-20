Writing to a Parameter File
===========================

The desired theoretical model parameterization must be written out to
the parameter file that will be passed to the ``rsdfit`` executable. The
easiest way to do this is to use the
:func:`~pyRSD.rsdfit.theory.parameters.ParameterSet.to_file` function
of the default parameter object.

The recommended workflow to configure the theory is:

1. Generate the default parameter set via the :func:`GalaxySpectrum.default_params`
function.

2. Make the desired changes to the parametrization, i.e., changing priors, etc

3. Write to an exisiting file using the :func:`to_file` function.

For example, assuming we have an existing parameter file entitled ``params.dat``,
we can write out our parameters as

.. ipython:: python

    from pyRSD.rsd import GalaxySpectrum

    # get the default parameters from an existing model
    model = GalaxySpectrum()
    params = model.default_params()

    # make any desired changes
    # ....

    # write out to file
    params.to_file('params.dat', mode='a') # append to this file

This will write out to the file each :class:`~pyRSD.rsdfit.theory.parameters.Parameter`
in the parameter set as a dictionary. Now, the ``params.dat`` looks like

.. literalinclude:: ../params.dat


.. ipython:: python
    :suppress:

    import os
    os.remove('params.dat')
