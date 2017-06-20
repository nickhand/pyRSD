Overview
========

The default model parametrization is the one described in Section 4.4 of
`Hand et al. 2017 <https://arxiv.org/abs/1706.02362>`_. See this section
for a detailed discussion of the free and constrained parameters, as well
as the priors used during parameter fitting. There are 13 free parameters
that are varied during the fitting procedure.

The default set of parameters can be easily loaded from a
:class:`~pyRSD.rsd.GalaxySpectrum` object as

.. ipython:: python

    from pyRSD.rsd import GalaxySpectrum

    model = GalaxySpectrum()
    params = model.default_params()
    print(params)

The ``params`` variable here is a :class:`pyRSD.rsdfit.parameters.ParameterSet` object,
which is a dictionary of :class:`pyRSD.rsdfit.parameters.Parameter` objects.
The :class:`~pyRSD.rsdfit.parameters.Parameter` object not only stores the value
of the parameter (as the :attr:`value` attribute), but also stores
information about the prior and whether the parameter is freely varied or constrained.
For example, the satellite fraction ``fs`` can be accessed as

.. ipython:: python

    fsat = params['fs']
    print(fsat.value)

    # freely varying
    print(fsat.vary)

    fsat.value = 0.12  # change the value to 0.12

    # uniform prior assumed by default
    print(fsat.prior)

The Free Parameters
-------------------

The 13 free parameters by default are:

.. ipython:: python

    for par in params.free:
      print(par)

The Constrained Parameters
--------------------------

There are several constrained parameters, i.e., parameters whose values are
solely determined by other parameters, in the default configuration. Note
also that since these parameters are not freely varied, they do not require
priors.

.. ipython:: python

    for par in params.constrained:
      print(par)

The :class:`~pyRSD.rsdfit.parameters.ParameterSet` object handles constrained
parameters automatically. For example, in our default configuration, we have the
parameter ``fsigma8``, which is the product of the growth rate ``f`` and
the mass variance ``sigma8_z``. We can change the value of either ``f`` or
``sigma8_z`` and the value of ``fsigma8_z`` will reflect those changes. For example,

.. ipython:: python

    print(params['fsigma8'])
    print(params['f']*params['sigma8_z'])

    params['f'].value = 0.75

    print(params['fsigma8'])
    print(params['f']*params['sigma8_z'])
