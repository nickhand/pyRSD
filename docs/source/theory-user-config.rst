.. currentmodule:: pyRSD.rsd

Configuring the Theory
=======================

Even if the user chooses to use the default theory parameters returned
by :func:`GalaxySpectrum.default_params`, there are several areas where the user
should customize the parameters for the specific data set being fit. We will
described these customizations in this section.

Changing the Parametrization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to make changes to the model parametrization is by
adjusting the parameters in the default
:class:`pyRSD.rsdfit.theory.parameters.ParameterSet` object. The user
should get the default parameters from the :func:`default_params` function
and then make the desired changes, before writing the parameters to a file.

For example, to fix the satellite fraction and only use 12 free parameters
in the fitting procedure, one can simply do

.. ipython:: python

    from pyRSD.rsd import GalaxySpectrum
    model = GalaxySpectrum()

    # default params
    params = model.default_params()

    # fix the satellite fraction
    params['fs'].vary = False

    # write new configuration to a file
    params.to_file('params.dat', mode='a')

.. ipython:: python
    :suppress:

    import os
    os.remove('params.dat')


The Sample Number Density
~~~~~~~~~~~~~~~~~~~~~~~~~

The model requires the number density of the sample to properly account
for 1-halo terms. This number density is assumed to be constant and in the
case of a number density that varies with redshift, the value can be thought
of an average number density.

.. note::

    Typically, the number density value used is the inverse of the sample
    shot noise.

The value of the ``nbar`` parameter should be updated by the user, as

.. ipython:: python

    # get the default params
    params = model.default_params()

    # this is the default nbar value
    print(params['nbar'])

    # change to the right value
    params['nbar'].value = 4e-5 # in units of (Mpc/h)^{-3}


Configuring the GalaxySpectrum Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~pyRSD.rsd.GalaxySpectrum` model has several configuration parameters,
which can be passed to :func:`~pyRSD.rsd.GalaxySpectrum.__init__` function.
Most of these parameters have sensible default values that should not be
changed by the user. However, there are a few that should be changed based
on the data set being fit. The most important are the sample redshift and
the cosmological parameters.

The redshift of the sample is taken to be constant. For samples that have
a number density varying with redshift, the redshift should be set to the
effective redshift of the sample. In the parameter file passed to ``rsdfit``
the redshift can be set as::

    model.z = 0.55

The cosmological parameters can also be specified by the user in the
parameter file -- they set the shape of the linear power spectrum which is the
input to the :class:`~pyRSD.rsd.GalaxySpectrum` model. However, if an
already initialized model is being passed to the ``rsdfit`` command via the
``-m`` flag, then the user doesn't necessarily need to specify the cosmology
again.

The cosmological parameters can be specified in a number of ways:

1. **the name of a file**

    The cosmological parameters can be read from a parameter file. See the
    ``pyRSD/data/params`` directory for example parameter files.

2. **the name of a builtin cosmology**

    The name of a builtin cosmology, such as ``Planck15`` or ``WMAP9``

3. **a dictionary of parameters**

    A dictionary of key/value pairs, such as ``Ob0``, ``Om0``, etc, which
    will be passed to the :class:`pyRSD.rsd.cosmology.Cosmology.__init__` function.

The cosmology parameters can be specified in the parameter file by
setting the ``model.params`` parameter. For example, to use the WMAP9
parameter set, simply specify the following in the parameter file::

    model.params = "WMAP9"

Additionally, there are other model configuraton parameters that can be
set in the parameter file, as long as they are prefixed with ``model.``. The
:attr:`config` attribute of the :class:`~pyRSD.rsd.GalaxySpectrum` object
gives these values

.. ipython:: python

    from pyRSD.rsd import GalaxySpectrum

    model = GalaxySpectrum()

    for k in sorted(model.config):
        print("%s = %s" %(k, str(model.config[k])))
