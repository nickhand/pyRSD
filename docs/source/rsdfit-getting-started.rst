Getting Started
===============

The ``rsdfit`` Executable
~~~~~~~~~~~~~~~~~~~~~~~~~~

The pyRSD package includes a ``rsdfit`` executable that is capable of
running parameter estimation using the power spectrum models
:class:`~pyRSD.rsd.GalaxySpectrum` and :class:`~pyRSD.rsd.QuasarSpectrum`
to fit power spectrum data input by the user.

After installing the pyRSD package, the ``rsdfit`` command should be
located on your ``PATH``. We can then print the help message for the command
via

.. command-output:: rsdfit -h

The ``rsdfit`` command contains four sub-commands: **nlopt**, **mcmc**, **restart**,
and **analyze**. The most important of these sub-commands are ``nlopt``
and ``mcmc``. This allows the user to run fresh parameter fits from scratch, which
is the main use case for the ``rsdfit`` executable.

The calling sequence for the ``mcmc`` command is

.. command-output:: rsdfit mcmc -h

and for the ``nlopt`` command is

.. command-output:: rsdfit nlopt -h

The three most important options for these sub-commands are:

1. **-p, ---params**

    The name of the main parameter file to load; this holds all of the
    relevant information about the data, theory, and main driver parameters.
    We'll discuss these parameter files in more detail in the next section.

2. **-o, ---output**

    This is the name of the directory where the results will be saved. Each
    time ``rsdfit`` is run, the parameter file is saved to a file "params.dat"
    in this directory as well. If this folder already exists and contains
    a "params.dat" file, those parameters will be used and results will be
    added to the existing directory.

3. **-m, ---model**

    The name of a file holding a :class:`~pyRSD.rsd.GalaxySpectrum` or
    :class:`~pyRSD.rsd.QuasarSpectrum` object to be loaded by the code.
    This argument is optional, but providing an already initialized model file
    will save a significant amount of time, since the code won't need
    to initialize a model from scratch. See :ref:`model-initialization` for
    more details on initializing and saving models.


Parameter Files
~~~~~~~~~~~~~~~

The ``rsdfit`` command is configured through a single main parameter file.
This parameter file is responsible for not only configuring the main ``rsdfit``
executable, but also for specifying the relevant theory parameters and
information about the data measurements to load. To separate the different
classes of parameters, the parameter names are prefixed with an additional
identifier, one of

1. **driver**: general parameters that determine the ``rsdfit`` configuration

2. **data**: the parameters specifying the data and covariance matrix to load

3. **theory**: the theoretical parameters for the desired pyRSD model, which determines the free pararameters, constrained parameters, etc, during the fitting procedure

4. **model**: parameters specifying the :class:`~pyRSD.rsd.GalaxySpectrum` or :class:`~pyRSD.rsd.QuasarSpectrum` configuration; these are parameters that are passed to the :func:`__init__` function of these model classes

We'll be exploring each of these parameter types in much more detail in the next
few sections, but for now, let's take a quick look at an example parameter file:

.. literalinclude:: ../../pyRSD/data/examples/params.dat
    :name: config-file

The parameter file uses a simple key/value assignment syntax to define
the parameters. It is also possible to use environment variables when
defining parameters, was was done in the above example for ``$(PYRSD_DATA)``.

To learn more about each section of the parameter file, please see
the next sections:

1. :ref:`specifying-data`
2. :ref:`specifying-theory`
3. :ref:`parameter-fits`
