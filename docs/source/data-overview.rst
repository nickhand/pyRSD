.. currentmodule:: pyRSD.rsdfit.data

Overview
========

The main class that is responsible for handling the data statistics, as
well as the covariance matrix used during parameter estimation, is
the :class:`pyRSD.rsdfit.data.PowerData` class.

Information about the parameters that are needed to initialize
the :class:`~pyRSD.rsdfit.data.PowerData` class can be found by using
the :func:`PowerData.help` function,

.. ipython:: python

    from pyRSD.rsdfit.data import PowerData

    # print out the help message for the parameters needed
    PowerData.help()

These parameters should be specified in the parameter file that is passed
to the ``rsdfit`` executable and the names of the parameters should be
prefixed with the ``data.`` prefix. In our example parameter file
discussed previously, we specify multipoles data to read from file as

.. literalinclude:: ../../pyRSD/data/examples/params.dat
    :lines: 16-31
    :encoding: latin-1


These parameters allow the user to specify which type of data is being
used by specifying the :attr:`mode` parameter, either ``pkmu`` for :math:`P(k,\mu)`
data or ``poles`` for :math:`P_\ell(k)` data. The user can also specify the
desired :math:`k` ranges to use when fitting the data, via the
:attr:`fitting_range` parameter.

The data itself must be read in from a plaintext file. Similarly, the
covariance matrix and grid file must also be read from a plaintext file.
See the next section :ref:`data-formats` for more specifics on the format
of these plaintext files.
