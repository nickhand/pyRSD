API
===

.. currentmodule:: pyRSD.rsd

Top level user functions to compute power spectra:

.. autosummary::

    GalaxySpectrum
    GalaxySpectrum.initialize
    GalaxySpectrum.power
    GalaxySpectrum.poles
    GalaxySpectrum.from_transfer


Loading GalaxySpectrum objects from and saving to pickle files:

.. autosummary::

    GalaxySpectrum.to_npy
    GalaxySpectrum.from_npy


Generating the default set of parameters for fitting the
:class:`~pyRSD.rsd.GalaxySpectrum` to data:

.. autosummary::

    GalaxySpectrum.default_params

Evaluating power spectra with an additional transfer function, i.e., on
a discrete (``k``, ``mu``) grid or convolved with a window:

.. autosummary::

    PkmuGrid
    PkmuTransfer
    PolesTransfer
    ~pyRSD.rsd.window.WindowTransfer


.. currentmodule:: pyRSD.rsd

The GalaxySpectrum Class
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GalaxySpectrum
    :members: poles, from_transfer, default_params

    .. automethod:: GalaxySpectrum.power(k, mu, flatten=False)

Gridded Power Spectra
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PkmuGrid

.. autoclass:: PkmuTransfer
    :members: __call__

.. autoclass:: PolesTransfer
    :members: __call__

Window-convolved Power Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyRSD.rsd.window.WindowTransfer
    :members: __call__
