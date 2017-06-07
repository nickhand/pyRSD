Interfacing with CLASS
======================

The power spectrum calculations performed
in pyRSD require the linear matter transfer function. We provide an interface
to the `CLASS <https://class-code.net>`_ CMB Boltzmann code via the
:class:`pyRSD.pygcl.Cosmology` class. The :mod:`pyRSD.pygcl` is the
Python General Cosmology Library is a swig-generated Python wrapper of a
C++ library that provides useful cosmological functionality, including
an interface to the CLASS code.

.. note::

    The pyRSD code relies internally on the :class:`pyRSD.pygcl.Cosmology`,
    and users need only use the :class:`pyRSD.rsd.cosmology.Cosmology` for
    most of the high-level calculations performed by pyRSD. Conversions
    between the classes will be performed automatically internally.
    We provide a brief introduction to the pygcl.Cosmology class for general
    knowledge about how pyRSD works.

A :class:`pyRSD.pygcl.Cosmology` object can be easily initialized
from a :class:`pyRSD.rsd.cosmology.Cosmology` object via the
:func:`~pyRSD.rsd.cosmology.Cosmology.to_class` function, as

.. ipython:: python

    from pyRSD.rsd.cosmology import Planck15
    from pyRSD import pygcl

    class_cosmo = Planck15.to_class()
    print(class_cosmo)

The :class:`pyRSD.pygcl.Cosmology` can compute various cosmological parameters
and background quantities as a function of redshift (see :ref:`the API<pygcl-cosmo>`).
Most importantly for pyRSD, it can compute the full matter transfer function:

.. ipython:: python

    # transfer function at k = 0.1 h/Mpc
    Tk = class_cosmo.EvaluateTransfer(0.1)
    print(Tk)
