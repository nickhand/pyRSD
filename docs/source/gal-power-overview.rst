Overview
========

.. currentmodule:: pyRSD.rsd

Theoretical Background
----------------------

The model is described in detail in the paper `Hand et al. 2017 <https://arxiv.org/abs/1706.02362>`_,
but we summarize the relevant information needed to get started using pyRSD
below.

Our galaxy model decomposes the galaxy sample into several subsamples, and
computes the correlations between each of those subsamples. The galaxy
sample gets divided into four subsamples:

================= =======================================================================
**Sample**        **Description**
type A centrals   isolated centrals (no satellites in the same halo)
type B centrals   non-isolated centrals (at least one satellite in same halo)
type A satellites isolated satellites (no other satellites in same halo)
type B satellites non-isolated satellites (at least one other satellite in the same halo)
================= =======================================================================

This sample decomposition is useful because it allows us to separately model
the correlations between galaxies in the same halo (denoted as "1-halo" terms)
and correlations that arise from galaxies in separate dark matter halos
(denoted as "2-halo" terms).

Thus, we can model the total galaxy power spectrum as

.. math::

    P^{gg}(k,\mu) = (1-f_s)^2 P^{cc} + 2 f_s (1-f_s) P^{cs} + f_s^2 P^{ss},

where :math:`f_s` is the satellite fraction and :math:`P^{cc}`,
:math:`P^{cs}`, and :math:`P^{ss}` are the centrals auto power, central-satellite
cross power, and satellite auto power, respectively.

.. _model-initialization:

Model Initialization
--------------------

The galaxy power spectrum model can be initialized from a cosmology and
a redshift specified by the user. For example, you can initialize the model
at :math:`z=0` for the Planck 2015 cosmology parameters as

.. ipython:: python

    from pyRSD.rsd import GalaxySpectrum
    from pyRSD.rsd.cosmology import Planck15
    model = GalaxySpectrum(z=0, params=Planck15)

Once the model object has been created, the underlying elements of the model
need to be initialized. Depending on your machine, this will take several minutes
to complete.

.. code-block:: python

    # model takes several minutes to initialize once
    model.initialize()

    # set kmin/kmax limits
    model.kmin = 1e-3
    model.kmax = 0.5

The initialization only needs to be done once, and once the model initialized,
the evaluation of the model will be fast (typically < 1 second), as long
as the desired :math:`k` value is valid, as specified by the :attr:`kmin`
and :attr:`kmax` attributes of the model.

.. warning::

    The :class:`~pyRSD.rsd.GalaxySpectrum` class has attributes :attr:`kmin` and :attr:`kmax`
    that specify where the valid wavenumber range over which the underlying model
    has been initialized. Outside of this range, the model must evaluate
    each component of the model for each :math:`k` value, which can be
    time-consuming. A warning will be printed when this occurs.


.. ipython:: python
    :suppress:

    import numpy
    from matplotlib import pyplot as plt
    model = GalaxySpectrum.from_npy('data/galaxy_model.npy')

Because the model initialization is time consuming, we recommend saving the
initialized model and then reading the model from disk when it is needed again.
This can be achieved with the numpy's pickling functionality:

.. code-block:: python

    # save the initialized model to disk
    model.to_npy('galaxy_model.npy')

    # read a new model from disk
    model2 = GalaxySpectrum.from_npy('galaxy_model.npy')

Model Parameters
----------------

The GalaxySpectrum object has several model parameters. These parameters all
of default values, but the attributes can be explicitly set by the user to
change the behavior of the model.

The parameters are:

========================= ========== ===================================================================================================================
**Cosmology**             **Name**   **Description**
:math:`\alpha_\parallel`  alpha_par  The Alcock-Paczynski effect parameter for parallel to the line-of-sight
:math:`\alpha_\perp`      alpha_perp The Alcock-Paczynski effect parameter for perpendicular to the line-of-sight
:math:`f`                 f          The growth rate at :math:`z`: :math:`f = d\mathrm{ln}D/d\mathrm{ln}a`
:math:`\sigma_8(z)`       sigma8_z   The mass variance at :math:`r = 8 \ \mathrm{Mpc}/h` at :math:`z`; this sets the linear power spectrum normalization
**Linear biases**
:math:`b_{1,c_A}`         b1_cA      The linear bias of the type A centrals sample
:math:`b_{1,c_B}`         b1_cB      The linear bias of the type B centrals sample
:math:`b_{1,s_A}`         b1_sA      The linear bias of the type A satellites sample
:math:`b_{1,s_B}`         b1_sB      The linear bias of the type B satellites sample
**Sample fractions**
:math:`f_s`               f_s        The satellite fraction, which is (total number of satellites / total number of galaxies)
:math:`f_{c_B}`           f_cB       The type B centrals fraction, which is (total number of type B centrals / total number of centrals)
:math:`f_{s_B}`           f_sB       The type B satellites fraction, which is (total number of type B satellites / total number of satellites)
**Finger-of-God damping**
:math:`\sigma_c`          sigma_c    The centrals FoG velocity dispersion, in units of :math:`\mathrm{Mpc}/h`
:math:`\sigma_{s_A}`      sigma_sA   The type A satellites FoG velocity dispersion, in units of :math:`\mathrm{Mpc}/h`
:math:`\sigma_{s_B}`      sigma_sB   The type B satellites FoG velocity dispersion, in units of :math:`\mathrm{Mpc}/h`
**1-halo amplitudes**
:math:`N_{c_B s}`         N_cBs      The amplitude of the constant 1-halo term between type B centrals, [units: :math:`(\mathrm{Mpc}/h)^3`]
:math:`N_{s_B s_B}`       N_sBsB     The amplitude of the constant 1-halo term between type B satellites, [units: :math:`(\mathrm{Mpc}/h)^3`]
========================= ========== ===================================================================================================================

The parameters can be updated via the usual attribute setting mechanism

.. ipython:: python

    # update normalization via sigma8_z
    model.sigma8_z = 0.62

    # update satellite fraction
    model.f_s = 0.12
