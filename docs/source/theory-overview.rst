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

.. _free-params:

The Free Parameters
-------------------

The 13 free parameters by default are:

.. ipython:: python

    for par in params.free:
      print(par)

=========================== ========== =========================================================================================================
**Parameter**               **Name**   **Description**
:math:`N_{s,\mathrm{mult}}` Nsat_mult  The mean number of satellites in halos with >1 sat
:math:`\alpha_\parallel`    alpha_par  The Alcock-Paczynski effect parameter for parallel to the line-of-sight
:math:`\alpha_\perp`        alpha_perp The Alcock-Paczynski effect parameter for perpendicular to the line-of-sight
:math:`b_{1,c_A}`           b1_cA      The linear bias of type A centrals (no satellites in the same halo)
:math:`f`                   f          The growth rate at :math:`z`: :math:`f = d\mathrm{ln}D/d\mathrm{ln}a`
:math:`f_{1h, s_B s_B}`     f1h_sBsB   An order unity amplitude value multiplying the 1-halo term, ``NsBsB``
:math:`f_s`                 f_s        The satellite fraction, which is (total number of satellites / total number of galaxies)
:math:`f_{s_B}`             f_sB       The type B satellites fraction, which is (total number of type B satellites / total number of satellites)
:math:`\gamma_{b_{1,s_A}}`  gamma_b1sA The relative fraction of ``b1_sA`` to ``b1_cA``
:math:`\gamma_{b_{1,s_B}}`  gamma_b1sB The relative fraction of ``b1_sB`` to ``b1_cA``
:math:`\sigma_8(z)`         sigma8_z   The mass variance at :math:`r = 8 \ \mathrm{Mpc}/h` at :math:`z`
:math:`\sigma_c`            sigma_c    The centrals FoG velocity dispersion, in units of :math:`\mathrm{Mpc}/h`
:math:`\sigma_{s_A}`        sigma_sA   The type A satellites FoG velocity dispersion, in units of :math:`\mathrm{Mpc}/h`
=========================== ========== =========================================================================================================



.. _constrained-params:

The Constrained Parameters
--------------------------

There are several constrained parameters, i.e., parameters whose values are
solely determined by other parameters, in the default configuration. Note
also that since these parameters are not freely varied, they do not require
priors.

.. ipython:: python

    for par in params.constrained:
      print(par)

===================== ======== ========================================================================================================
**Parameter**         **Name** **Description**
:math:`F_\mathrm{AP}` F_AP     The AP parameter, given by :math:`alpha_\parallel / \alpha_\perp`
:math:`N_{c_B s}`     N_cBs    The amplitude of the constant 1-halo term between type B centrals, [units: :math:`(\mathrm{Mpc}/h)^3`]
:math:`N_{s_B s_B}`   N_sBsB   The amplitude of the constant 1-halo term between type B satellites, [units: :math:`(\mathrm{Mpc}/h)^3`]
:math:`\alpha`        alpha    The isotropic AP dilation, given by :math:`(\alpha_\perp^2 \alpha_\parallel)^{1/3}`
:math:`b_1`           b1       The total galaxy linear bias
:math:`b_{1,c}`       b1_c     The linear bias of the central sample
:math:`b_{1,c_B}`     b1_cB    The linear bias of type B centrals (1 or more satellite(s) in the same halo)
:math:`b_{1,s}`       b1_s     The linear bias of the satellite sample
:math:`b_{1,s_A}`     b1_sA    The linear bias of the type A satellites sample
:math:`b_{1,s_B}`     b1_sB    The linear bias of the type B satellites sample
:math:`b_1 \sigma_8`  b1sigma8 The value of :math:`b_1(z) \times \sigma_8(z)`
:math:`\epsilon`      epsilon  The anisotropic AP warping, given by :math:`(\alpha_\perp / \alpha_\parallel)^{-1/3} - 1`
:math:`f_{c_B}`       f_cB     The type B centrals fraction, which is (total number of type B centrals / total number of centrals)
:math:`f \sigma_8`    fsigma8  The value of :math:`f(z) \times \sigma_8(z)`
:math:`\sigma_{s_B}`  sigma_sB The type B satellites FoG velocity dispersion, in units of :math:`\mathrm{Mpc}/h`
===================== ======== ========================================================================================================


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
