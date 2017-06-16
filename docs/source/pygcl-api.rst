API
---

.. _pygcl-cosmo:

The Cosmology object
~~~~~~~~~~~~~~~~~~~~

Methods that return various cosmological parameters:

.. currentmodule:: pyRSD.pygcl

.. autoclass:: Cosmology
    :members: EvaluateTransfer

    .. method:: H0()

      the present-day Hubble constant in units of km/s/Mpc

    .. method:: h()

      the dimensionless Hubble constant

    .. method:: Tcmb()

      CMB temperature today in Kelvin

    .. method:: Omega0_b()

      present-day baryon density parameter

    .. method:: Omega0_cdm()

      present-day cold dark matter density fraction

    .. method:: Omega0_ur()

       present-day ultra-relativistic neutrino density fraction

    .. method:: Omega0_m()

       present-day non-relativistic density fraction

    .. method:: Omega0_r()

      present-day relativistic density fraction

    .. method:: Omega0_g()

      present-day photon density fraction

    .. method:: Omega0_lambda()

      present-day cosmological constant density fraction

    .. method:: Omega0_fld()

       present-day dark energy fluid density fraction (valid if Omega0_lambda is unspecified)

    .. method:: Omega0_k()

      present-day curvature density fraction

    .. method:: w0_fld()

      present-day fluid equation of state parameter

    .. method:: wa_fld()

      present-day equation of state derivative

    .. method:: n_s()

      the spectral index of the primordial power spectrum

    .. method:: k_pivot()

      the pivot scale in 1/Mpc

    .. method:: A_s()

      scalar amplitude = curvature power spectrum at pivot scale

    .. method:: ln_1e10_A_s()

      convenience function returns log (1e10*A_s)

    .. method:: sigma8()

      convenience function to return sigma8 at z = 0

    .. method:: k_max()

       maximum k value computed in h/Mpc

    .. method:: k_min()

      minimum k value computed in h/Mpc

    .. method:: z_drag()

      the baryon drag redshift

    .. method:: rs_drag()

      the comoving sound horizon at the baryon drag redshifts

    .. method:: tau_reio()

      the reionization optical depth

    .. method:: z_reio()

      the redshift of reionization

    .. method:: rho_crit(cgs=False)

      the critical density at z = 0 in units of h^2 M_sun / Mpc^3 if cgs = False,
      or in units of h^2 g / cm^3

Methods that return background quantities as a function of redshift:

  .. method:: f_z(z)

    the logarithmic growth rate, dlnD/dlna, at z

  .. method:: H_z(z)

    the Hubble parameter at z in km/s/Mpc

  .. method:: Da_z(z)

    the angular diameter distance to z in Mpc -- this is Dm/(1+z)

  .. method:: Dc_z(z)

    the conformal distance to z in the flat case in Mpc

  .. method:: Dm_z(z)

    the comoving radius coordinate in Mpc, which is equal to the conformal
    distance in the flat case

  .. method:: D_z(z)

    the growth function D(z) / D(0) (normalized to unity at z = 0)

  .. method:: Sigma8_z(z)

    the scalar amplitude at z, equal to sigma8 * D(z)

  .. method:: Omega_m_z(z)

    Omega0_m as a function of z

  .. method :: rho_bar_z(z, cgs=False)

    the mean matter density in units of h^2 M_sun / Mpc^3 if cgs = False, or
    in units of g / cm^3

  .. method:: rho_crit_z(z, cgs=False)

    the critical matter density in units of h^2 M_sun / Mpc^3 if cgs = False, or
    in units of g / cm^3

  .. method:: dV(z)

    the comoving volume element per unit solid angle per unit redshift in Gpc^3

  .. method:: V(zmin, zmax, Nz=1024)

    the comoving volume between two redshifts (full sky)

Power spectrum objects
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearPS

    Compute the linear power spectrum, using CLASS

    .. method:: __init__(pygcl.Cosmology cosmo, float z=0)

      initialize the linear power spectrum for a given cosmology and redshift

    .. method:: __call__(k)

      evaluate the linear power spectrum at the wavenumber ``k``, where ``k``
      is in units of :math:`h/\mathrm{Mpc}`

    .. method:: SetSigma8AtZ(sigma8_z)

      set the normalization of the power spectrum via setting :math:`\sigma_8(z)`

.. autoclass:: ZeldovichP00

    Compute the density auto power spectrum in the Zel'dovich approximation

    .. method:: __init__(pygcl.Cosmology cosmo, float z, bool approx_lowk=False)

      initialize the class for a given cosmology and redshift; if ``approx_lowk``
      is True, use a low ``k`` approximation of the Zel'dovich approximation

    .. method:: __call__(k)

      evaluate the Zel'dovich power spectrum at the wavenumber ``k``, where ``k``
      is in units of :math:`h/\mathrm{Mpc}`

    .. method:: SetSigma8AtZ(sigma8_z)

      set the normalization of the power spectrum via setting :math:`\sigma_8(z)`

.. autoclass:: ZeldovichP01

    Compute the density - radial momentum cross power spectrum in the Zel'dovich approximation

    .. method:: __init__(pygcl.Cosmology cosmo, float z, bool approx_lowk=False)

      initialize the class for a given cosmology and redshift; if ``approx_lowk``
      is True, use a low ``k`` approximation of the Zel'dovich approximation

    .. method:: __call__(k)

      evaluate the Zel'dovich power spectrum at the wavenumber ``k``, where ``k``
      is in units of :math:`h/\mathrm{Mpc}`

    .. method:: SetSigma8AtZ(sigma8_z)

      set the normalization of the power spectrum via setting :math:`\sigma_8(z)`

.. autoclass:: ZeldovichP11

    Compute the radial momentum auto power spectrum in the Zel'dovich approximation

    .. method:: __init__(pygcl.Cosmology cosmo, float z, bool approx_lowk=False)

      initialize the class for a given cosmology and redshift; if ``approx_lowk``
      is True, use a low ``k`` approximation of the Zel'dovich approximation

    .. method:: __call__(k)

      evaluate the Zel'dovich power spectrum at the wavenumber ``k``, where ``k``
      is in units of :math:`h/\mathrm{Mpc}`

    .. method:: SetSigma8AtZ(sigma8_z)

      set the normalization of the power spectrum via setting :math:`\sigma_8(z)`

Correlation function objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CorrelationFunction

    Compute the linear correlation function by Fourier transforming
    the linear power spectrum

    .. method:: __init__(pygcl.LinearPS plin, kmin=1e-4, kmax=10)

      initialize the class from a linear power spectrum object; ``kmin`` and
      ``kmax`` correspond to the limits of the numerical integration when
      doing the Fourier transform.

    .. method:: __call__(r)

      evaluate the correlation function at the separation ``r``, where ``r``
      is in units of :math:`\mathrm{Mpc}/h`

.. autoclass:: ZeldovichCF

    Compute the density auto correlation function in the Zel'dovich approximation

    .. method:: __init__(pygcl.Cosmology cosmo, float z, kmin=1e-4, kmax=10)

      initialize the class for a given cosmology and redshift; ``kmin`` and
      ``kmax`` correspond to the limits of the numerical integration when
      doing the Fourier transform.

    .. method:: __call__(r)

      evaluate the Zel'dovich correlation function at the separation ``r``,
      where ``r`` is in units of :math:`\mathrm{Mpc}/h`

    .. method:: SetSigma8AtZ(sigma8_z)

      set the normalization of the correlation function via setting :math:`\sigma_8(z)`
