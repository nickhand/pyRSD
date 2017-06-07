API
---

.. currentmodule:: pyRSD.rsd.cosmology

Top level user functions:

.. autosummary::

  Cosmology
  Cosmology.clone
  Cosmology.from_astropy
  Cosmology.to_class

There are builtin, default Cosmology objects available:

====================================== ============================== ====  ===== =======
Name                                   Source                         H0    Om    Flat
====================================== ============================== ====  ===== =======
:attr:`~pyRSD.rsd.cosmology.WMAP5`     Komatsu et al. 2009            70.2  0.277 Yes
:attr:`~pyRSD.rsd.cosmology.WMAP7`     Komatsu et al. 2011            70.4  0.272 Yes
:attr:`~pyRSD.rsd.cosmology.WMAP9`     Hinshaw et al. 2013            69.3  0.287 Yes
:attr:`~pyRSD.rsd.cosmology.Planck13`  Planck Collab 2013, Paper XVI  67.8  0.307 Yes
:attr:`~pyRSD.rsd.cosmology.Planck15`  Planck Collab 2015, Paper XIII 67.7  0.307 Yes
====================================== ============================== ====  ===== =======

The Cosmology class inherits the following **attributes** from the
:class:`astropy.cosmology.FLRW` class:

.. currentmodule:: astropy.cosmology.FLRW

.. autosummary::

  H0
  Neff
  Ob0
  Ode0
  Odm0
  Ogamma0
  Ok0
  Om0
  Onu0
  Tcmb0
  Tnu0
  critical_density0
  h
  has_massive_nu
  hubble_distance
  hubble_time
  m_nu

The Cosmology class inherits the following **methods** from the
:class:`astropy.cosmology.FLRW` class:

.. autosummary::

  H
  Ob
  Ode
  Odm
  Ogamma
  Ok
  Om
  Onu
  Tcmb
  Tnu
  abs_distance_integrand
  absorption_distance
  age
  angular_diameter_distance
  angular_diameter_distance_z1z2
  arcsec_per_kpc_comoving
  arcsec_per_kpc_proper
  comoving_distance
  comoving_transverse_distance
  comoving_volume
  critical_density
  de_density_scale
  differential_comoving_volume
  distmod
  efunc
  inv_efunc
  kpc_comoving_per_arcmin
  kpc_proper_per_arcmin
  lookback_distance
  lookback_time
  lookback_time_integrand
  luminosity_distance
  nu_relative_density
  scale_factor
  w

.. _available-cosmo:

Available Cosmologies
~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: pyRSD.rsd.cosmology.Planck13
.. autoattribute:: pyRSD.rsd.cosmology.Planck15
.. autoattribute:: pyRSD.rsd.cosmology.WMAP5
.. autoattribute:: pyRSD.rsd.cosmology.WMAP7
.. autoattribute:: pyRSD.rsd.cosmology.WMAP9

pyRSD.rsd.cosmology.Cosmology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyRSD.rsd.cosmology

.. autoclass:: Cosmology
  :members: clone, from_astropy, to_class


.. _pygcl-cosmo:

pyRSD.pygcl.Cosmology
~~~~~~~~~~~~~~~~~~~~~

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
