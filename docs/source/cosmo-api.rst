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
