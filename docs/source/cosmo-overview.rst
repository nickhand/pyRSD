Overview
========

Users can specify cosmological parameters by creating a new
:class:`~pyRSD.rsd.cosmology.Cosmology` object or by using one of the builtin
cosmologies (see :ref:`available-cosmo`).

When constructing a new Cosmology object, parameter values should be specified
as keyword parameters. The parameters that can be specified are:

========= ======================================================================
Parameter Description
========= ======================================================================
H0        The Hubble constant at z=0, in km/s/Mpc
Om0       The matter density/critical density at z=0
Ob0       The baryon density/critical density at z=0
Ode0      The dark energy density/critical density at z=0
w0        The dark energy equation of state
Tcmb0     The temperature of the CMB in K at z=0
Neff      The the effective number of neutrino species
m_nu      The mass of neutrino species in eV
sigma8    The the mass variance on the scale of R=8 Mpc/h at z=0, which sets the
          normalization of the linear power spectrum
n_s       The the spectral index of the primoridal power spectrum
flat      if True, automatically set Ode0 such that Ok0 is zero
========= ======================================================================

.. note::

  The :class:`pyRSD.rsd.cosmology.Cosmology` class is nearly identical to the
  :class:`astropy.cosmology.FLRW` object, with the addition of the ``n_s``
  and ``sigma8`` attributes

Examples
--------

.. ipython:: python

    from pyRSD.rsd import cosmology

    # initialize a new Cosmology
    cosmo = cosmology.Cosmology(H0=70, sigma8=0.80, n_s=0.96)

    # access parameters as attribute or key entry
    print(cosmo['sigma8'], cosmo.sigma8)

    # compute the comoving distance to z = 0.4
    Dz = cosmo.comoving_distance(0.4)
    print(Dz)

The Cosmology class is read-only; changes to the parameters should be
performed with the :func:`~pyRSD.rsd.cosmology.Cosmology.clone` function,
which creates a copy of the class, with any specified changes.

.. ipython:: python

    new_cosmo = cosmo.clone(sigma8=0.85, Om0=0.27)

    # compare sigma8
    print(cosmo.sigma8, new_cosmo.sigma8)

    # compare Om0
    print(cosmo.Om0, new_cosmo.Om0)

    # everything else stays the same
    print(cosmo.n_s, new_cosmo.n_s)
