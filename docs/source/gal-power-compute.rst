Computing Power Spectra
=======================

.. currentmodule:: pyRSD.rsd

With an initialized model, we can compute the power spectrum either
as a function of :math:`k` and :math:`\mu`, :math:`P(k,\mu)`, or compute
the multipoles of the power spectrum, :math:`P_\ell(k)`. This
is accomplished either by calling the :func:`GalaxySpectrum.power`
or :func:`GalaxySpectrum.poles` functions.


.. ipython:: python
    :suppress:
    
    import numpy
    from matplotlib import pyplot as plt
    from pyRSD.rsd import GalaxySpectrum

    model = GalaxySpectrum.from_npy('data/galaxy_power.npy')

For example, to compute :math:`P(k,\mu)` for 5 :math:`\mu` bins:

.. ipython:: python

    k = numpy.logspace(-2, numpy.log10(0.4), 100)

    # this is mu = 0.1, 0.3, 0.5, 0.7, 0.9
    mu = numpy.arange(0.1, 1.0, 0.2)
    Pkmu = model.power(k, mu) # shape is (100,5)

    for i, imu in enumerate(mu):
      plt.loglog(k, Pkmu[:,i], label=r"$\mu = %.1f$" %imu)

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)

    @savefig pkmu_model_plot.png width=6in
    plt.show()

And, for example, the monopole, quadrupole, and hexadecapole (:math:`\ell=0,2,4`)
can be computed as

.. ipython:: python

    ells = [0, 2, 4]
    Pell = model.poles(k, ells) # list of 3 (100,) arrays

    for i, ell in enumerate(ells):
      plt.loglog(k, Pell[i], label=r"$\ell = %d$" %ell)

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P_\ell$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)

    @savefig poles_model_plot.png width=6in
    plt.show()
