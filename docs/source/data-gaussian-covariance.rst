.. currentmodule:: pyRSD.rsdfit.data

Generating Gaussian Covariance Matrices
=======================================

In this section, we describe how users can use either a
:class:`~pyRSD.rsd.GalaxySpectrum` or :class:`~pyRSD.rsd.QuasarSpectrum` model
instance to compute the analytic Gaussian
covariance matrix for either multipole or :math:`P(k,\mu)` wedges
using the :class:`PoleCovarianceMatrix` and :class:`PkmuCovarianceMatrix`
class objects.

Surveys with a varying :math:`n(z)`
-----------------------------------

For surveys with a number density of objects that varies with redshift,
we provide the :func:`PoleCovarianceMatrix.cutsky_gaussian_covariance`
function to compute a Gaussian estimate of the covariance. There are a few
key parameters for computing this estimate:

- **model** : the pyRSD model object, used to compute :math:`P(k,\mu)` in the covariance calculation
- **nbar** : a callable function that returns the :math:`n(z)` of the survey
- **fsky** : the sky fraction that survey covers
- **k** : the wavenumbers where the covariance is evaluated, in units of :math:`h/\mathrm{Mpc}`
- **ells** : the multipoles to compute the covariance of
- **zmin, zmax** : the redshift range of the survey
- **P0_FKP** : the value of :math:`P_0` to use in the FKP weights, given by :math:`1/(1+n(z)P_0)`

Below is an example of how to create a :class:`PoleCovarianceMatrix` object
holding the Gaussian covariance estimate for the BOSS DR12 data set.

.. code-block:: python

    import numpy
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from pyRSD.rsd import GalaxySpectrum
    from pyRSD.rsdfit.data import PoleCovarianceMatrix

    # load n(z) from file and interpolate it
    filename = 'nbar_DR12v5_CMASSLOWZ_North_om0p31_Pfkp10000.dat'
    nbar = numpy.loadtxt(filename, skiprows=3)
    nbar = spline(nbar[:,0], nbar[:,3])

    # the sky fraction the survey covers
    fsky = 0.1436

    # the model instance to compute P(k,mu)
    model = GalaxySpectrum(params='boss_dr12_fidcosmo.ini')

    # the k values to compute covariance at
    k = numpy.arange(0., 0.4, 0.005) + 0.005/2

    # the multipoles to compute covariance of
    ells = [0,2,4]

    # the redshift range of the survey
    zmin = 0.2
    zmax = 0.5

    # the FKP weight P0
    P0_FKP = 1e4

    # the PoleCovarianceMatrix holding the Gaussian covariance
    C = PoleCovarianceMatrix.cutsky_gaussian_covariance(model, k, ells, nbar, fsky, zmin, zmax, P0_FKP=P0_FKP)

.. warning::

    If using the Gaussian covariance in parameter fits, be sure to ensure that
    the ``k`` parameter used in the covariance matrix
    calculation agrees with :math:`k` range used for the data statistics
    in the file specified by the :attr:`data.data_file`.


Simulation boxes with a constant :math:`\bar{n}`
------------------------------------------------

For periodic simulation boxes with a constant number density :math:`\bar{n}`, we
can compute the Gaussian covariance matrix of multipoles using
:func:`PoleCovarianceMatrix.periodic_gaussian_covariance` and for :math:`P(k,\mu)`
wedges using :func:`PkmuCovarianceMatrix.periodic_gaussian_covariance`.
These estimates can be computed by specifying the number density :math:`\bar{n}`
and the volume of the simulation box. For more details, see equations 16
and 17 of `Grieb et al. 2015 <https://arxiv.org/abs/1509.04293>`_.

For example, we can generate a :class:`PkmuCovarianceMatrix` object holding
the Gaussian wedge covariance using:

.. code-block:: python

    import numpy
    from pyRSD.rsd import GalaxySpectrum
    from pyRSD.rsdfit.data import PkmuCovarianceMatrix

    # volume of the box
    volume = 1380.0**3

    # constant number density in the box
    nbar = 3e-4

    # the RSD model
    model = GalaxySpectrum(params='boss_dr12_fidcosmo.ini')

    # evaluate the covariance at these wavenumbers
    k = numpy.arange(0., 0.4, 0.005) + 0.005/2

    # the edges of the P(k,mu) wedges
    mu_edges = [0., 0.2, 0.4, 0.6, 0.8, 1.0]

    # the PkmuCovarianceMatrix holding the Gaussian covariance
    C = PkmuCovarianceMatrix.periodic_gaussian_covariance(model, k, ells, nbar, volume)


And, we can generate a :class:`PoleCovarianceMatrix` object holding
the Gaussian multipole covariance using:

.. code-block:: python

    import numpy
    from pyRSD.rsd import GalaxySpectrum
    from pyRSD.rsdfit.data import PoleCovarianceMatrix

    # volume of the box
    volume = 1380.0**3

    # constant number density of the box
    nbar = 3e-4

    # model for computing P(k,mu)^2 in covariance calculation
    model = GalaxySpectrum(params='boss_dr12_fidcosmo.ini')

    # the wavenumbers where the covariance is evaluated
    k = numpy.arange(0., 0.4, 0.005) + 0.005/2

    # compute the covariance of these multipoles
    ells = [0,2,4]

    # the PoleCovarianceMatrix holding the Gaussian covariance
    C = PoleCovarianceMatrix.periodic_gaussian_covariance(model, k, ells, nbar, volume)

.. warning::

    If using the Gaussian covariance in parameter fits, be sure to ensure that
    the ``k`` parameter used in the covariance matrix
    calculation agrees with :math:`k` range used for the data statistics
    in the file specified by the :attr:`data.data_file`.
