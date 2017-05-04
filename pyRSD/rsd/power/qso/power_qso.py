from pyRSD.rsd._cache import parameter, cached_property, interpolated_function
from pyRSD.rsd import tools, HaloSpectrum
from pyRSD import numpy as np, pygcl
from ..gal.fog_kernels import FOGKernel

from scipy.special import legendre
from scipy.integrate import simps, quad
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import contextlib
import warnings

def vectorize_if_needed(func, *x):
    """
    Helper function to vectorize functions on array inputs;
    borrowed from :mod:`astropy.cosmology.core`
    """
    if any(map(isiterable, x)):
        return np.vectorize(func)(*x)
    else:
        return func(*x)

def isiterable(obj):
    """
    Returns `True` if the given object is iterable;
    borrowed from :mod:`astropy.cosmology.core`
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def growth_function(cosmo, z):

    # compute E(z)
    Om0, Ode0, Ok0 = cosmo['Omega0_m'], cosmo['Omega0_lambda'], cosmo['Omega0_k']
    efunc = lambda a: np.sqrt(a ** -3 * Om0 + Ok0 * a ** -2 + Ode0)

    # this is 1 / (E(a) * a)**3, with H(a) = H0 * E(a)
    integrand = lambda a: 1.0 / (a * efunc(a))**3
    f = lambda red: quad(integrand, 0., 1./(1+red))[0]
    return efunc(1./(1+z))*vectorize_if_needed(f, z)

class QuasarSpectrum(HaloSpectrum):
    """
    The quasar redshift space power spectrum, a subclass of
    :class:`~pyRSD.rsd.HaloSpectrum` for biased redshift space power spectra
    """
    def __init__(self, fog_model='gaussian', **kwargs):
        """
        Initialize the QuasarSpectrum

        Parameters
        ----------
        fog_model : str, optional
            the string specifying the FOG model to use; one of
            ['modified_lorentzian', 'lorentzian', 'gaussian'].
            Default is 'gaussian'
        """
        # the base class
        super(QuasarSpectrum, self).__init__(**kwargs)

        # set the defaults
        self.fog_model = fog_model
        self.sigma_fog = 4.0
        self.include_2loop = False
        self.N = 0

        # fnl parameters
        self.f_nl = 0
        self.p = 1.6 # good for quasars

    def default_params(self):
        """
        Return a QuasarPowerParameters instance holding the default
        model parameters configuration

        The model associated with the parameter is ``self``
        """
        from pyRSD.rsdfit.theory import QuasarPowerParameters
        return QuasarPowerParameters.from_defaults(model=self)

    @parameter
    def fog_model(self, val):
        """
        Function to return the FOG suppression factor, which reads in a
        single variable `x = k \mu \sigma`
        """
        allowable = ['modified_lorentzian', 'lorentzian', 'gaussian']
        if val not in allowable:
            raise ValueError("`fog_model` must be one of %s" %allowable)

        return val

    @parameter
    def sigma_fog(self, val):
        """
        The FOG velocity dispersion in Mpc/h
        """
        return val

    @parameter
    def p(self, val):
        """
        The value encoding the type of tracer; must be between 1 and 1.6,
        with p=1 suitable for LRGs and p=1.6 suitable for quasars

        The scale-dependent bias is proportional to: ``b1 - p``
        """
        if not 1 <= val <= 1.6:
            raise ValueError("f_nl ``p`` value should be between 1 and 1.6 (inclusive)")
        return val

    @parameter
    def f_nl(self, val):
        """
        The amplitude of the local-type non-Gaussianity
        """
        return val

    @parameter
    def N(self, val):
        """
        Constant offset to model, set to 0 by default
        """
        return val

    @cached_property("fog_model")
    def FOG(self):
        """
        Return the FOG function
        """
        return FOGKernel.factory(self.fog_model)

    #---------------------------------------------------------------------------
    # primordial non-Gaussianity functions
    #---------------------------------------------------------------------------
    @cached_property()
    def delta_crit(self):
        """
        The usual critical overdensity value for collapse from Press-Schechter
        """
        return 1.686

    @interpolated_function("k", "cosmo", interp="k")
    def alpha_png(self, k):
        """
        The primordial non-Gaussianity alpha value

        see e.g., equation 2 of Mueller et al 2017
        """
        # transfer function, normalized to unity
        Tk = (self.normed_power_lin(k)/k**self.cosmo.n_s())**0.5
        Tk /= Tk.max()

        # normalization
        c = pygcl.Constants.c_light / pygcl.Constants.km # in km/s
        H0 = 100 # in units of h km/s/Mpc

        # normalizes growth function to unity in matter-dominated epoch
        g_ratio = growth_function(self.cosmo, 0.)
        g_ratio /= (1+100.) * growth_function(self.cosmo, 100.)

        return 2*k**2 * Tk * self.D / (3*self.cosmo.Omega0_m()) * (c/H0)**2 * g_ratio

    @interpolated_function("k", "b1", "f_nl", interp="k")
    def delta_bias(self, k):
        """
        The scale-dependent bias introduced by primordial non-Gaussianity
        """
        if self.f_nl == 0: return k*0.

        return 2*(self.b1-self.p)*self.f_nl*self.delta_crit/self.alpha_png(k)

    @interpolated_function("k", "b1", "delta_bias", interp="k")
    def btot(self, k):
        """
        The total bias, accounting for scale-dependent bias introduced by
        primordial non-Gaussianity
        """
        return self.b1 + self.delta_bias(k)

    #---------------------------------------------------------------------------
    # power as a function of mu
    #---------------------------------------------------------------------------
    @interpolated_function("k", "sigma8_z", "_power_norm", "btot", interp="k")
    def P_mu0(self, k):
        """
        The isotropic part of the Kaiser formula
        """
        return self.btot(k)**2 * self.normed_power_lin(k)

    @interpolated_function("k", "sigma8_z", "f", "_power_norm", "btot", interp="k")
    def P_mu2(self, k):
        """
        The mu^2 term of the Kaiser formula
        """
        return 2*self.f*self.btot(k) * self.normed_power_lin(k)

    @interpolated_function("k", "sigma8_z", "f", "_power_norm", interp="k")
    def P_mu4(self, k):
        """
        The mu^4 term of the Kaiser formula
        """
        return self.f**2 * self.normed_power_lin(k)

    @interpolated_function(interp="k")
    def P_mu6(self, k):
        """
        The mu^6 term is zero in the Kaiser formula
        """
        return k*0.

    @tools.alcock_paczynski
    def power(self, k, mu, flatten=False):
        """
        Return the redshift space power spectrum at the specified value of mu,
        including terms up to ``mu**self.max_mu``.

        Parameters
        ----------
        k : float or array_like
            The wavenumbers in `h/Mpc` to evaluate the model at
        mu : float, array_like
            The mu values to evaluate the power at.

        Returns
        -------
        pkmu : float, array_like
            The power model P(k, mu). If `mu` is a scalar, return dimensions
            are `(len(self.k), )`. If `mu` has dimensions (N, ), the return
            dimensions are `(len(k), N)`, i.e., each column corresponds is the
            model evaluated at different `mu` values. If `flatten = True`, then
            the returned array is raveled, with dimensions of `(N*len(self.k), )`
        """
        # the linear kaiser P(k,mu)
        pkmu = super(QuasarSpectrum, self).power(k, mu)

        # add FOG damping
        G = self.FOG(k, mu, self.sigma_fog)
        pkmu *= G**2

        # add shot noise offset
        pkmu += self.N

        if flatten: pkmu = np.ravel(pkmu, order='F')
        return pkmu

    @tools.alcock_paczynski
    def derivative_k(self, k, mu):
        """
        The derivative with respect to `k_AP`
        """
        G = self.FOG(k, mu, self.sigma_fog)
        Gprime = self.FOG.derivative_k(k, mu, self.sigma_fog)

        deriv = super(QuasarSpectrum, self).derivative_k(k, mu)
        power = super(QuasarSpectrum, self).power(k, mu)

        return G**2 * deriv + 2 * G*Gprime * power

    @tools.alcock_paczynski
    def derivative_mu(self, k, mu):
        """
        The derivative with respect to `mu_AP`
        """
        G = self.FOG(k, mu, self.sigma_fog)
        Gprime = self.FOG.derivative_mu(k, mu, self.sigma_fog)

        deriv = super(QuasarSpectrum, self).derivative_mu(k, mu)
        power = super(QuasarSpectrum, self).power(k, mu)

        return G**2 * deriv + 2 * G*Gprime * power
