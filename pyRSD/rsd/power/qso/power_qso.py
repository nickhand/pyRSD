from pyRSD.rsd._cache import parameter, cached_property, interpolated_function
from pyRSD.rsd import tools, HaloSpectrum
from pyRSD import numpy as np
from ..gal.fog_kernels import FOGKernel

from scipy.special import legendre
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import contextlib
import warnings


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
    # power as a function of mu
    #---------------------------------------------------------------------------
    @interpolated_function("k", "sigma8_z", "b1", "_power_norm", interp="k")
    def P_mu0(self, k):
        """
        The isotropic part of the Kaiser formula
        """
        return self.b1**2 * self.normed_power_lin(k)

    @interpolated_function("k", "sigma8_z", "b1", "f", "_power_norm", interp="k")
    def P_mu2(self, k):
        """
        The mu^2 term of the Kaiser formula
        """
        return 2*self.f*self.b1 * self.normed_power_lin(k)

    @interpolated_function("k", "sigma8_z", "b1", "f", "_power_norm", interp="k")
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

    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def power(self, k, mu, flatten=True):
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
        pkmu = self._power(k, mu)

        # add FOG damping
        G = self.FOG(k, mu, self.sigma_fog)
        pkmu *= G**2

        # add shot noise offset
        pkmu += self.N

        if flatten: pkmu = np.ravel(pkmu, order='F')
        return pkmu

    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_k(self, k, mu):
        """
        The derivative with respect to `k_AP`
        """
        G = self.FOG(k, mu, self.sigma_fog)
        Gprime = self.FOG.derivative_k(k, mu, self.sigma_fog)

        deriv = super(QuasarSpectrum, self).derivative_k(k, mu)
        power = self.power(k, mu)

        return G**2 * deriv + 2 * G*Gprime * power

    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_mu(self, k, mu):
        """
        The derivative with respect to `mu_AP`
        """
        G = self.FOG(k, mu, self.sigma_fog)
        Gprime = self.FOG.derivative_k(k, mu, self.sigma_fog)

        deriv = super(QuasarSpectrum, self).derivative_mu(k, mu)
        power = self.power(k, mu)

        return G**2 * deriv + 2 * G*Gprime * power
