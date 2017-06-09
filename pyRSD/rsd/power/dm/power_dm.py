import fnmatch
import contextlib
import warnings
from six import string_types
from scipy.special import legendre
from scipy.integrate import simps

from pyRSD.rsd._cache import Cache, parameter, interpolated_function, cached_property
from pyRSD.rsd import cosmology, tools, INTERP_KMIN, INTERP_KMAX, __version__
from pyRSD import pygcl, numpy as np, data as sim_data, os

from pyRSD.rsd.pt_integrals import PTIntegralsMixin
from pyRSD.rsd.sim_loader import SimLoaderMixin
from pyRSD.rsd.simulation import SimulationPdv, SimulationP11
from pyRSD.rsd.hzpt import InterpolatedHZPTModels

class DarkMatterSpectrum(Cache, SimLoaderMixin, PTIntegralsMixin):
    """
    The dark matter power spectrum in redshift space
    """
    # splines and interpolation variables
    k_interp = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 250)
    spline = tools.RSDSpline
    spline_kwargs = {'bounds_error' : True, 'fill_value' : 0}

    def __init__(self, kmin=1e-3,
                       kmax=0.5,
                       Nk=200,
                       z=0.,
                       params=cosmology.Planck15,
                       include_2loop=False,
                       transfer_fit="CLASS",
                       max_mu=4,
                       interpolate=False,
                       k0_low=5e-3,
                       linear_power_file=None,
                       Pdv_model_type='jennings',
                       **kwargs):
        """
        Parameters
        ----------
        kmin : float, optional
            The minimum wavenumber to compute the power spectrum at [units: `h/Mpc`]

        kmax : float, optional
            The maximum wavenumber to compute the power spectrum at [units: `h/Mpc`]

        Nk : int, optional
            The number of log-spaced bins to use as the underlying domain for splines

        z : float, optional
            The redshift to compute the power spectrum at. Default = 0.

        params : pyRSD.cosmology.Cosmology, str
            Either a Cosmology instance or the name of a file to load
            parameters from; see the 'data/params' directory for examples

        include_2loop : bool, optional
            If `True`, include 2-loop contributions in the model terms. Default
            is `False`.

        transfer_fit : str, optional
            The name of the transfer function fit to use. Default is `CLASS`
            and the options are {`CLASS`, `EH`, `EH_NoWiggle`, `BBKS`},
            or the name of a data file holding (k, T(k))

        max_mu : {0, 2, 4, 6, 8}, optional
            Only compute angular terms up to mu**(``max_mu``). Default is 4.

        interpolate: bool, optional
            Whether to return interpolated results for underlying power moments

        k0_low : float, optional (`5e-3`)
            below this wavenumber, evaluate any power in "low-k mode", which
            essentially just uses SPT at low-k

        linear_power_file : str, optional (`None`)
            string specifying the name of a file which gives the linear
            power spectrum, from which the transfer function in ``cosmo``
            will be initialized
        """
        # overload cosmo with a cosmo_filename kwargs to handle deprecated syntax
        if 'cosmo_filename' in kwargs:
            params = kwargs.pop('cosmo_filename')

        # set and save the model version automatically
        self.__version__ = __version__

        # mix in the sim loader class
        SimLoaderMixin.__init__(self)

        # set the input parameters
        self.interpolate       = interpolate
        self.transfer_fit      = transfer_fit
        self.params            = params
        self.max_mu            = max_mu
        self.include_2loop     = include_2loop
        self.z                 = z
        self.kmin              = kmin
        self.kmax              = kmax
        self.Nk                = Nk
        self.k0_low            = k0_low
        self.linear_power_file = linear_power_file
        self.Pdv_model_type    = Pdv_model_type

        # initialize the cosmology parameters and set defaults
        self.sigma8_z          = self.cosmo.Sigma8_z(self.z)
        self.f                 = self.cosmo.f_z(self.z)
        self.alpha_par         = 1.
        self.alpha_perp        = 1.
        self.sigma_v2          = 0.
        self.sigma_bv2         = 0.
        self.sigma_bv4         = 0.

        # set the models we want to use
        # default is to use all models
        for kw in self.allowable_kwargs:
            if fnmatch.fnmatch(kw, 'use_*_model'):
                setattr(self, kw, kwargs.pop(kw, True))

        # extra keywords
        if len(kwargs):
            for k in kwargs:
                warnings.warn("extra keyword `%s` is ignored" %k)

        # mix in the PT intergrals mixin
        PTIntegralsMixin.__init__(self)

    def __getstate__(self):
        """
        Custom pickling that removes `lru_cache` objects from
        the cache, which will ensure pickling succeeds
        """
        d = self.__dict__
        for k in list(self._cache):
            if hasattr(self._cache[k], 'cache_info'):
                d['_cache'].pop(k)

        return d

    def initialize(self):
        """
        Initialize the underlying splines, etc of the model
        """
        k = 0.5*(self.kmin+self.kmax)
        return self.power(k, 0.5)

    @contextlib.contextmanager
    def use_cache(self):
        """
        Cache repeated calls to functions defined in this class, assuming
        constant `k` and `mu` input values
        """
        from pyRSD.rsd.tools import cache_on, cache_off

        try:
            cache_on()
            yield
        except:
            raise
        finally:
            cache_off()

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @contextlib.contextmanager
    def preserve(self, **kwargs):
        """
        Context manager that preserves the state of the model
        upon exiting the context by first saving and then restoring it
        """
        # save the current state of the model
        set_params = {}; unset_params = []
        for k in self._param_names:
            if '__'+k in self.__dict__:
                set_params[k] = getattr(self, k)
            else:
                unset_params.append(k)
        cache = self._cache.copy()

        # current model params
        model_params = {}
        for k in self.allowable_kwargs:
            model_params[k] = getattr(self, k)

        # set any kwargs passed
        for k in kwargs:
            if k not in self.allowable_kwargs:
                raise ValueError("keywords to this function must be in `allowable_kwargs`")
            setattr(self, k, kwargs[k])

        yield

        # restore model params
        for k in model_params:
            setattr(self, k, model_params[k])

        # restore the model to previous state
        for k in set_params:
            setattr(self, k, set_params[k])
        for k in unset_params:
            if '__'+k in self.__dict__:
                delattr(self, k)
        for k in cache:
            self._cache[k] = cache[k]

    @contextlib.contextmanager
    def use_spt(self):
        """
        Context manager to turn off all models
        """
        from collections import OrderedDict
        params = OrderedDict()
        for kw in self.allowable_kwargs:
            if fnmatch.fnmatch(kw, 'use_*_model'):
                params[kw] = False

        try:

            # save the original state
            state = {k:getattr(self, k, None) for k in params}

            # update the state to low-k mode
            for k,v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            yield
        except:
            pass
        finally:
            # restore the original state
            for k, v in state.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    @contextlib.contextmanager
    def load_dm_sims(self, val):
        """
        Context manager to load simulation data for certain dark matter terms
        """
        allowed = ['teppei_lowz', 'teppei_midz', 'teppei_highz']
        if val not in allowed:
            raise ValueError("Allowed simulations to load are %s" %allowed)

        z_tags = {'teppei_lowz' : '000', 'teppei_midz' : '509', 'teppei_highz' : '989'}
        z_tag = z_tags[val]

        # get the data
        P00_mu0_data = getattr(sim_data, 'P00_mu0_z_0_%s' %z_tag)()
        P01_mu2_data = getattr(sim_data, 'P01_mu2_z_0_%s' %z_tag)()
        P11_mu4_data = getattr(sim_data, 'P11_mu4_z_0_%s' %z_tag)()
        Pdv_mu0_data = getattr(sim_data, 'Pdv_mu0_z_0_%s' %z_tag)()

        self._load('P00_mu0', P00_mu0_data[:,0], P00_mu0_data[:,1])
        self._load('P01_mu2', P01_mu2_data[:,0], P01_mu2_data[:,1])
        self._load('P11_mu4', P11_mu4_data[:,0], P11_mu4_data[:,1])
        self._load('Pdv', Pdv_mu0_data[:,0], Pdv_mu0_data[:,1])

        yield

        self._unload('P00_mu0')
        self._unload('P01_mu2')
        self._unload('P11_mu4')
        self._unload('Pdv')

    @parameter
    def interpolate(self, val):
        """
        Whether we want to interpolate any underlying models
        """
        self._update_models('interpolate', ['hzpt'], val)
        return val

    @parameter
    def transfer_fit(self, val):
        """
        The transfer function fitting method
        """
        allowed = ['CLASS', 'EH', 'EH_NoWiggle', 'BBKS']
        if val not in allowed:
            raise ValueError("`transfer_fit` must be one of %s" %allowed)
        return val

    @parameter
    def cosmo_filename(self, val):
        """
        The cosmology filename; this is deprecated, use attr:`params` instead
        """
        self.params = val
        return val

    @parameter
    def params(self, val, default="cosmo_filename"):
        """
        The cosmology parameters by the user
        """
        # check cosmology object
        if val is None:
            val = cosmology.Planck15
        if not isinstance(val, (string_types, cosmology.Cosmology)):
            raise TypeError(("input `cosmo` keyword should be a parameter file name"
                             "or a pyRSD.cosmology.Cosmology object"))
        return val

    @parameter
    def linear_power_file(self, val):
        """
        The name of the file holding the cosmological parameters
        """
        return val

    @parameter
    def max_mu(self, val):
        """
        Only compute the power terms up to and including `max_mu`. Should
        be one of [0, 2, 4, 6]
        """
        allowed = [0, 2, 4, 6]
        if val not in allowed:
            raise ValueError("`max_mu` must be one of %s" %allowed)
        return val

    @parameter
    def include_2loop(self, val):
        """
        Whether to include 2-loop terms in the power spectrum calculation
        """
        return val

    @parameter
    def z(self, val):
        """
        Redshift to evaluate power spectrum at
        """
        # update the dependencies
        models = ['P11_sim_model', 'Pdv_sim_model']
        self._update_models('z', models, val)

        return val

    @parameter
    def k0_low(self, val):
        """
        Wavenumber to transition to use only SPT for lower wavenumber
        """
        if val < 5e-4:
            raise ValueError("`k0_low` must be greater than 5e-4 h/Mpc")
        return val

    @parameter
    def kmin(self, val):
        """
        Minimum observed wavenumber needed for results
        """
        if val < INTERP_KMIN:
            raise ValueError("cannot compute model below %.2e h/Mpc due to PT integrals")
        return val

    @parameter
    def kmax(self, val):
        """
        Maximum observed wavenumber needed for results
        """
        if val > INTERP_KMAX:
            raise ValueError("cannot compute model above %.2f h/Mpc due to PT integrals")
        return val

    @parameter
    def Nk(self, val):
        """
        Number of log-spaced wavenumber bins to use in underlying splines
        """
        return val

    @parameter
    def sigma8_z(self, val):
        """
        The value of Sigma8(z) (mass variances within 8 Mpc/h at z) to compute
        the power spectrum at, which gives the normalization of the
        linear power spectrum
        """
        # update the dependencies
        models = ['hzpt', 'P11_sim_model', 'Pdv_sim_model']
        self._update_models('sigma8_z', models, val)

        return val

    @parameter
    def f(self, val):
        """
        The growth rate, defined as the `dlnD/dlna`.
        """
        # update the dependencies
        models = ['hzpt', 'Pdv_sim_model', 'P11_sim_model']
        self._update_models('f', models, val)

        return val

    @parameter
    def alpha_perp(self, val):
        """
        The perpendicular Alcock-Paczynski effect scaling parameter, where
        :math: `k_{perp, true} = k_{perp, true} / alpha_{perp}`
        """
        return val

    @parameter
    def alpha_par(self, val):
        """
        The parallel Alcock-Paczynski effect scaling parameter, where
        :math: `k_{par, true} = k_{par, true} / alpha_{par}`
        """
        return val

    @parameter(default=1.0)
    def alpha_drag(self, val):
        """
        The ratio of the sound horizon in the fiducial cosmology
        to the true cosmology
        """
        return val

    @parameter(default="sigma_lin")
    def sigma_v(self, val):
        """
        The velocity dispersion at `z`. If not provided, defaults to the
        linear theory prediction (as given by `self.sigma_lin`) [units: Mpc/h]
        """
        return val

    @parameter
    def sigma_v2(self, val):
        """
        The additional, small-scale velocity dispersion, evaluated using the
        halo model and weighted by the velocity squared. [units: km/s]

        .. math:: (sigma_{v2})^2 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} v_{\parallel}^2
        """
        return val

    @parameter
    def sigma_bv2(self, val):
        """
        The additional, small-scale velocity dispersion, evaluated using the
        halo model and weighted by the bias times velocity squared. [units: km/s]

        .. math:: (sigma_{bv2})^2 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} b(M) v_{\parallel}^2
        """
        return val

    @parameter
    def sigma_bv4(self, val):
        """
        The additional, small-scale velocity dispersion, evaluated using the
        halo model and weighted by bias times the velocity squared. [units: km/s]

        .. math:: (sigma_{bv4})^4 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} b(M) v_{\parallel}^4
        """
        return val

    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("cosmo")
    def fiducial_rs_drag(self):
        """
        The sound horizon at the drag redshift in the cosmology of ``cosmo``
        """
        return self.cosmo.rs_drag()

    @cached_property("transfer_fit")
    def transfer_fit_int(self):
        """
        The integer value representing the transfer function fitting method
        """
        return getattr(pygcl.transfers, self.transfer_fit)

    @cached_property("params", "transfer_fit", "linear_power_file")
    def cosmo(self):
        """
        A `pygcl.Cosmology` object holding the cosmological parameters
        """
        # convert from cosmology.Cosmology to pygcl.Cosmology
        if isinstance(self.params, cosmology.Cosmology):
            kws = {'transfer':self.transfer_fit_int, 'linear_power_file':self.linear_power_file}
            return self.params.to_class(**kws)

        # convert string to pygcl.Cosmology
        if self.linear_power_file is not None:
            k, Pk = np.loadtxt(self.linear_power_file, unpack=True)
            return pygcl.Cosmology.from_power(self.params, k, Pk)
        else:
            return pygcl.Cosmology(self.params, self.transfer_fit_int)

    @cached_property("Nk", "kmin", "kmax")
    def k(self):
        """
        Return a range in wavenumbers, set by the minimum/maximum
        allowed values, given the desired `kmin` and `kmax` and the
        current values of the AP effect parameters, `alpha_perp` and `alpha_par`
        """
        kmin = min(0.005, 0.9*self.kmin)
        if kmin < INTERP_KMIN: kmin = INTERP_KMIN
        kmax = max(0.9, 1.1*self.kmax)
        if kmax > INTERP_KMAX: kmax = INTERP_KMAX
        return np.logspace(np.log10(kmin), np.log10(kmax), self.Nk)

    @cached_property("z", "cosmo")
    def D(self):
        """
        The growth function at z
        """
        return self.cosmo.D_z(self.z)

    @cached_property("z", "cosmo")
    def conformalH(self):
        """
        The conformal Hubble parameter, defined as `H(z) / (1 + z)`
        """
        return self.cosmo.H_z(self.z) / (1. + self.z)

    @cached_property("cosmo")
    def power_lin(self):
        """
        A 'pygcl.LinearPS' object holding the linear power spectrum at z = 0
        """
        return pygcl.LinearPS(self.cosmo, 0.)

    @cached_property("cosmo")
    def power_lin_nw(self):
        """
        A 'pygcl.LinearPS' object holding the linear power spectrum at z = 0,
        using the Eisenstein-Hu no-wiggle transfer function
        """
        cosmo = self.cosmo.clone(tf=pygcl.transfers.EH_NoWiggle)
        return pygcl.LinearPS(cosmo, 0.)

    @cached_property("sigma8_z", "cosmo")
    def _power_norm(self):
        """
        The factor needed to normalize the linear power spectrum
        in `power_lin` to the desired sigma_8, as specified by `sigma8_z`,
        and the desired redshift `z`
        """
        return (self.sigma8_z / self.cosmo.sigma8())**2

    @cached_property("_power_norm", "_sigma_lin_unnormed")
    def sigma_lin(self):
        """
        The dark matter velocity dispersion at z, as evaluated in
        linear theory [units: Mpc/h]. Normalized to `sigma8_z`
        """
        return self._power_norm**0.5 * self._sigma_lin_unnormed

    @cached_property("power_lin")
    def _sigma_lin_unnormed(self):
        """
        The dark matter velocity dispersion at z = 0, as evaluated in
        linear theory [units: Mpc/h]. This is not properly normalized
        """
        return np.sqrt(self.power_lin.VelocityDispersion())

    #---------------------------------------------------------------------------
    # models to use
    #---------------------------------------------------------------------------
    @parameter
    def use_P00_model(self, val):
        """
        If `True`, use the `HZPT` model for P00
        """
        return val

    @parameter
    def use_P01_model(self, val):
        """
        If `True`, use the `HZPT` model for P01
        """
        return val

    @parameter
    def use_P11_model(self, val):
        """
        If `True`, use the `HZPT` model for P11
        """
        return val

    @parameter
    def use_Pdv_model(self, val):
        """
        Whether to use interpolated sim results for Pdv
        """
        return val

    @parameter
    def use_Pvv_model(self, val):
        """
        Whether to use Jenning model for Pvv or SPT
        """
        return val

    @parameter
    def Pdv_model_type(self, val):
        """
        Either `jennings` or `sims` to describe the Pdv model
        """
        allowed = ['jennings', 'sims']
        if val not in allowed:
            raise ValueError("`Pdv_model_type` must be one of %s" %str(allowed))
        return val

    @cached_property("cosmo")
    def hzpt(self):
        """
        The class holding the (possibly interpolated) HZPT models
        """
        kw = {'interpolate':self.interpolate}
        return InterpolatedHZPTModels(self.cosmo, self.sigma8_z, self.f, **kw)

    @cached_property("power_lin")
    def P11_sim_model(self):
        """
        The class holding the model for the P11 dark matter term, based
        on interpolating simulations
        """
        return SimulationP11(self.power_lin, self.z, self.sigma8_z, self.f)

    @cached_property("power_lin")
    def Pdv_sim_model(self):
        """
        The class holding the model for the Pdv dark matter term, based
        on interpolating simulations
        """
        return SimulationPdv(self.power_lin, self.z, self.sigma8_z, self.f)

    def Pdv_jennings(self, k):
        """
        Return the density-divergence cross spectrum using the fitting
        formula from Jennings et al 2012 (arxiv: 1207.1439)
        """
        a0 = -12483.8; a1 = 2.554; a2 = 1381.29; a3 = 2.540
        D = self.D
        s8 = self.sigma8_z / D

        # z = 0 results
        self.hzpt._P00.sigma8_z = s8
        P00_z0 = self.hzpt.P00(k)

        # redshift scaling
        z_scaling = 3. / (D + D**2 + D**3)

        # reset sigma8_z
        self.hzpt._P00.sigma8_z = self.sigma8_z
        g = (a0 * P00_z0**0.5 + a1 * P00_z0**2) / (a2 + a3 * P00_z0)
        toret = (g - P00_z0) / z_scaling**2 + self.hzpt.P00(k)
        return - self.f * toret

    def Pvv_jennings(self, k):
        """
        Return the divergence auto spectrum using the fitting
        formula from Jennings et al 2012 (arxiv: 1207.1439)
        """
        a0 = -12480.5; a1 = 1.824; a2 = 2165.87; a3 = 1.796
        D = self.D
        s8 = self.sigma8_z / D

        # z = 0 results
        self.hzpt._P00.sigma8_z = s8
        P00_z0 = self.hzpt.P00(k)

        # redshift scaling
        z_scaling = 3. / (D + D**2 + D**3)

        # reset sigma8_z
        self.hzpt._P00.sigma8_z = self.sigma8_z
        g = (a0 * P00_z0**0.5 + a1 * P00_z0**2) / (a2 + a3 * P00_z0)
        toret = (g - P00_z0) / z_scaling**2 + self.hzpt.P00(k)
        return self.f**2 * toret

    def to_npy(self, filename):
        """
        Save to a ``.npy`` file by calling :func:`numpy.save`
        """
        np.save(filename, self)

    @classmethod
    def from_npy(cls, filename):
        """
        Load a model from a ``.npy`` file
        """
        from pyRSD.rsd import load_model
        return load_model(filename)

    #---------------------------------------------------------------------------
    # utility functions
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Update the attributes. Checks that the current value is not equal to
        the new value before setting.
        """
        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except Exception as e:
                raise RuntimeError("failure to set parameter `%s` to value %s: %s" %(k, str(v), str(e)))

    @property
    def config(self):
        """
        Return a dictionary holding the model configuration

        This holds the value of all attributes in the
        :attr:`allowable_kwargs` list
        """
        allowed = self.__class__.allowable_kwargs
        return {k:getattr(self, k) for k in allowed}

    def _update_models(self, name, models, val):
        """
        Update the specified attribute for the models given
        """
        for model in models:
            if model in self._cache:
                setattr(getattr(self, model), name, val)

    #---------------------------------------------------------------------------
    # power term attributes
    #---------------------------------------------------------------------------
    def normed_power_lin(self, k):
        """
        The linear power evaluated at the specified `k` and at `z`,
        normalized to `sigma8_z`
        """
        return self._power_norm * self.power_lin(k)

    def normed_power_lin_nw(self, k):
        """
        The Eisenstein-Hu no-wiggle, linear power evaluated at the specified
        `k` and at `self.z`, normalized to `sigma8_z`
        """
        return self._power_norm * self.power_lin_nw(k)

    @interpolated_function("P00", "k", interp="k")
    def P_mu0(self, k):
        """
        The full power spectrum term with no angular dependence. Contributions
        from P00.
        """
        return self.P00.mu0(k)

    @interpolated_function("P01", "P11", "P02", "k", interp="k")
    def P_mu2(self, k):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        return self.P01.mu2(k) + self.P11.mu2(k) + self.P02.mu2(k)

    @interpolated_function("P11", "P02", "P12", "P22", "P03", "P13",
                           "P04", "include_2loop", "k", interp="k")
    def P_mu4(self, k):
        """
        The full power spectrum term with mu^4 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13 (2-loop), and P04 (2-loop).
        """
        Pk = self.P11.mu4(k) + self.P02.mu4(k) + self.P12.mu4(k) + self.P22.mu4(k) + self.P03.mu4(k)
        if self.include_2loop: Pk += self.P13.mu4(k) + self.P04.mu4(k)
        return Pk

    @interpolated_function("P12", "P22", "P13", "P04", "include_2loop", "k", interp="k")
    def P_mu6(self, k):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P12, P22, P13, and P04 (2-loop).
        """
        Pk = self.P12.mu6(k) + self.P22.mu6(k) + self.P13.mu6(k)
        if self.include_2loop: Pk += self.P04.mu6(k)
        return Pk

    @interpolated_function("z", "_power_norm", "k", interp="k")
    def Pdd(self, k):
        """
        The 1-loop auto-correlation of density.
        """
        norm = self._power_norm
        return norm*(self.power_lin(k) + norm*self._Pdd_0(k))

    @interpolated_function("f", "z", "_power_norm", "use_Pdv_model", "Pdv_model_type", "Pdv_loaded", "k", interp="k")
    def Pdv(self, k):
        """
        The 1-loop cross-correlation between dark matter density and velocity
        divergence.
        """
        # check for any user-loaded values
        if self.Pdv_loaded:
            return self._get_loaded_data('Pdv', k)
        else:
            if self.use_Pdv_model:
                if self.Pdv_model_type == 'jennings':
                    return self.Pdv_jennings(k)
                elif self.Pdv_model_type == 'sims':
                    return self.Pdv_sim_model(k)
            else:
                norm = self._power_norm
                return (-self.f)*norm*(self.power_lin(k) + norm*self._Pdv_0(k))

    @interpolated_function("f", "z", "_power_norm", "use_Pvv_model", "k", interp="k")
    def Pvv(self, k):
        """
        The 1-loop auto-correlation of velocity divergence.
        """
        if self.use_Pvv_model:
            return self.Pvv_jennings(k)
        else:
            norm = self._power_norm
            return self.f**2 * norm*(self.power_lin(k) + norm*self._Pvv_0(k))

    @cached_property("z", "sigma8_z", "use_P00_model", "power_lin", "P00_mu0_loaded")
    def P00(self):
        """
        The isotropic, zero-order term in the power expansion, corresponding
        to the density field auto-correlation. No angular dependence.
        """
        from .P00 import P00PowerTerm
        return P00PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "use_P01_model", "power_lin",
                     "max_mu", "P01_mu2_loaded")
    def P01(self):
        """
        The correlation of density and momentum density, which contributes
        mu^2 terms to the power expansion.
        """
        from .P01 import P01PowerTerm
        return P01PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "use_P11_model", "power_lin",
                     "max_mu", "include_2loop", "P11_mu2_loaded", "P11_mu4_loaded")
    def P11(self):
        """
        The auto-correlation of momentum density, which has a scalar portion
        which contributes mu^4 terms and a vector term which contributes
        mu^2*(1-mu^2) terms to the power expansion. This is the last term to
        contain a linear contribution.
        """
        from .P11 import P11PowerTerm
        return P11PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "power_lin", "max_mu",
                     "include_2loop", "sigma_v", "sigma_bv2", "P00")
    def P02(self):
        """
        The correlation of density and energy density, which contributes
        mu^2 and mu^4 terms to the power expansion. There are no
        linear contributions here.
        """
        from .P02 import P02PowerTerm
        return P02PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "power_lin", "max_mu",
                     "include_2loop", "sigma_v", "sigma_bv2", "P01")
    def P12(self):
        """
        The correlation of momentum density and energy density, which contributes
        mu^4 and mu^6 terms to the power expansion. There are no linear
        contributions here. Two-loop contribution uses the mu^2 contribution
        from the P01 term.
        """
        from .P12 import P12PowerTerm
        return P12PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "power_lin", "max_mu",
                     "include_2loop", "sigma_v", "sigma_bv2", "P00", "P02")
    def P22(self):
        """
        The autocorelation of energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear
        contributions here.
        """
        from .P22 import P22PowerTerm
        return P22PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "power_lin", "max_mu",
                     "include_2loop", "sigma_v", "sigma_v2", "P01")
    def P03(self):
        """
        The cross-corelation of density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^4 terms.
        """
        from .P03 import P03PowerTerm
        return P03PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "power_lin", "max_mu",
                     "include_2loop", "sigma_v", "sigma_bv2", "sigma_v2", "P11")
    def P13(self):
        """
        The cross-correlation of momentum density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^6 terms at 1-loop order and
        mu^4 terms at 2-loop order.
        """
        from .P13 import P13PowerTerm
        return P13PowerTerm(self)

    @cached_property("f",  "z", "sigma8_z", "power_lin", "max_mu",
                     "include_2loop", "sigma_v", "sigma_bv4", "P02", "P00")
    def P04(self):
        """
        The cross-correlation of density with the rank four tensor field
        ((1+delta)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        from .P04 import P04PowerTerm
        return P04PowerTerm(self)

    #---------------------------------------------------------------------------
    # main user callables
    #---------------------------------------------------------------------------
    @tools.broadcast_kmu
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
        # the return array
        pkmu = self._power(k, mu)
        if flatten: pkmu = np.ravel(pkmu, order='F')
        return pkmu

    def poles(self, k, poles, flatten=False, Nmu=41):
        """
        Return the multipole moments specified by `poles`, where `poles` is a
        list of integers, i.e., [0, 2, 4]

        Parameter
        ---------
        k : float, array_like
            The wavenumbers to evaluate the power spectrum at, in `h/Mpc`
        poles : init, array_like
            The `ell` values of the multipole moments
        flatten : bool, optional
            If `True`, flatten the return array, which will have a length of
            `len(k) * len(poles)`

        Returns
        -------
        poles : array_like
            returns tuples of arrays for each ell value in ``poles``
        """
        scalar = np.isscalar(poles)
        if scalar: poles = [poles]

        if len(k) == Nmu: Nmu += 1
        mus = np.linspace(0., 1., Nmu)
        Pkmus = self.power(k, mus)

        if len(poles) != len(k):
            toret = ()
            for ell in poles:
                kern = (2*ell+1.)*legendre(ell)(mus)
                val = np.array([simps(kern*d, x=mus) for d in Pkmus])
                toret += (val,)

            if scalar:
                return toret[0]
            else:
                return toret if not flatten else np.ravel(toret, order='F')
        else:
            kern = np.asarray([(2*ell+1.)*legendre(ell)(mus) for ell in poles])
            return np.array([simps(d, x=mus) for d in kern*Pkmus])

    def from_transfer(self, transfer, flatten=False, **kws):
        """
        Return the power (either P(k,mu) or multipoles), accounting for
        discrete binning effects using the input transfer function

        This calls :func:`power` to evaluate the model at discrete ``k``
        and ``mu`` values

        Parameter
        ---------
        transfer : PkmuTransfer, PolesTransfer, WindowTransfer
            the transfer class which accounts for the discrete binning effects
        flatten : bool, optional
            If `True`, flatten the return array, which will have a length of
            `Nk * Nmu` or `Nk * Nell`
        """
        grid = transfer.grid

        # check some bounds for window convolution
        from pyRSD.rsd.window import WindowTransfer
        if isinstance(transfer, WindowTransfer):

            # check k-range inconsistency
            kmin, kmax = transfer.kmin, transfer.kmax
            if (kmin < self.kmin).any():
                warnings.warn("min k of window transfer (%s) is less than model's kmin (%.2e)" %(kmin, self.kmin))
            if (kmax > self.kmax).any():
                warnings.warn("max k of window transfer (%s) is greater than model's kmax (%.2e)" %(kmax, self.kmax))

            # check bad values
            if self.kmin > 5e-3:
                warnings.warn("doing window convolution with dangerous model kmin (%.2e)" %self.kmin)
            if self.kmax < 0.5:
                warnings.warn("doing window convolution with dangerous model kmax (%.2e)" %self.kmax)

        power = self.power(grid.k[grid.notnull], grid.mu[grid.notnull])
        transfer.power = power
        return transfer(flatten=flatten, **kws)

    @tools.alcock_paczynski
    def _P_mu0(self, k, mu):
        """
        Return the AP-distorted P[mu^0]
        """
        return self.P_mu0(k)

    @tools.alcock_paczynski
    def _P_mu2(self, k, mu):
        """
        Return the AP-distorted mu^2 P[mu^2]
        """
        return mu**2 * self.P_mu2(k)

    @tools.alcock_paczynski
    def _P_mu4(self, k, mu):
        """
        Return the AP-distorted mu^4 P[mu^4]
        """
        return mu**4 * self.P_mu4(k)

    @tools.alcock_paczynski
    def _P_mu6(self, k, mu):
        """
        Return the AP-distorted mu^6 P[mu^6]
        """
        return mu**6 * self.P_mu6(k)

    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_k(self, k, mu):
        """
        Return the derivative of :func:`power` with
        respect to `k`
        """
        toret = 0
        funcs = [self.P_mu0, self.P_mu2, self.P_mu4, self.P_mu6]

        i = 0
        while i <= (self.max_mu//2):
            toret += mu**(2*i) * funcs[i](k, derivative=True)
            i += 1

        return toret

    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_mu(self, k, mu):
        """
        Return the derivative of :func:`power` with
        respect to `mu`
        """
        toret = 0
        funcs = [self.P_mu0, self.P_mu2, self.P_mu4, self.P_mu6]

        i = 0
        while i <= (self.max_mu//2):

            # derivative of mu^(2i)
            if i != 0: toret += (2.*i) * mu**(2*i-1) * funcs[i](k)
            i += 1

        return toret

    def _power(self, k, mu):
        """
        Return the power as sum of mu powers
        """

        if self.max_mu > 6:
            raise NotImplementedError("cannot compute power spectrum including terms with order higher than mu^6")

        toret = 0
        funcs = [self._P_mu0, self._P_mu2, self._P_mu4, self._P_mu6]

        i = 0
        while i <= (self.max_mu//2):
            toret += funcs[i](k, mu)
            i += 1

        return np.nan_to_num(toret)

    @tools.monopole
    def monopole(self, k, mu, **kwargs):
        """
        The monopole moment of the power spectrum. Include mu terms up to
        mu**max_mu.
        """
        return self.power(k, mu, **kwargs)

    @tools.quadrupole
    def quadrupole(self, k, mu, **kwargs):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to
        mu**max_mu.
        """
        return self.power(k, mu, **kwargs)

    @tools.hexadecapole
    def hexadecapole(self, k, mu, **kwargs):
        """
        The hexadecapole moment of the power spectrum. Include mu terms up to
        mu**max_mu.
        """
        return self.power(k, mu, **kwargs)

    @tools.tetrahexadecapole
    def tetrahexadecapole(self, k, mu, **kwargs):
        """
        The tetrahexadecapole (ell=6) moment of the power spectrum
        """
        return self.power(k, mu, **kwargs)

class PowerTerm(object):
    """
    Class to hold the data for each term in the power expansion.
    """
    def __init__(self):

        # total angular dependences
        self.total = Angular()

        # initialize scalar/vector sub terms
        self.scalar = Angular()
        self.vector = Angular()

        # initialize with/without velocity dispersion sub terms
        self.no_velocity   = Angular()
        self.with_velocity = Angular()


class Angular(object):
    """
    Class to keep track of the different angular terms for each power term.
    """
    def __init__(self):
        self.mu0 = 0.
        self.mu2 = 0.
        self.mu4 = 0.
        self.mu6 = 0.
        self.mu8 = 0.
