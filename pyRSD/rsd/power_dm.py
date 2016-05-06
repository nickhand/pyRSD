import fnmatch

from ._cache import Cache, parameter, interpolated_property, cached_property
from . import tools, INTERP_KMIN, INTERP_KMAX
from .. import pygcl, numpy as np, data as sim_data, os

from ._integrals import Integrals
from ._sim_loader import SimLoader
from .simulation import SimulationPdv, SimulationP11
from .halo_zeldovich import HaloZeldovichP00, HaloZeldovichP01, HaloZeldovichP11

def verify_krange(k, kmin, kmax):
    if np.amin(k) < kmin:
        raise ValueError("cannot compute power spectrum for k < %.2e; adjust `kmin` parameter" %kmin)
    if np.amax(k) > kmax:
        raise ValueError("cannot compute power spectrum for k > %.2e; adjust `kmax` parameter" %kmax)

class DarkMatterSpectrum(Cache, SimLoader, Integrals):
    """
    The dark matter power spectrum in redshift space
    """
    __version__ = '0.1.0'
    
    # splines and interpolation variables
    k_interp = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 200)
    spline = tools.RSDSpline
    spline_kwargs = {'bounds_error' : True, 'fill_value' : 0}
    
    def __init__(self, kmin=1e-3,
                       kmax=0.5,
                       Nk=100,
                       z=0., 
                       cosmo_filename="planck1_WP.ini",
                       include_2loop=False,
                       transfer_fit="CLASS",
                       max_mu=4,
                       interpolate=True,
                       load_dm_sims=None,
                       k0_low=5e-3,
                       enhance_wiggles=True,
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
            
        cosmo_filename : str
            The cosmological parameters to use, specified as the name
            of the file holding the `CLASS` parameter file. Default 
            is `planck1_WP.ini`.
            
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
            
        load_dm_sims : str or `None`, optional
            load the DM simulation measurements for a set of builtin simulations;
            must be one of ``['teppei_lowz', 'teppei_midz', 'teppei_highz']``
        
        k0_low : float, optional (`5e-3`)
            below this wavenumber, evaluate any power in "low-k mode", which
            essentially just uses SPT at low-k
            
        enhance_wiggles : bool, optional (`True`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
            
        linear_power_file : str, optional (`None`)
            string specifying the name of a file which gives the linear 
            power spectrum, from which the transfer function in ``cosmo``
            will be initialized
        """   
        SimLoader.__init__(self)
        
        # set the input parameters
        self.interpolate       = interpolate
        self.transfer_fit      = transfer_fit
        self.cosmo_filename    = cosmo_filename
        self.max_mu            = max_mu
        self.include_2loop     = include_2loop
        self.z                 = z 
        self.load_dm_sims      = load_dm_sims
        self.kmin              = kmin
        self.kmax              = kmax
        self.Nk                = Nk
        self.k0_low            = k0_low
        self.enhance_wiggles   = enhance_wiggles
        self.linear_power_file = linear_power_file
        self.Pdv_model_type    = Pdv_model_type
        
        # initialize the cosmology parameters and set defaults
        self.sigma8_z          = self.cosmo.Sigma8_z(self.z)
        self.f                 = self.cosmo.f_z(self.z)
        self.alpha_par         = 1.
        self.alpha_perp        = 1.
        self.small_scale_sigma = 0.
        self.sigma_v           = self.sigma_lin
        self.sigma_v2          = 0.
        self.sigma_bv2         = 0.
        self.sigma_bv4         = 0.
        
        # set the models we want to use
        # default is to use all models
        for kw in self.allowable_kwargs:
            if fnmatch.fnmatch(kw, 'use_*_model'):
                setattr(self, kw, kwargs.pop(kw, True))
            
        # check the version        
        version = kwargs.pop('__version__', self.__version__)
        if self.__version__ != version:
            msg = "trying to initialize a model of the wrong version:\n"
            msg += '\tcurrent model version: %s\n' %(self.__version__)
            msg += '\tdesired model version: %s\n' %(version)
            raise ValueError(msg)
        
        # extra keywords
        if len(kwargs):                    
            for k in kwargs:
                print "warning: extra keyword `%s` is ignored" %k
        
        # finally, initialize the integrals    
        Integrals.__init__(self)
                    
    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def load_dm_sims(self, val):
        """
        Load simulation data for the dark matter terms
        """
        # try to unload any load sim data
        if val is None:
            names = ['P00_mu0', 'P01_mu2', 'P11_mu4', 'Pdv']
            for name in names:
                if getattr(self, name+'_loaded'):
                    self.unload(name)
        
        else:
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
            
            self.load('P00_mu0', P00_mu0_data[:,0], P00_mu0_data[:,1])
            self.load('P01_mu2', P01_mu2_data[:,0], P01_mu2_data[:,1])
            self.load('P11_mu4', P11_mu4_data[:,0], P11_mu4_data[:,1])
            self.load('Pdv', Pdv_mu0_data[:,0], Pdv_mu0_data[:,1])
            
        return val
        
    @parameter
    def interpolate(self, val):
        """
        Whether we want to interpolate any underlying models
        """
        # set the dependencies
        models = ['P00_hzpt_model', 'P01_hzpt_model']
        self._update_models('interpolate', models, val)
        
        return val
        
    @parameter
    def enhance_wiggles(self, val):
        """
        Whether to enhance the wiggles over the default HZPT model
        """
        # set the dependencies
        models = ['P00_hzpt_model', 'P01_hzpt_model']
        self._update_models('enhance_wiggles', models, val)
        
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
        The name of the file holding the cosmological parameters
        """
        return val
        
    @parameter
    def linear_power_file(self, val):
        """
        The name of the file holding the cosmological parameters
        """
        if val is not None:
            if not os.path.exists(val):
                raise ValueError("the specified file `%s` for `linear_power_file` does not exist" %val)
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
        return val
        
    @parameter
    def kmax(self, val):
        """
        Minimum observed wavenumber needed for results
        """
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
        models = ['P00_hzpt_model', 'P01_hzpt_model', 'P11_hzpt_model', 
                    'P11_sim_model', 'Pdv_sim_model']
        self._update_models('sigma8_z', models, val)
        
        return val
        
    @parameter
    def f(self, val):
        """
        The growth rate, defined as the `dlnD/dlna`.
        """
        # update the dependencies
        models = ['P01_hzpt_model', 'P11_hzpt_model', 'Pdv_sim_model', 'P11_sim_model']
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
            
    @parameter
    def sigma_v(self, val):
        """
        The velocity dispersion at z = 0. If not provided, defaults to the 
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
    @cached_property("transfer_fit")
    def transfer_fit_int(self):
        """
        The integer value representing the transfer function fitting method
        """
        return getattr(pygcl.Cosmology, self.transfer_fit)
                        
    @cached_property("cosmo_filename", "transfer_fit", "linear_power_file")
    def cosmo(self):
        """
        A `pygcl.Cosmology` object holding the cosmological parameters
        """
        if self.linear_power_file is not None:
            return pygcl.Cosmology.from_power(self.cosmo_filename, self.linear_power_file)
        else:
            return pygcl.Cosmology(self.cosmo_filename, self.transfer_fit_int)
                      
    @cached_property("alpha_perp", "alpha_par", "kmin", "kmax", "Nk")
    def k(self):
        """
        Return a range in wavenumbers, set by the minimum/maximum
        allowed values, given the desired `kmin` and `kmax` and the
        current values of the AP effect parameters, `alpha_perp` and `alpha_par`
        """
        kmin = min(self.k_true(self.kmin, 0.), self.k_true(self.kmin, 1.))
        kmax = max(self.k_true(self.kmax, 0.), self.k_true(self.kmax, 1.))

        if kmin < INTERP_KMIN:
            msg = "Minimum possible k value with this `kmin` and "
            msg += "`alpha_par`, `alpha_perp` values is below the minimum "
            msg += "value used for interpolation purposes; exiting before "
            msg += "bad things happen."
            raise ValueError(msg)

        if kmax > INTERP_KMAX:
            msg = "Maximum possible k value with this `k_obs` and "
            msg += "`alpha_par`, `alpha_perp` values is above the maximum "
            msg += "value used for interpolation purposes; exiting before "
            msg += "bad things happen."
            raise ValueError(msg)

        return np.logspace(np.log10(kmin), np.log10(kmax), self.Nk)
        
    @cached_property("z", "cosmo")
    def D(self):
        """
        The growth function at z
        """
        return self.cosmo.D_z(self.z)
        
    @cached_property("cosmo")
    def _cosmo_sigma8(self):
        """
        The sigma8 value from the cosmology
        """
        return self.cosmo.sigma8()
    
    @cached_property("cosmo", "z")
    def _cosmo_sigma8_z(self):
        """
        The sigma8(z) value from the cosmology
        """
        return self.cosmo.Sigma8_z(self.z)
    
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
            
    @cached_property("cosmo_filename")
    def power_lin_nw(self):
        """
        A 'pygcl.LinearPS' object holding the linear power spectrum at z = 0, 
        using the Eisenstein-Hu no-wiggle transfer function
        """
        cosmo = pygcl.Cosmology(self.cosmo_filename, pygcl.Cosmology.EH_NoWiggle)
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
    def Pdv_model_type(self, val):
        """
        Either `jennings` or `sims` to describe the Pdv model
        """
        allowed = ['jennings', 'sims']
        if val not in allowed:
            raise ValueError("`Pdv_model_type` must be one of %s" %str(allowed))
        return val

    @cached_property("cosmo")
    def P00_hzpt_model(self):
        """
        The class holding the HZPT model for the P00 dark matter term
        """
        kw = {'interpolate':self.interpolate, 'enhance_wiggles':self.enhance_wiggles}
        return HaloZeldovichP00(self.cosmo, self.sigma8_z, **kw)
    
    @cached_property("cosmo")
    def P01_hzpt_model(self):
        """
        The class holding the HZPT model for the P01 dark matter term
        """
        kw = {'interpolate':self.interpolate, 'enhance_wiggles':self.enhance_wiggles}
        return HaloZeldovichP01(self.cosmo, self.sigma8_z, self.f, **kw)
    
    @cached_property("cosmo")
    def P11_hzpt_model(self):
        """
        The class holding the HZPT model for the P11 dark matter term
        """
        kw = {'interpolate':self.interpolate, 'enhance_wiggles':self.enhance_wiggles}
        return HaloZeldovichP11(self.cosmo, self.sigma8_z, self.f, **kw)
    
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
        self.P00_hzpt_model.sigma8_z = s8
        P00_z0 = self.P00_hzpt_model(k)
        
        # redshift scaling
        z_scaling = 3. / (D + D**2 + D**3)
        
        # reset sigma8_z
        self.P00_hzpt_model.sigma8_z = self.sigma8_z
        g = (a0 * P00_z0**0.5 + a1 * P00_z0**2) / (a2 + a3 * P00_z0)
        toret = (g - P00_z0) / z_scaling**2 + self.P00_hzpt_model(k)
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
        self.P00_hzpt_model.sigma8_z = s8
        P00_z0 = self.P00_hzpt_model(k)
        
        # redshift scaling
        z_scaling = 3. / (D + D**2 + D**3)
        
        # reset sigma8_z
        self.P00_hzpt_model.sigma8_z = self.sigma8_z
        g = (a0 * P00_z0**0.5 + a1 * P00_z0**2) / (a2 + a3 * P00_z0)
        toret = (g - P00_z0) / z_scaling**2 + self.P00_hzpt_model(k)
        return self.f**2 * toret
        
    @classmethod
    def from_npy(self, filename):
        """
        Load from a numpy `.npy` file
        """
        return np.load(filename).tolist()
        
    @classmethod
    def from_pickle(self, filename):
        """
        Load from a pickle file
        """
        return pickle.load(open(filename, 'r'), protocol=-1)
        
    def to_npy(self, filename):
        """
        Save to a numpy `.npy` file
        """
        np.save(filename, self)
        
    def from_pickle(self, filename):
        """
        Save to a pickle file. This is slower than `to_npy`, 
        so that is the preferred serialization method
        """
        pickle.dump(self, open(filename, 'w'), protocol=-1)
        
    #---------------------------------------------------------------------------
    # utility functions
    #---------------------------------------------------------------------------         
    def k_true(self, k_obs, mu_obs):
        """
        Return the `true` k values, given an observed (k, mu)
        """
        F = self.alpha_par / self.alpha_perp
        if (F != 1.):
            return (k_obs/self.alpha_perp)*(1 + mu_obs**2*(1./F**2 - 1))**(0.5)
        else:
            return k_obs/self.alpha_perp
                
    def mu_true(self, mu_obs):
        """
        Return the `true` mu values, given an observed mu
        """
        F = self.alpha_par / self.alpha_perp
        return (mu_obs/F) * (1 + mu_obs**2*(1./F**2 - 1))**(-0.5)
    
    def update(self, **kwargs):
        """
        Update the attributes. Checks that the current value is not equal to 
        the new value before setting.
        """
        for k, v in kwargs.iteritems():
            try:
                if hasattr(self, k) and getattr(self, k) == v: 
                    continue
                setattr(self, k, v)
            except Exception as e:
                raise RuntimeError("failure to set parameter `%s` to value %s: %s" %(k, str(v), str(e)))
    
    def to_dict(self):
        """
        Return a dictionary of the allowable parameters
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
        
    @interpolated_property("P00", interp="k")
    def P_mu0(self, k):
        """
        The full power spectrum term with no angular dependence. Contributions
        from P00.
        """
        return self.P00.total.mu0
        
    @interpolated_property("P01", "P11", "P02", interp="k")
    def P_mu2(self, k):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        return self.P01.total.mu2 + self.P11.total.mu2 + self.P02.total.mu2
    
    @interpolated_property("P11", "P02", "P12", "P22", "P03", "P13", 
                           "P04", "include_2loop", interp="k")
    def P_mu4(self, k):
        """
        The full power spectrum term with mu^4 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13 (2-loop), and P04 (2-loop).
        """
        Pk = self.P11.total.mu4 + self.P02.total.mu4 + self.P12.total.mu4 + self.P22.total.mu4 + self.P03.total.mu4
        if self.include_2loop: Pk += self.P13.total.mu4 + self.P04.total.mu4
        return Pk
        
    @interpolated_property("P12", "P22", "P13", "P04", "include_2loop", interp="k")
    def P_mu6(self, k):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P12, P22, P13, and P04 (2-loop).
        """
        Pk = self.P12.total.mu6 + self.P22.total.mu6 + self.P13.total.mu6
        if self.include_2loop: Pk += self.P04.total.mu6
        return Pk
            
    @cached_property("k", "z", "_power_norm")
    def Pdd(self):
        """
        The 1-loop auto-correlation of density.
        """
        norm = self._power_norm
        return norm*(self.power_lin(self.k) + norm*self._Pdd_0(self.k))
        
    @cached_property("k", "f", "z", "_power_norm", "use_Pdv_model", "Pdv_model_type", "Pdv_loaded")
    def Pdv(self):
        """
        The 1-loop cross-correlation between dark matter density and velocity 
        divergence.
        """
        # check for any user-loaded values
        if self.Pdv_loaded:
            return self.get_loaded_data('Pdv', self.k)
        else:
            if self.use_Pdv_model:
                if self.Pdv_model_type == 'jennings':
                    return self.Pdv_jennings(self.k)
                elif self.Pdv_model_type == 'sims':
                    return self.Pdv_sim_model(self.k)
            else:
                norm = self._power_norm
                return (-self.f)*norm*(self.power_lin(self.k) + norm*self._Pdv_0(self.k))
    
    @cached_property("k", "f", "z", "_power_norm")
    def Pvv(self):
        """
        The 1-loop auto-correlation of velocity divergence.
        """
        norm = self._power_norm
        return self.f**2 * norm*(self.power_lin(self.k) + norm*self._Pvv_0(self.k))
    
    @cached_property("k", "z", "sigma8_z", "use_P00_model", "power_lin", 
                     "P00_mu0_loaded", "enhance_wiggles")
    def P00(self):
        """
        The isotropic, zero-order term in the power expansion, corresponding
        to the density field auto-correlation. No angular dependence.
        """
        P00 = PowerTerm()
            
        # check and return any user-loaded values
        if self.P00_mu0_loaded:
            P00.total.mu0 = self.get_loaded_data('P00_mu0', self.k)
        else:
                
            # use the DM model
            if self.use_P00_model:
                P00.total.mu0 = self.P00_hzpt_model(self.k)
            # use pure PT
            else:
                # the necessary integrals 
                I00 = self.I00(self.k)
                J00 = self.J00(self.k)
        
                P11 = self.normed_power_lin(self.k)
                P22 = 2*I00
                P13 = 6*self.k**2*J00*P11
                P00.total.mu0 = P11 + P22 + P13
            
        return P00
            
    @cached_property("k", "f",  "z", "sigma8_z", "use_P01_model", "power_lin", 
                     "max_mu", "P01_mu2_loaded", "enhance_wiggles")
    def P01(self):
        """
        The correlation of density and momentum density, which contributes
        mu^2 terms to the power expansion.
        """
        P01 = PowerTerm()
        if self.max_mu >= 2:
            
            # check and return any user-loaded values
            if self.P01_mu2_loaded:
                P01.total.mu2 = self.get_loaded_data('P01_mu2', self.k)
            else:
            
                # use the DM model
                if self.use_P01_model:
                    P01.total.mu2 = self.P01_hzpt_model(self.k)
                # use pure PT
                else:                
                    # the necessary integrals 
                    I00 = self.I00(self.k)
                    J00 = self.J00(self.k)
        
                    Plin = self.normed_power_lin(self.k)
                    P01.total.mu2 = 2*self.f*(Plin + 4.*(I00 + 3*self.k**2*J00*Plin))
        
        return P01
        
    @cached_property("k", "f",  "z", "sigma8_z", "use_P11_model", "power_lin", 
                     "max_mu", "include_2loop", "P11_mu2_loaded", "P11_mu4_loaded")
    def P11(self):
        """
        The auto-correlation of momentum density, which has a scalar portion 
        which contributes mu^4 terms and a vector term which contributes
        mu^2*(1-mu^2) terms to the power expansion. This is the last term to
        contain a linear contribution.
        """
        P11 = PowerTerm()
        
        # do mu^2 terms?
        if self.max_mu >= 2:
            
            # check and return any user-loaded values
            if self.P11_mu2_loaded:
                Pvec = self.get_loaded_data('P11_mu2', self.k)
                P11.vector.mu2 = P11.total.mu2 = Pvec
                P11.vector.mu4 = -Pvec
            else:
                
                # do the vector part, contributing mu^2 and mu^4 terms
                if not self.include_2loop:
                    Pvec = self.f**2 * self.I31(self.k)
                else:
                    I1 = self.Ivvdd_h01(self.k)
                    I2 = self.Idvdv_h03(self.k)
                    Pvec = self.f**2 * (I1 + I2)
            
                # save the mu^2 vector term
                P11.vector.mu2 = P11.total.mu2 = Pvec
                P11.vector.mu4 = -Pvec
            
            # do mu^4 terms?
            if self.max_mu >= 4: 
                  
                # check and return any user-loaded values
                if self.P11_mu4_loaded:
                    P11.total.mu4 = self.get_loaded_data('P11_mu4', self.k)
                else:
                    
                    # use HZPT?
                    if self.use_P11_model:
                        P11.total.mu4 = self.P11_hzpt_model(self.k) - self.f**2 * self.I31(self.k)
                    else:
                        # compute the scalar mu^4 contribution
                        if self.include_2loop:
                            I1 = self.Ivvdd_h02(self.k)
                            I2 = self.Idvdv_h04(self.k)
                            C11_contrib = I1 + I2
                        else:
                            C11_contrib = self.I13(self.k)
                
                        # the necessary integrals 
                        I11 = self.I11(self.k)
                        I22 = self.I22(self.k)
                        J11 = self.J11(self.k)
                        J10 = self.J10(self.k)
                
                        Plin = self.normed_power_lin(self.k)
                        part2 = 2*I11 + 4*I22 + 6*self.k**2 * (J11 + 2*J10)*Plin
                        P_scalar = self.f**2 * (Plin + part2 + C11_contrib) - P11.vector.mu4
                
                        # save the scalar/vector mu^4 terms
                        P11.scalar.mu4 = P_scalar
                        P11.total.mu4 = P11.scalar.mu4 + P11.vector.mu4
            
            return P11
            
    @cached_property("k", "f",  "z", "sigma8_z", "power_lin", "max_mu", 
                     "include_2loop", "sigma_v", "sigma_bv2", "P00")
    def P02(self):
        """
        The correlation of density and energy density, which contributes
        mu^2 and mu^4 terms to the power expansion. There are no 
        linear contributions here.
        """
        P02 = PowerTerm()
        
        # do mu^2 terms?
        if self.max_mu >= 2:        
            Plin = self.normed_power_lin(self.k)
            
            # the necessary integrals 
            I02 = self.I02(self.k)
            J02 = self.J02(self.k)

            # the mu^2 no velocity terms
            P02.no_velocity.mu2 = self.f**2 * (I02 + 2.*self.k**2*J02*Plin)
            
            # the mu^2 terms depending on velocity (velocities in Mpc/h)
            sigma_lin = self.sigma_v
            sigma_02  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH)
            sigsq_eff = sigma_lin**2 + sigma_02**2

            if self.include_2loop:
                P02.with_velocity.mu2 = -(self.f*self.k)**2 * sigsq_eff*self.P00.total.mu0
            else:
                P02.with_velocity.mu2 = -(self.f*self.k)**2 * sigsq_eff*Plin
        
            # save the total mu^2 term
            P02.total.mu2 = P02.with_velocity.mu2 + P02.no_velocity.mu2
            
            # do mu^4 terms?
            if self.max_mu >= 4: 
                
                # the necessary integrals 
                I20 = self.I20(self.k)
                J20 = self.J20(self.k)
                P02.total.mu4 = P02.no_velocity.mu4 = self.f**2 * (I20 + 2*self.k**2*J20*Plin)
                
        return P02
        
    @cached_property("k", "f",  "z", "sigma8_z", "power_lin", "max_mu", 
                     "include_2loop", "sigma_v", "sigma_bv2", "P01")
    def P12(self):
        """
        The correlation of momentum density and energy density, which contributes
        mu^4 and mu^6 terms to the power expansion. There are no linear 
        contributions here. Two-loop contribution uses the mu^2 contribution
        from the P01 term.
        """
        P12 = PowerTerm()
        
        # do mu^4 terms?
        if self.max_mu >= 4:
            Plin = self.normed_power_lin(self.k)
            
            # the necessary integrals 
            I12 = self.I12(self.k)
            I03 = self.I03(self.k)
            J02 = self.J02(self.k)
            
            # do the mu^4 terms that don't depend on velocity
            P12.no_velocity.mu4 = self.f**3 * (I12 - I03 + 2*self.k**2*J02*Plin)
        
            # now do mu^4 terms depending on velocity (velocities in Mpc/h)
            sigma_lin = self.sigma_v  
            sigma_12  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH) 
            sigsq_eff = sigma_lin**2 + sigma_12**2
        
            if self.include_2loop:
                P12.with_velocity.mu4 = -0.5*(self.f*self.k)**2 * sigsq_eff*self.P01.total.mu2
            else:
                P12.with_velocity.mu4 = -self.f*(self.f*self.k)**2 * sigsq_eff*Plin
        
            # total mu^4 is velocity + no velocity terms
            P12.total.mu4 = P12.with_velocity.mu4 + P12.no_velocity.mu4
            
            # do mu^6 terms?
            if self.max_mu >= 6:
                
                # the necessary integrals 
                I21 = self.I21(self.k)
                I30 = self.I30(self.k)
                J20 = self.J20(self.k)
                
                P12.no_velocity.mu6 = self.f**3 * (I21 - I30 + 2*self.k**2*J20*Plin)
                P12.total.mu6 = P12.no_velocity.mu6
        
        return P12
        
    @cached_property("k", "f",  "z", "sigma8_z", "power_lin", "max_mu", 
                     "include_2loop", "sigma_v", "sigma_bv2", "P00", "P02")
    def P22(self):
        """
        The autocorelation of energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear 
        contributions here. 
        """
        P22 = PowerTerm()
        
        # velocity terms come in at 2-loop here
        if self.include_2loop:
            
            # velocities in units of Mpc/h
            sigma_lin = self.sigma_v
            sigma_22  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH) 
            sigsq_eff = sigma_lin**2 + sigma_22**2
            
            J02 = self.J02(self.k)
            J20 = self.J20(self.k)
            
        # do mu^4 terms?
        if self.max_mu >= 4:
            
            Plin = self.normed_power_lin(self.k)
            
            # 1-loop or 2-loop terms from <v^2 | v^2 > 
            if not self.include_2loop:
                P22.no_velocity.mu4 = 1./16*self.f**4 * self.I23(self.k)
            else:
                I23_2loop = self.Ivvvv_f23(self.k)
                P22.no_velocity.mu4 = 1./16*self.f**4 * I23_2loop

            # now add in the extra 2 loop terms, if specified
            if self.include_2loop:
                           
                # one more 2-loop term for <v^2 | v^2>
                extra_vv_mu4 = (self.f*self.k)**4 * Plin*J02**2
                
                # term from <v^2 | d v^2>
                extra_vdv_mu4 = -0.5*(self.f*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                
                # 1st term coming from <dv^2 | dv^2>
                extra1_dvdv_mu4 = 0.25*(self.f*self.k)**4 * sigsq_eff**2 * self.P00.total.mu0
                                    
                # 2nd term from <dv^2 | dv^2> is convolution of P22_bar and P00
                extra2_dvdv_mu4 = 0.5*(self.f*self.k)**4 * self.P00.total.mu0*self.sigmasq_k(self.k)**2
                
                # store the extra two loop terms
                extra = extra_vv_mu4 + extra_vdv_mu4 + extra1_dvdv_mu4 + extra2_dvdv_mu4
                P22.total.mu4 = P22.no_velocity.mu4 + extra
                
            else:
                P22.total.mu4 = P22.no_velocity.mu4
                
            # do mu^6 terms?
            if self.max_mu >= 6:
                
                # 1-loop or 2-loop terms that don't depend on velocity
                if not self.include_2loop:
                    P22.no_velocity.mu6 = 1./8*self.f**4 * self.I32(self.k)
                else:
                    I32_2loop = self.Ivvvv_f32(self.k)
                    P22.no_velocity.mu6 = 1./8*self.f**4 * I32_2loop
                    
                # now add in the extra 2 loop terms, if specified
                if self.include_2loop:

                    # term from <v^2 | d v^2>
                    extra_vdv_mu6 = -0.5*(self.f*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                    
                    # one more 2-loop term for <v^2 | v^2>
                    extra_vv_mu6  = 2*(self.f*self.k)**4 * Plin*J02*J20
                    
                    # save the totals
                    extra = extra_vv_mu6 + extra_vdv_mu6 
                    P22.total.mu6 = P22.no_velocity.mu6 + extra
                    
                else:
                    P22.total.mu6 =P22.no_velocity.mu6

                # do mu^8 terms?
                if self.max_mu >= 8:
                    
                    # 1-loop or 2-loop terms that don't depend on velocity
                    if not self.include_2loop:
                        P22.no_velocity.mu8 = 1./16*self.f**4 * self.I33(self.k)
                    else:
                        I33_2loop = self.Ivvvv_f33(self.k)
                        P22.no_velocity.mu8 = 1./16*self.f**4 * I33_2loop
                        
                        # extra 2 loop term from modeling <v^2|v^2>
                        P22.no_velocity.mu8 += (self.f*self.k)**4 * Plin*J20**2
                        
                    P22.total.mu8 = P22.no_velocity.mu8
                    
        return P22
        
    @cached_property("k", "f",  "z", "sigma8_z", "power_lin", "max_mu", 
                     "include_2loop", "sigma_v", "sigma_v2", "P01")
    def P03(self):
        """
        The cross-corelation of density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^4 terms.
        """
        P03 = PowerTerm()
        
        # do mu^4 terms?
        if self.max_mu >= 4:
            Plin = self.normed_power_lin(self.k)
            
            # only terms depending on velocity here (velocities in Mpc/h)
            sigma_lin = self.sigma_v 
            sigma_03  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH)
            sigsq_eff = sigma_lin**2 + sigma_03**2

            # either 1 or 2 loop quantities
            if self.include_2loop:
                P03.with_velocity.mu4 = -0.5*(self.f*self.k)**2 * sigsq_eff * self.P01.total.mu2
            else:
                P03.with_velocity.mu4 = -self.f*(self.f*self.k)**2 *sigsq_eff*Plin
        
            P03.total.mu4 = P03.with_velocity.mu4

        return P03
        
    @cached_property("k", "f",  "z", "sigma8_z", "power_lin", "max_mu", 
                     "include_2loop", "sigma_v", "sigma_bv2", "sigma_v2", "P11")
    def P13(self):
        """
        The cross-correlation of momentum density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^6 terms at 1-loop order and 
        mu^4 terms at 2-loop order.
        """
        P13 = PowerTerm()
        
        # compute velocity weighting in Mpc/h
        sigma_lin = self.sigma_v 
        sigma_13_v  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH) 
        sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
        
        if self.include_2loop:
            sigma_13_s  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH) 
            sigsq_eff_scalar = sigma_lin**2 + sigma_13_s**2
            
        # do mu^4 terms?
        if self.max_mu >= 4:
        
            # mu^4 is only 2-loop
            if self.include_2loop:

                A = -(self.f*self.k)**2
                P13_vel_mu4 = A*sigsq_eff_vector*self.P11.total.mu2
                P13.total.mu4 = P13.with_velocity.mu4 = P13_vel_mu4

            # do mu^6 terms?
            if self.max_mu >= 6:
                
                Plin = self.normed_power_lin(self.k)
                
                # mu^6 velocity terms at 1 or 2 loop
                if self.include_2loop:
                    A = -(self.f*self.k)**2
                    P13.with_velocity.mu6 = A*sigsq_eff_scalar*self.P11.total.mu4
                else:
                    P13.with_velocity.mu6 = -self.f**2 *(self.f*self.k)**2 * sigsq_eff_scalar*Plin
                    
                P13.total.mu6 = P13.with_velocity.mu6
        
        return P13
        
    @cached_property("k", "f",  "z", "sigma8_z", "power_lin", "max_mu", 
                     "include_2loop", "sigma_v", "sigma_bv4", "P02", "P00")
    def P04(self):
        """
        The cross-correlation of density with the rank four tensor field
        ((1+delta)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        P04 = PowerTerm()
        
        # only 2-loop terms here...
        if self.include_2loop:
            
            # compute the relevant small-scale + linear velocities in Mpc/h
            sigma_lin = self.sigma_v 
            sigma_04  = self.sigma_bv4 * self.cosmo.h() / (self.f*self.conformalH) 
            sigsq_eff = sigma_lin**2 + sigma_04**2
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # do P04 mu^4 terms depending on velocity
                P04_vel_mu4_1 = -0.5*(self.f*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                P04_vel_mu4_2 = 0.25*(self.f*self.k)**4 * sigsq_eff**2 * self.P00.total.mu0
                P04.with_velocity.mu4 = P04_vel_mu4_1 + P04_vel_mu4_2
                
                # do P04 mu^4 terms without vel dependence
                P04.no_velocity.mu4 = 1./12.*(self.f*self.k)**4 * self.P00.total.mu0*self.velocity_kurtosis
            
                # save the total
                P04.total.mu4 = P04.with_velocity.mu4 + P04.no_velocity.mu4
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # only terms depending on velocity
                    P04.with_velocity.mu6 = -0.5*(self.f*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                    P04.total.mu6 = P04.with_velocity.mu6
                    
        return P04
        
    #---------------------------------------------------------------------------
    # main user callables
    #---------------------------------------------------------------------------
    @tools.broadcast_kmu
    def power(self, k_obs, mu_obs, flatten=False):
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
        Pkmu : float, array_like
            The power model P(k, mu). If `mu` is a scalar, return dimensions
            are `(len(self.k), )`. If `mu` has dimensions (N, ), the return
            dimensions are `(len(k), N)`, i.e., each column corresponds is the
            model evaluated at different `mu` values. If `flatten = True`, then
            the returned array is raveled, with dimensions of `(N*len(self.k), )`
        """
        verify_krange(k_obs, self.kmin, self.kmax)
        
        # determine the true k/mu values and broadcast them to final shape
        mu = self.mu_true(mu_obs)
        k = self.k_true(k_obs, mu_obs)
        k, mu = np.broadcast_arrays(k, mu)
        
        # volume scaling
        vol_scaling = 1./(self.alpha_perp**2 * self.alpha_par)
        
        toret = np.zeros(k.shape)
        idx = k >= self.k0_low
        
        def _power(_k, _mu):
            if self.max_mu == 0:
                P_out = self.P_mu0(_k)
            elif self.max_mu == 2:
                P_out = self.P_mu0(_k) + _mu**2*self.P_mu2(_k)
            elif self.max_mu == 4:
                P_out = self.P_mu0(_k) + _mu**2*self.P_mu2(_k) + _mu**4*self.P_mu4(_k)
            elif self.max_mu == 6:
                P_out = self.P_mu0(_k) + _mu**2*self.P_mu2(_k) + _mu**4*self.P_mu4(_k) + _mu**6*self.P_mu6(_k)
            elif self.max_mu == 8:
                raise NotImplementedError("Cannot compute power spectrum including terms with order higher than mu^6")
            return np.nan_to_num(vol_scaling*P_out)
        
        # k >= k0_low
        if idx.sum():
            toret[idx] = _power(k[idx], mu[idx])
            
        # k < k0_low
        if (~idx).sum():
            A = _power(self.k0_low, mu[~idx])
            with tools.LowKPowerMode(self):
                norm = A / _power(self.k0_low, mu[~idx])
                toret[~idx] = _power(k[~idx], mu[~idx]) * norm

        if flatten: toret = np.ravel(toret, order='F')
        return toret
    
    @tools.monopole
    def monopole(self, k, mu, **kwargs):
        """
        The monopole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(k, mu)
            
    @tools.quadrupole
    def quadrupole(self, k, mu, **kwargs):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(k, mu)
    
    @tools.hexadecapole
    def hexadecapole(self, k, mu, **kwargs):
        """
        The hexadecapole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(k, mu)
                
    
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
