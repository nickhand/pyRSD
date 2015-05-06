from ._cache import Cache, parameter, interpolated_property, cached_property
from . import tools, INTERP_KMIN, INTERP_KMAX
from .. import pygcl, numpy as np

from ._integrals import Integrals
from ._sim_loader import SimLoader
from .simulation import SimulationPdv, SimulationP11
from .halo_zeldovich import HaloZeldovichP00, HaloZeldovichP01

#-------------------------------------------------------------------------------
class DarkMatterSpectrum(Cache, Integrals, SimLoader):
    """
    The dark matter power spectrum in redshift space
    """
    # splines and interpolation variables
    k_interp = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 100)
    spline = tools.RSDSpline
    spline_kwargs = {'bounds_error' : True, 'fill_value' : 0}
    
    # kwargs
    allowable_models = ['P00', 'P01', 'P11', 'Pdv']
    allowable_kwargs = ['k', 'z', 'cosmo', 'include_2loop', 'transfer_fit', \
                        'max_mu', 'interpolate']
    allowable_kwargs += ['use_%s_model' %m for m in allowable_models]
    
    #---------------------------------------------------------------------------
    def __init__(self, k=np.logspace(-2, np.log10(0.5), 100),
                       z=0., 
                       cosmo="planck1_WP.ini",
                       include_2loop=False,
                       transfer_fit="CLASS",
                       max_mu=4,
                       interpolate=True,
                       **kwargs):
        """
        Parameters
        ----------
        k : array_like, optional
            The wavenumbers to compute power spectrum at [units: `h/Mpc`]
            
        z : float, optional
            The redshift to compute the power spectrum at. Default = 0.
            
        cosmo : str
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
        """
        # initialize the Cache subclass first
        Cache.__init__(self)
        
        # set the input parameters
        self.hires          = False # by default
        self.interpolate    = interpolate
        self.transfer_fit   = transfer_fit
        self.cosmo_filename = cosmo
        self.max_mu         = max_mu
        self.include_2loop  = include_2loop
        self.z              = z 
        self.k_input        = k
        
        # initialize the cosmology parameters and set defaults
        self.sigma8            = self.cosmo.sigma8()
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
        for model in self.allowable_models:
            name = 'use_%s_model' %model
            val = kwargs.get(name, True)
            setattr(self, name, val)
        
        # initialize the other abstract base classes
        Integrals.__init__(self)
        SimLoader.__init__(self)
        
    #---------------------------------------------------------------------------
    # ATTRIBUTES
    #---------------------------------------------------------------------------
    @parameter
    def interpolate(self, val):
        """
        Whether we want to interpolate any underlying models
        """
        # set the dependencies
        models = ['P00_model', 'P01_model']
        self._update_models('interpolate', models, val)
        
        return val
    
    @parameter
    def transfer_fit(self, val):
        """
        The transfer function fitting method
        """
        allowed = ['CLASS', 'EH', 'EH_NoWiggle', 'BBKS']
        if val in allowed:
            return getattr(pygcl.Cosmology, val)
        else:
            raise ValueError("`transfer_fit` must be one of %s" %allowed)

    @parameter
    def cosmo_filename(self, val):
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
        models = ['P00_model', 'P01_model', 'Pdv_model', 'P11_model']
        self._update_models('z', models, val)
        
        return val

    @parameter
    def k_input(self, val):
        """
        The input wavenumbers specified by the user
        """
        return val
    
    @parameter
    def hires(self, val):
        """
        If `True`, return "high-resolution" results with 20x as many wavenumber
        data points.
        """
        return val
        
    @parameter
    def sigma8(self, val):
        """
        The value of Sigma8 (mass variances within 8 Mpc/h at z = 0) to compute 
        the power spectrum at, which gives the normalization of the 
        linear power spectrum
        """
        # update the dependencies
        models = ['P00_model', 'P01_model', 'Pdv_model', 'P11_model']
        self._update_models('sigma8', models, val)
        
        return val
        
    @parameter
    def f(self, val):
        """
        The growth rate, defined as the `dlnD/dlna`.
        """
        # update the dependencies
        models = ['P01_model', 'Pdv_model', 'P11_model']
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
    def small_scale_sigma(self, val):
        """
        Additional small scale sigma in km/s
        """
        self.sigma_bv2 = self.sigma_v2 = self.sigma_bv4 = val
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
    # CACHED PROPERTIES
    #---------------------------------------------------------------------------
    @cached_property('z', 'cosmo')
    def _normalized_sigma8_z(self):
        """
        Return the normalized sigma8(z) from the input cosmology
        """
        return self.cosmo.Sigma8_z(self.z) / self.cosmo.sigma8()
        
    @cached_property("sigma8", "_normalized_sigma8_z")
    def sigma8_z(self):
        """
        Return sigma8(z), normalized to the desired sigma8 at z = 0
        """
        return self.sigma8 * self._normalized_sigma8_z
            
    @cached_property("cosmo_filename", "transfer_fit")
    def cosmo(self):
        """
        A `pygcl.Cosmology` object holding the cosmological parameters
        """
        return pygcl.Cosmology(self.cosmo_filename, self.transfer_fit)
        
    @cached_property("k_input", "hires")
    def k_obs(self):
        """
        The "observed" wavenumbers to compute the power spectrum at.
        """
        if not self.hires:
            return self.k_input
        else:
            lo = np.amin(self.k_input)
            hi = np.amax(self.k_input)
            return np.linspace(lo, hi, 20*len(self.k_input))
              
    @cached_property("alpha_perp", "alpha_par", "k_obs")
    def k(self):
        """
        Return a hires range in wavenumbers, set by the minimum/maximum
        allowed values, given the desired wavenumbers `k_obs` and the
        current values of the AP effect parameters, `alpha_perp` and `alpha_par`
        """
        k_mu0 = self.k_true(self.k_obs, 0.)
        k_mu1 = self.k_true(self.k_obs, 1.)
        kmin = 0.95*min(np.amin(k_mu0), np.amin(k_mu1))
        kmax = 1.05*max(np.amax(k_mu0), np.amax(k_mu1))

        if kmin < INTERP_KMIN:
            msg = "Minimum possible k value with this `k_obs` and "
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

        return np.logspace(np.log10(kmin), np.log10(kmax), 500)
    
    @cached_property("z", "cosmo")
    def D(self):
        """
        The growth function, normalized to unity at z = 0
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
            
    @cached_property("cosmo_filename")
    def power_lin_nw(self):
        """
        A 'pygcl.LinearPS' object holding the linear power spectrum at z = 0, 
        using the Eisenstein-Hu no-wiggle transfer function
        """
        cosmo = pygcl.Cosmology(self.cosmo_filename, pygcl.Cosmology.EH_NoWiggle)
        return pygcl.LinearPS(cosmo, 0.)
                
    @cached_property("sigma8", "cosmo")
    def _power_norm(self):
        """
        The factor needed to normalize the linear power spectrum 
        in `power_lin` to the desired sigma_8, as specified by `sigma8`
        """
        return (self.sigma8 / self.cosmo.sigma8())**2  
        
    @cached_property("_power_norm", "_sigma_lin_unnormed")
    def sigma_lin(self):
        """
        The dark matter velocity dispersion at z = 0, as evaluated in 
        linear theory [units: Mpc/h]. Normalized to `self.sigma8`
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
    # MODELS TO USE
    #---------------------------------------------------------------------------
    @parameter
    def use_P00_model(self, val):
        """
        Whether to use Halo Zeldovich model for P00
        """
        return val

    #---------------------------------------------------------------------------
    @parameter
    def use_P01_model(self, val):
        """
        Whether to use Halo Zeldovich model for P01
        """
        return val

    #---------------------------------------------------------------------------
    @parameter
    def use_Pdv_model(self, val):
        """
        Whether to use interpolated sim results for Pdv
        """
        return val

    #---------------------------------------------------------------------------
    @parameter
    def use_P11_model(self, val):
        """
        Whether to use interpolated sim results for P11
        """
        return val

    #---------------------------------------------------------------------------
    @cached_property("cosmo")
    def P00_model(self):
        """
        The class holding the Halo Zeldovich model for the P00 dark matter term
        """
        return HaloZeldovichP00(self.cosmo, self.z, self.sigma8, self.interpolate)
    
    #---------------------------------------------------------------------------
    @cached_property("cosmo")
    def P01_model(self):
        """
        The class holding the Halo Zeldovich model for the P01 dark matter term
        """
        return HaloZeldovichP01(self.cosmo, self.z, self.sigma8, self.f, self.interpolate)
    
    #---------------------------------------------------------------------------
    @cached_property("power_lin_nw")
    def P11_model(self):
        """
        The class holding the model for the P11 dark matter term
        """
        return SimulationP11(self.power_lin_nw, self.z, self.sigma8, self.f)

    #---------------------------------------------------------------------------
    @cached_property("power_lin_nw")
    def Pdv_model(self):
        """
        The class holding the model for the Pdv dark matter term
        """
        return SimulationPdv(self.power_lin_nw, self.z, self.sigma8, self.f)
          
    #---------------------------------------------------------------------------
    # UTILITY FUNCTIONS
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
                
    #---------------------------------------------------------------------------
    def mu_true(self, mu_obs):
        """
        Return the `true` mu values, given an observed mu
        """
        F = self.alpha_par / self.alpha_perp
        return (mu_obs/F) * (1 + mu_obs**2*(1./F**2 - 1))**(-0.5)
    
    #-------------------------------------------------------------------------------
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
            except:
                pass
    
    #---------------------------------------------------------------------------
    def _update_models(self, name, models, val):
        """
        Update the specified attribute for the models given
        """
        for model in models:
            if hasattr(self, "_%s__%s" %(self.__class__.__name__, model)):
                setattr(getattr(self, model), name, val)
                
    #---------------------------------------------------------------------------    
    def normed_power_lin(self, k):
        """
        The linear power evaluated at the specified `k` and at `self.z`, 
        normalized to `self.sigma8`
        """
        return self._power_norm * self.D**2 * self.power_lin(k)

    #---------------------------------------------------------------------------
    def normed_power_lin_nw(self, k):
        """
        The Eisenstein-Hu no-wiggle, linear power evaluated at the specified 
        `k` and at `self.z`, normalized to `self.sigma8`
        """
        return self._power_norm * self.D**2 * self.power_lin_nw(k)
                
            
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES
    #---------------------------------------------------------------------------
    @interpolated_property("P00", interp="k")
    def P_mu0(self, k):
        """
        The full power spectrum term with no angular dependence. Contributions
        from P00.
        """
        return self.P00.total.mu0
        
    #---------------------------------------------------------------------------
    @interpolated_property("P01", "P11", "P02", interp="k")
    def P_mu2(self, k):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        return self.P01.total.mu2 + self.P11.total.mu2 + self.P02.total.mu2
        
    #---------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
    @interpolated_property("P12", "P22", "P13", "P04", "include_2loop", interp="k")
    def P_mu6(self, k):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P12, P22, P13, and P04 (2-loop).
        """
        Pk = self.P12.total.mu6 + self.P22.total.mu6 + self.P13.total.mu6
        if self.include_2loop: Pk += self.P04.total.mu6
        return Pk
            
    #---------------------------------------------------------------------------
    @cached_property("k", "z", "sigma8")
    def Pdd(self):
        """
        The 1-loop auto-correlation of density.
        """
        norm = self._power_norm*self.D**2
        return norm*(self.power_lin(self.k) + norm*self._Pdd_0(self.k))
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f", "z", "sigma8", "use_Pdv_model", "Pdv_loaded")
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
                return self.Pdv_model(self.k)
            else:
                norm = self._power_norm*self.D**2
                return (-self.f)*norm*(self.power_lin(self.k) + norm*self._Pdv_0(self.k))
          
    #---------------------------------------------------------------------------
    @cached_property("k", "f", "z", "sigma8")
    def Pvv(self):
        """
        The 1-loop auto-correlation of velocity divergence.
        """
        norm = self._power_norm*self.D**2
        return self.f**2 * norm*(self.power_lin(self.k) + norm*self._Pvv_0(self.k))
    
    #---------------------------------------------------------------------------
    @cached_property("k", "z", "sigma8", "use_P00_model", "power_lin", 
                     "P00_mu0_loaded")
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
                P00.total.mu0 = self.P00_model(self.k)
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
            
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "use_P01_model", "power_lin", 
                     "max_mu", "P01_mu2_loaded")
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
                    P01.total.mu2 = self.P01_model(self.k)
                # use pure PT
                else:                
                    # the necessary integrals 
                    I00 = self.I00(self.k)
                    J00 = self.J00(self.k)
        
                    Plin = self.normed_power_lin(self.k)
                    P01.total.mu2 = 2*self.f*(Plin + 4.*(I00 + 3*self.k**2*J00*Plin))
        
        return P01
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "use_P11_model", "power_lin", 
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
                    
                    # use the DM model
                    if self.use_P11_model:
                        P11.total.mu4 = self.P11_model(self.k)
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
            
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "power_lin", "max_mu", 
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
            sigma_02  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
            sigsq_eff = sigma_lin**2 + sigma_02**2

            if self.include_2loop:
                P02.with_velocity.mu2 = -(self.f*self.D*self.k)**2 * sigsq_eff*self.P00.total.mu0
            else:
                P02.with_velocity.mu2 = -(self.f*self.D*self.k)**2 * sigsq_eff*Plin
        
            # save the total mu^2 term
            P02.total.mu2 = P02.with_velocity.mu2 + P02.no_velocity.mu2
            
            # do mu^4 terms?
            if self.max_mu >= 4: 
                
                # the necessary integrals 
                I20 = self.I20(self.k)
                J20 = self.J20(self.k)
                P02.total.mu4 = P02.no_velocity.mu4 = self.f**2 * (I20 + 2*self.k**2*J20*Plin)
                
        return P02
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "power_lin", "max_mu", 
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
            sigma_12  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff = sigma_lin**2 + sigma_12**2
        
            if self.include_2loop:
                P12.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff*self.P01.total.mu2
            else:
                P12.with_velocity.mu4 = -self.f*(self.f*self.D*self.k)**2 * sigsq_eff*Plin
        
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
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "power_lin", "max_mu", 
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
            sigma_22  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
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
                extra_vdv_mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                
                # 1st term coming from <dv^2 | dv^2>
                extra1_dvdv_mu4 = 0.25*(self.f*self.D*self.k)**4 * sigsq_eff**2 * self.P00.total.mu0
                                    
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
                    extra_vdv_mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                    
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
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "power_lin", "max_mu", 
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
            sigma_03  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
            sigsq_eff = sigma_lin**2 + sigma_03**2

            # either 1 or 2 loop quantities
            if self.include_2loop:
                P03.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01.total.mu2
            else:
                P03.with_velocity.mu4 = -self.f*(self.f*self.D*self.k)**2 *sigsq_eff*Plin
        
            P03.total.mu4 = P03.with_velocity.mu4

        return P03
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "power_lin", "max_mu", 
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
        sigma_13_v  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
        sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
        
        if self.include_2loop:
            sigma_13_s  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff_scalar = sigma_lin**2 + sigma_13_s**2
            
        # do mu^4 terms?
        if self.max_mu >= 4:
        
            # mu^4 is only 2-loop
            if self.include_2loop:

                A = -(self.f*self.D*self.k)**2
                P13_vel_mu4 = A*sigsq_eff_vector*self.P11.total.mu2
                P13.total.mu4 = P13.with_velocity.mu4 = P13_vel_mu4

            # do mu^6 terms?
            if self.max_mu >= 6:
                
                Plin = self.normed_power_lin(self.k)
                
                # mu^6 velocity terms at 1 or 2 loop
                if self.include_2loop:
                    A = -(self.f*self.D*self.k)**2
                    P13.with_velocity.mu6 = A*sigsq_eff_scalar*self.P11.total.mu4
                else:
                    P13.with_velocity.mu6 = -self.f**2 *(self.f*self.D*self.k)**2 * sigsq_eff_scalar*Plin
                    
                P13.total.mu6 = P13.with_velocity.mu6
        
        return P13
        
    #---------------------------------------------------------------------------
    @cached_property("k", "f",  "z", "sigma8", "power_lin", "max_mu", 
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
            sigma_04  = self.sigma_bv4 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff = sigma_lin**2 + sigma_04**2
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # do P04 mu^4 terms depending on velocity
                P04_vel_mu4_1 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                P04_vel_mu4_2 = 0.25*(self.f*self.k)**4 * (self.D**2*sigsq_eff)**2 * self.P00.total.mu0
                P04.with_velocity.mu4 = P04_vel_mu4_1 + P04_vel_mu4_2
                
                # do P04 mu^4 terms without vel dependence
                P04.no_velocity.mu4 = 1./12.*(self.f*self.k)**4 * self.P00.total.mu0*self.velocity_kurtosis
            
                # save the total
                P04.total.mu4 = P04.with_velocity.mu4 + P04.no_velocity.mu4
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # only terms depending on velocity
                    P04.with_velocity.mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                    P04.total.mu6 = P04.with_velocity.mu6
                    
        return P04
    
    #---------------------------------------------------------------------------
    def _power_one_mu(self, mu_obs):
        """
        Internal function to evaluate P(k, mu) at a scalar mu value
        """
        # set the observed mu value
        mu = self.mu_true(mu_obs)
        k = self.k_true(self.k_obs, mu_obs)
        vol_scaling = 1./(self.alpha_perp**2 * self.alpha_par)
                
        if self.max_mu == 0:
            P_out = self.P_mu0(k)
        elif self.max_mu == 2:
            P_out = self.P_mu0(k) + mu**2*self.P_mu2(k)
        elif self.max_mu == 4:
            P_out = self.P_mu0(k) + mu**2*self.P_mu2(k) + mu**4*self.P_mu4(k)
        elif self.max_mu == 6:
            P_out = self.P_mu0(k) + mu**2*self.P_mu2(k) + mu**4*self.P_mu4(k) + mu**6*self.P_mu6(k)
        elif self.max_mu == 8:
            raise NotImplementedError("Cannot compute power spectrum including terms with order higher than mu^6")
            
        return np.nan_to_num(vol_scaling*P_out)
    
    #---------------------------------------------------------------------------
    def power(self, mu, flatten=False):
        """
        Return the redshift space power spectrum at the specified value of mu, 
        including terms up to ``mu**self.max_mu``.
        
        Parameters
        ----------
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
        if np.isscalar(mu):
            return self._power_one_mu(mu)
        else:
            toret = np.vstack([self._power_one_mu(imu) for imu in mu]).T
            if flatten: toret = np.ravel(toret, order='F')
            return toret
    
    #---------------------------------------------------------------------------
    @tools.monopole
    def monopole(self, mu):
        """
        The monopole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(mu)
            
    #---------------------------------------------------------------------------
    @tools.quadrupole
    def quadrupole(self, mu):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(mu)
    
    #---------------------------------------------------------------------------
    @tools.hexadecapole
    def hexadecapole(self, mu):
        """
        The hexadecapole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(mu)
                
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------   
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
        
#-------------------------------------------------------------------------------  
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

#-------------------------------------------------------------------------------


        
        
