from .. import numpy as np
from ._cache import parameter, cached_property, interpolated_property
from .power_dm import DarkMatterSpectrum, PowerTerm
from .simulation import SigmavFits, NonlinearBiasFits, \
                        StochasticityLogModel, StochasticityGPModel, PhmResidualGPModel


#-------------------------------------------------------------------------------
class BiasedSpectrum(DarkMatterSpectrum):
    
    allowable_models = DarkMatterSpectrum.allowable_models + ['Phm']
    allowable_kwargs = DarkMatterSpectrum.allowable_kwargs + \
                        ['sigmav_from_sims', 'use_tidal_bias', 'use_Phm_model', \
                         'stoch_model']    

    #---------------------------------------------------------------------------
    def __init__(self, sigmav_from_sims=True, use_tidal_bias=False, 
                    stoch_model='gaussian_process', **kwargs):
        
        # initalize the dark matter power spectrum
        super(BiasedSpectrum, self).__init__(**kwargs)
        
        # set the default parameters
        self.sigmav_from_sims = sigmav_from_sims
        self.use_tidal_bias   = use_tidal_bias
        self.include_2loop    = False # don't violate galilean invariance, fool
        self.stoch_model      = stoch_model
        self.b1               = 2.
        if (self.__class__.__name__ != "HaloSpectrum"):
            self.b1_bar           = 2.
            
        # turn off the Phm model by default
        val = kwargs.get('use_Phm_model', False)
        self.use_Phm_model = val
        
        # set A0/A1 of log model
        self.A0 = 0.
        self.A1 = 0.
        
        
    #---------------------------------------------------------------------------
    # ATTRIBUTES
    #---------------------------------------------------------------------------
    @parameter
    def use_Phm_model(self, val):
        """
        Whether to use a GP model for the Phm residual
        """
        return val
        
    @parameter
    def interpolate(self, val):
        """
        Whether we want to interpolate any underlying models
        """
        # set the dependencies
        models = ['P00_model', 'P01_model', 'stochasticity_gp_model', \
                  'Phm_residual_gp_model']
        self._update_models('interpolate', models, val)
        
        return val
        
    @parameter
    def z(self, val):
        """
        Redshift to evaluate power spectrum at
        """
        # update the dependencies
        models = ['P00_model', 'P01_model', 'Pdv_model', 'P11_model', \
                  'stochasticity_gp_model', 'Phm_residual_gp_model']
        self._update_models('z', models, val)

        return val
            
    @parameter
    def sigmav_from_sims(self, val):
        """
        Whether to use the linear velocity dispersion for halos as 
        computed from simulations.
        """
        return val
        
    @parameter
    def use_tidal_bias(self, val):
        """
        Whether to include the tidal bias terms
        """
        return val
                
    @parameter
    def stoch_model(self, val):
        """
        Attribute determining the stochasticity model to use.
        """
        allowable = ['gaussian_process', 'log', 'fit_log']
        if isinstance(val, basestring):
            if val not in allowable:
                raise ValueError("`stoch_model` must be one of %s or a scalar float" %allowable)
        else:
            if not np.isscalar(val):
                raise ValueError("`stoch_model` must be one of %s or a scalar float" %allowable)
        return val
        
    @parameter
    def b1(self, val):
        """
        The linear bias factor of the first tracer.
        """
        return val
    
    @parameter
    def b1_bar(self, val):
        """
        The linear bias factor of the 2nd tracer.
        """
        return val
                
    @parameter
    def A0(self, val):
        """
        The constant amplitude of the stochasticity log model
        """
        return val

    @parameter
    def A1(self, val):
        """
        The slope of the stochasticity log model
        """
        return val
    
    #---------------------------------------------------------------------------
    # CACHED PROPERTIES
    #---------------------------------------------------------------------------
    @cached_property("b1", "z")
    def b2_00(self):
        """
        The quadratic, local bias used for the P00_ss term for the 1st tracer.
        """
        return self.nonlinear_bias_fitter(self.b1, self.z, col='b2_00')
        
    @cached_property("b1_bar", "z")
    def b2_00_bar(self):
        """
        The quadratic, local bias used for the P00_ss term for the 2nd tracer.
        """
        return self.nonlinear_bias_fitter(self.b1_bar, self.z, col='b2_00')
    
    @cached_property("b1", "z")
    def b2_01(self):
        """
        The quadratic, local bias used for the P01_ss term for the 1st tracer.
        """
        return self.nonlinear_bias_fitter(self.b1, self.z, col='b2_01')        
    
    @cached_property("b1_bar", "z")
    def b2_01_bar(self):
        """
        The quadratic, local bias used for the P01_ss term for the 2nd tracer.
        """
        return self.nonlinear_bias_fitter(self.b1_bar, self.z, col='b2_01')
    
    @cached_property("b1", "use_tidal_bias")
    def bs(self):
        """
        The quadratic, nonlocal tidal bias factor for the first tracer
        """
        if self.use_tidal_bias:
            return -2./7 * (self.b1 - 1.)
        else:
            return 0.
        
    @cached_property("b1_bar", "use_tidal_bias")
    def bs_bar(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        if self.use_tidal_bias:
            return -2./7 * (self.b1_bar - 1.)
        else:
            return 0.
    
    @cached_property()
    def nonlinear_bias_fitter(self):
        """
        Interpolator from simulation data for nonlinear biases
        """
        return NonlinearBiasFits()
          
    @cached_property()
    def sigmav_fitter(self):
        """
        Interpolator from simulation data for linear sigmav of halos
        """
        return SigmavFits()
        
    @cached_property()
    def stochasticity_log_model(self):
        """
        The log model for stochasticity, as measured from simulations
        """
        return StochasticityLogModel()

    @cached_property()
    def stochasticity_gp_model(self):
        """
        The GP model for stochasticity, as measured from simulations
        """
        return StochasticityGPModel(self.z, self.interpolate)  
        
    @cached_property()
    def Phm_residual_gp_model(self):
        """
        The GP model for the Phm residual, as measured from simulations
        """
        return PhmResidualGPModel(self.z, self.interpolate)      

    @cached_property("b1", "b1_bar", "sigma8", "sigmav_from_sims", "sigma_v")
    def biased_sigma_v(self):
        """
        The velocity dispersion at z = 0. 
        
        If not provided and ``sigmav_from_sims = False``, this defaults to 
        the linear theory prediction [units: Mpc/h]
        """
        if self.sigmav_from_sims:
            
            # get the sigma in Mpc/h with D(z)*f(z) behavior divided out
            mean_bias = np.sqrt(self.b1*self.b1_bar)
            sigma_v = self.sigmav_fitter(mean_bias, self.z)
            
            # normalize to correct sigma8
            norm_factor = self.sigma8 / 0.807
            return sigma_v * norm_factor
        else:
            return self.sigma_v
            
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES
    #---------------------------------------------------------------------------
    @cached_property("b1", "P00", "use_Phm_model")
    def Phm(self):
        """
        The halo - matter cross correlation for the 1st tracer
        """
        Phm = PowerTerm()
        
        if not self.use_Phm_model:
            term1 = self.b1*self.P00.total.mu0
            term2 = self.b2_00*self.K00(self.k)
            term3 = self.bs*self.K00s(self.k)
            Phm.total.mu0 = term1 + term2 + term3
        else:
            Phm_residual = self.Phm_residual_gp_model(self.b1, self.k)
            Pzel = self.P00_model.zeldovich_power(self.k)
            
            Phm.total.mu0 = self.b1*Pzel + Phm_residual
            
        return Phm
        
    @cached_property("b1_bar", "P00", "use_Phm_model")
    def Phm_bar(self):
        """
        The halo - matter cross correlation for the 2nd tracer
        """
        Phm = PowerTerm()
        
        if not self.use_Phm_model:
            term1 = self.b1_bar*self.P00.total.mu0
            term2 = self.b2_00_bar*self.K00(self.k)
            term3 = self.bs_bar*self.K00s(self.k)
            Phm.total.mu0 = term1 + term2 + term3
        else:
            Phm_residual = self.Phm_residual_gp_model(self.b1_bar, self.k)
            Pzel = self.P00_model.zeldovich_power(self.k)

            Phm.total.mu0 = self.b1_bar*Pzel + Phm_residual

        return Phm
        
    @cached_property("stoch_model", "k", "b1", "b1_bar", "z", "A0", "A1")
    def stochasticity(self):
        """
        The isotropic stochasticity term due to the discreteness of the halos, 
        i.e., Poisson noise.
        """
        if not isinstance(self.stoch_model, basestring):
            return self.k*0. + self.stoch_model
        else:
            mean_bias = (self.b1*self.b1_bar)**0.5
            if self.stoch_model == 'gaussian_process':
                return self.stochasticity_gp_model(mean_bias, self.k)
            elif self.stoch_model == 'log':
                return self.stochasticity_log_model(self.k, mean_bias, self.z)
            elif self.stoch_model == 'fit_log':
                return self.A0 + self.A1*np.log(self.k)
                
                
    @cached_property("P00_ss_no_stoch", "stochasticity")
    def P00_ss(self):
        """
        The isotropic, halo-halo power spectrum, including the (possibly
        k-dependent) stochasticity term, as specified by the user.
        """
        stoch = self.stochasticity
        P00_ss = PowerTerm()
        P00_ss.total.mu0 = self.P00_ss_no_stoch.total.mu0 + stoch
        return P00_ss
            
    #---------------------------------------------------------------------------
    @cached_property("P00", "Phm", "Phm_bar")
    def P00_ss_no_stoch(self):
        """
        The isotropic, halo-halo power spectrum, without any stochasticity term.
        """    
        P00_ss_no_stoch = PowerTerm()
        term1 = self.b1_bar*self.Phm.total.mu0 + self.b1*self.Phm_bar.total.mu0
        term2 = (self.b1*self.b1_bar)*self.P00.total.mu0
        P00_ss_no_stoch.total.mu0 = term1 - term2
        
        return P00_ss_no_stoch
            
    #---------------------------------------------------------------------------
    @cached_property("b1", "b1_bar", "max_mu", "Pdv", "P01")
    def P01_ss(self):
        """
        The correlation of the halo density and halo momentum fields, which 
        contributes mu^2 terms to the power expansion.
        """        
        P01_ss = PowerTerm()

        # do mu^2 terms?
        if self.max_mu >= 2:
            
            # get the integral attributes
            K10  = self.K10(self.k)
            K10s = self.K10s(self.k)
            K11  = self.K11(self.k)
            K11s = self.K11s(self.k)
            
            term1 = (self.b1*self.b1_bar) * self.P01.total.mu2
            term2 = -self.Pdv*(self.b1*(1. - self.b1_bar) + self.b1_bar*(1. - self.b1))
            term3 = self.f*((self.b2_01 + self.b2_01_bar)*K10 + (self.bs + self.bs_bar)*K10s )
            term4 = self.f*((self.b1_bar*self.b2_01 + self.b1*self.b2_01_bar)*K11 + \
                        (self.b1_bar*self.bs + self.b1*self.bs_bar)*K11s)
    
            P01_ss.total.mu2 = term1 + term2 + term3 + term4
        return P01_ss
        
    #---------------------------------------------------------------------------
    @cached_property("b1", "b1_bar", "max_mu", "P02", "P00_ss_no_stoch",
                     "sigma_v", "sigma_bv2")
    def P02_ss(self):
        """
        The correlation of the halo density and halo kinetic energy, which 
        contributes mu^2 and mu^4 terms to the power expansion.
        """
        b1, b1_bar       = self.b1, self.b1_bar
        b2_00, b2_00_bar = self.b2_00, self.b2_00_bar
        bs, bs_bar       = self.bs, self.bs_bar
        
        P02_ss = PowerTerm()
        
        # do mu^2 terms?
        if self.max_mu >= 2:
            
            # the mu^2 terms depending on velocity (velocities in Mpc/h)
            sigma_lin = self.biased_sigma_v
            sigma_02  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
            sigsq_eff = sigma_lin**2 + sigma_02**2
            
            # get the integral attributes
            K20_a = self.K20_a(self.k)
            K20s_a = self.K20s_a(self.k)
            
            term1_mu2 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu2            
            term2_mu2 =  -(self.f*self.D*self.k)**2 * sigsq_eff * self.P00_ss_no_stoch.total.mu0
            term3_mu2 = 0.5*self.f**2 * ( (b2_00 + b2_00_bar)*K20_a + (bs + bs_bar)*K20s_a )
            P02_ss.total.mu2 = term1_mu2 + term2_mu2 + term3_mu2
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # get the integral attributes
                K20_b = self.K20_b(self.k)
                K20s_b = self.K20s_b(self.k)
                
                term1_mu4 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu4
                term2_mu4 = 0.5*self.f**2 * ( (b2_00 + b2_00_bar)*K20_b + (bs + bs_bar)*K20s_b )
                P02_ss.total.mu4 = term1_mu4 + term2_mu4
        
        return P02_ss
            
    #---------------------------------------------------------------------------
    @cached_property("b1", "b1_bar", "max_mu", "P11")
    def P11_ss(self):
        """
        The auto-correlation of the halo momentum field, which 
        contributes mu^2 and mu^4 terms to the power expansion. The mu^2 terms
        come from the vector part, while the mu^4 dependence is dominated by
        the scalar part on large scales (linear term) and the vector part
        on small scales.
        """
        b1, b1_bar = self.b1, self.b1_bar
        P11_ss = PowerTerm()

        # do mu^2 terms?
        if self.max_mu >= 2:
            
            # this is C11 at 2-loop order
            I1 = self.Ivvdd_h01(self.k)
            I2 = self.Idvdv_h03(self.k)
            P11_ss.total.mu2 = (b1*b1_bar)*self.f**2 * (I1 + I2)
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                Plin = self.normed_power_lin(self.k)
                
                # get the integral attributes
                I22 = self.I22(self.k)
                J10 = self.J10(self.k)
                
                # first term is mu^4 part of P11
                term1_mu4 = self.P11.total.mu4
                
                # second term is B11 coming from P11
                term2_mu4 = (b1 + b1_bar - 2.)*self.f**2 * (6.*self.k**2*Plin*J10 + 2*I22)
                
                # third term is mu^4 part of C11 (at 2-loop)
                I1 = self.Ivvdd_h02(self.k)
                I2 = self.Idvdv_h04(self.k)
                term3_mu4 =  (b1*b1_bar - 1)*self.f**2 * (I1 + I2)

                P11_ss.total.mu4 = term1_mu4 + term2_mu4 + term3_mu4
        
        return P11_ss
            
    #---------------------------------------------------------------------------
    @cached_property("P01_ss", "sigma_v", "sigma_v2")
    def P03_ss(self):
        """
        The cross-corelation of halo density with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 terms.
        """
        P03_ss = PowerTerm()
        
        # do mu^4 term?
        if self.max_mu >= 4:
            
            # optionally add small scale velocity
            sigma_lin = self.biased_sigma_v 
            sigma_03  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
            sigsq_eff = sigma_lin**2 + sigma_03**2
            
            P03_ss.total.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01_ss.total.mu2
    
        return P03_ss
            
    #---------------------------------------------------------------------------
    @cached_property("P01_ss", "sigma_v", "sigma_bv2")
    def P12_ss(self):
        """
        The correlation of halo momentum and halo kinetic energy density, which 
        contributes mu^4 and mu^6 terms to the power expansion.
        """
        b1, b1_bar = self.b1, self.b1_bar
        P12_ss = PowerTerm()
        
        # do mu^4 terms?
        if self.max_mu >= 4:
            Plin = self.normed_power_lin(self.k)
            
            # now do mu^4 terms depending on velocity (velocities in Mpc/h)
            sigma_lin = self.biased_sigma_v  
            sigma_12  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff = sigma_lin**2 + sigma_12**2
            
            # get the integral attributes
            I12 = self.I12(self.k)
            I03 = self.I03(self.k)
            J02 = self.J02(self.k)
            
            term1_mu4 = self.f**3 * (I12 - 0.5*(b1 + b1_bar)*I03 + 2*self.k**2 * J02*Plin)
            term2_mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01_ss.total.mu2
            P12_ss.total.mu4 = term1_mu4 + term2_mu4
            
            # do mu^6 terms?
            if self.max_mu >= 6:
                
                # get the integral attributes
                I21 = self.I21(self.k)
                I30 = self.I30(self.k)
                J20 = self.J20(self.k)
                
                P12_ss.total.mu6 = self.f**3 * (I21 - 0.5*(b1+b1_bar)*I30 + 2*self.k**2*J20*Plin)
        
        return P12_ss
            
    #---------------------------------------------------------------------------
    @cached_property("P11_ss", "sigma_v", "sigma_bv2", "sigma_v2")
    def P13_ss(self):
        """
        The cross-correlation of halo momentum with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 and mu^6 terms.
        """
        P13_ss = PowerTerm()
        
        # vector small scale velocity additions
        sigma_lin = self.biased_sigma_v 
        sigma_13_v  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
        sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
        
        # the amplitude
        A = -(self.f*self.D*self.k)**2
        
        # do mu^4 terms?
        if self.max_mu >= 4:
            
            # using P11_ss mu^2 terms 
            P13_ss.total.mu4 = A*sigsq_eff_vector*self.P11_ss.total.mu2
        
            # do mu^6 terms?
            if self.max_mu >= 6:
                
                # scalar small scale velocity additions
                sigma_13_s  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                sigsq_eff_scalar = sigma_lin**2 + sigma_13_s**2
                
                # using P11_ss mu^4 terms
                P13_ss.total.mu6 = A*sigsq_eff_scalar*self.P11_ss.total.mu4
        
        return P13_ss
            
    #---------------------------------------------------------------------------
    @cached_property("P22", "Pdd", "P02", "P00_ss_no_stoch", "sigma_v", "sigma_bv2")
    def P22_ss(self):
        """
        The auto-corelation of halo kinetic energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear 
        contributions here. 
        """
        b1, b1_bar = self.b1, self.b1_bar
        P22_ss = PowerTerm()
        
        # do mu^4 terms?
        if self.max_mu >= 4:
            
            # velocities in units of Mpc/h
            sigma_lin = self.biased_sigma_v
            sigma_22  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff = sigma_lin**2 + sigma_22**2
        
            # 1-loop P22bar
            term1 = self.P22.no_velocity.mu4
            
            # add convolution to P22bar
            term2 = 0.5*(self.f*self.k)**4 * (b1*b1_bar * self.Pdd) * self.sigmasq_k(self.k)**2
            
            # b1 * P02_bar
            term3 = -0.5*(self.k*self.f*self.D)**2 * sigsq_eff * ( 0.5*(b1 + b1_bar)*self.P02.no_velocity.mu2)
            
            # sigma^4 x P00_ss
            term4 = 0.25*(self.k*self.f*self.D)**4 * sigsq_eff**2 * self.P00_ss_no_stoch.total.mu0
            
            P22_ss.total.mu4 = term1 + term2 + term3 + term4
            
            # do mu^6 terms?
            if self.max_mu >= 6:
                
                term1 = self.P22.no_velocity.mu6
                term2 = -0.5*(self.k*self.f*self.D)**2 * sigsq_eff * (0.5*(b1 + b1_bar)*self.P02.no_velocity.mu4)
                P22_ss.total.mu6 = term1 + term2
            
                    
        return P22_ss
            
    #---------------------------------------------------------------------------
    @cached_property("P02", "P00_ss_no_stoch", "sigma_v", "sigma_bv4")
    def P04_ss(self):
        """
        The cross-correlation of halo density with the rank four tensor field
        ((1+delta_h)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        b1, b1_bar = self.b1, self.b1_bar
        P04_ss = PowerTerm()
        
        # do mu^4 terms?
        if self.max_mu >= 4:
            
            # compute the relevant small-scale + linear velocities in Mpc/h
            sigma_lin = self.biased_sigma_v 
            sigma_04  = self.sigma_bv4 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff = sigma_lin**2 + sigma_04**2
            
            # contribution from P02[mu^2]
            term1 = -0.25*(b1 + b1_bar)*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
            
            # contribution here from P00_ss * vel^4
            A = (1./12)*(self.f*self.D*self.k)**4 * self.P00_ss_no_stoch.total.mu0
            vel_kurtosis = self.velocity_kurtosis / self.D**4
            term2 = A*(3.*sigsq_eff**2 + vel_kurtosis)
            
            P04_ss.total.mu4 = term1 + term2
        
            # do mu^6 terms?
            if self.max_mu >= 6:
                P04_ss.total.mu6 = -0.25*(b1 + b1_bar)*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
            
        return P04_ss
            
    #---------------------------------------------------------------------------
    @interpolated_property("P00_ss", interp="k")
    def P_mu0(self, k):
        """
        The full halo power spectrum term with no angular dependence. Contributions
        from P00_ss.
        """
        return self.P00_ss.total.mu0

    
    #---------------------------------------------------------------------------
    @interpolated_property("P01_ss", "P11_ss", "P02_ss", interp="k")
    def P_mu2(self, k):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_ss, P11_ss, and P02_ss.
        """
        return self.P01_ss.total.mu2 + self.P11_ss.total.mu2 + self.P02_ss.total.mu2


    #---------------------------------------------------------------------------
    @interpolated_property("P11_ss", "P02_ss", "P12_ss", "P22_ss", "P03_ss",
                           "P13_ss", "P04_ss", interp="k")
    def P_mu4(self, k):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_ss, P02_ss, P12_ss, P03_ss, P13_ss, P22_ss, and P04_ss.
        """
        return self.P11_ss.total.mu4 + self.P02_ss.total.mu4 + self.P12_ss.total.mu4 + \
               self.P22_ss.total.mu4 + self.P03_ss.total.mu4 + self.P13_ss.total.mu4 + \
               self.P04_ss.total.mu4
    
            
    #---------------------------------------------------------------------------
    @interpolated_property("P12_ss", interp="k")
    def P_mu6(self, k):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_ss, P13_ss, P22_ss.
        """
        return self.P12_ss.total.mu6 + 1./8*self.f**4 * self.I32(self.k)

    #---------------------------------------------------------------------------    
#-------------------------------------------------------------------------------  

        
