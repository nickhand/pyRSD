import contextlib
from .. import numpy as np

# tools
from ._cache import parameter, cached_property, interpolated_property
from .tools import RSDSpline, BiasToSigmaRelation

# base model
from .power_dm import DarkMatterSpectrum
from .power_dm import PowerTerm

# halo zeldovich models
from .halo_zeldovich import HaloZeldovichPhm

# simulation fits
from .simulation import VelocityDispersionFits
from .simulation import Pmu2ResidualCorrection
from .simulation import Pmu4ResidualCorrection
from .simulation import AutoStochasticityFits
from .simulation import CrossStochasticityFits

# nonliner biasing
from .nonlinear_biasing import NonlinearBiasingMixin

class BiasedSpectrum(DarkMatterSpectrum, NonlinearBiasingMixin):
    """
    The power spectrum of two biased tracers, with linear biases `b1`
    and `b1_bar` in redshift space
    """    
    def __init__(self, use_tidal_bias=False,
                       use_mean_bias=False,
                       vel_disp_from_sims=False,  
                       correct_mu2=False,
                       correct_mu4=False,
                       use_vlah_biasing=False,
                       **kwargs):
        
        # initalize the dark matter power spectrum
        super(BiasedSpectrum, self).__init__(**kwargs)
        
        # the nonlinear biasing class
        NonlinearBiasingMixin.__init__(self)
        
        # set the default parameters
        self.use_mean_bias      = use_mean_bias
        self.vel_disp_from_sims = vel_disp_from_sims
        self.use_tidal_bias     = use_tidal_bias
        self.include_2loop      = False # don't violate galilean invariance, fool
        self.b1                 = 2.
        
        # correction models
        self.correct_mu2 = correct_mu2
        self.correct_mu4 = correct_mu4
        
        # whether to use Vlah et al nonlinear biasing
        self.use_vlah_biasing = use_vlah_biasing
        
        # set b1_bar, unless we are fixed
        if (self.__class__.__name__ != "HaloSpectrum"): self.b1_bar = 2.
         
    #---------------------------------------------------------------------------
    # attributes
    #---------------------------------------------------------------------------
    @parameter
    def use_vlah_biasing(self, val):
        """
        Whether to use the nonlinear biasing scheme from Vlah et al. 2013
        """           
        return val
        
    @parameter
    def correct_mu2(self, val):
        """
        Whether to correct the halo P[mu2] model, using a sim-calibrated model
        """           
        return val
        
    @parameter
    def correct_mu4(self, val):
        """
        Whether to correct the halo P[mu4] model, using a sim-calibrated model
        """           
        return val
                
    @parameter
    def use_mean_bias(self, val):
        """
        Evaluate cross-spectra using the geometric mean of the biases 
        """           
        return val
        
    @parameter
    def interpolate(self, val):
        """
        Whether we want to interpolate any underlying models
        """
        # set the dependencies
        models = ['P00_hzpt_model', 'P01_hzpt_model', 'P11_hzpt_model', 'Phm_hzpt_model']
        self._update_models('interpolate', models, val)
        
        return val
        
    @parameter
    def enhance_wiggles(self, val):
        """
        Whether to enhance the wiggles over the default HZPT model
        """
        # set the dependencies
        models = ['P00_hzpt_model', 'P01_hzpt_model', 'Phm_hzpt_model']
        self._update_models('enhance_wiggles', models, val)
        
        return val
        
    @parameter
    def sigma8_z(self, val):
        """
        The value of Sigma8 (mass variances within 8 Mpc/h at z) to compute 
        the power spectrum at, which gives the normalization of the 
        linear power spectrum
        """
        # update the dependencies
        models = ['P00_hzpt_model', 'P01_hzpt_model', 'P11_hzpt_model', 
                    'P11_sim_model', 'Pdv_sim_model', 'Phm_hzpt_model']
        self._update_models('sigma8_z', models, val)

        return val
    
    @parameter
    def z(self, val):
        """
        Redshift to evaluate power spectrum at
        """
        # update the dependencies
        models = ['P00_hzpt_model', 'P01_hzpt_model', 'P11_hzpt_model', 
                  'Phm_hzpt_model', 'bias_to_sigma_relation',
                  'P11_sim_model', 'Pdv_sim_model']
        self._update_models('z', models, val)

        return val
            
    @parameter
    def vel_disp_from_sims(self, val):
        """
        Whether to use the linear velocity dispersion for halos as 
        computed from simulations, which includes a slight mass
        dependence 
        """
        return val
        
    @parameter
    def use_tidal_bias(self, val):
        """
        Whether to include the tidal bias terms
        """
        return val
                
    @parameter
    def use_Phm_model(self, val):
        """
        If `True`, use the `HZPT` model for Phm
        """
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
                
    #---------------------------------------------------------------------------
    # simulation-calibrated model parameters
    #---------------------------------------------------------------------------                  
    @cached_property()
    def vel_disp_fitter(self):
        """
        Interpolator from simulation data for linear velocity dispersion of halos
        """
        return VelocityDispersionFits()
        
    @cached_property()
    def Pmu2_correction(self):
        """
        The parameters for the halo P[mu2] correction
        """
        return Pmu2ResidualCorrection()
        
    @cached_property()
    def Pmu4_correction(self):
        """
        The parameters for the halo P[mu4] correction
        """
        return Pmu4ResidualCorrection()

    @cached_property()
    def auto_stochasticity_fits(self):
        """
        The prediction for the auto stochasticity from a GP
        """
        return AutoStochasticityFits()
        
    @cached_property()
    def cross_stochasticity_fits(self):
        """
        The prediction for the cross stochasticity from a GP
        """
        return CrossStochasticityFits()
            
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("cosmo")
    def Phm_hzpt_model(self):
        """
        The class holding the Halo Zeldovich model for the Phm term
        """
        kw = {'interpolate':self.interpolate, 'enhance_wiggles':self.enhance_wiggles}
        return HaloZeldovichPhm(self.cosmo, self.z, self.sigma8_z, **kw)
        
    @cached_property("use_mean_bias", "b1", "b1_bar")
    def _ib1(self):
        """
        The internally used bias of the 1st tracer
        """
        if not self.use_mean_bias:
            return self.b1
        else:
            return (self.b1*self.b1_bar)**0.5
    
    @cached_property("use_mean_bias", "b1", "b1_bar")
    def _ib1_bar(self):
        """
        The internally used bias of the first tracer
        """
        if not self.use_mean_bias:
            return self.b1_bar
        else:
            return (self.b1*self.b1_bar)**0.5
                    
    @cached_property("_ib1", "use_tidal_bias")
    def bs(self):
        """
        The quadratic, nonlocal tidal bias factor for the first tracer
        """
        if self.use_tidal_bias:
            return -2./7 * (self._ib1 - 1.)
        else:
            return 0.
        
    @cached_property("_ib1_bar", "use_tidal_bias")
    def bs_bar(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        if self.use_tidal_bias:
            return -2./7 * (self._ib1_bar - 1.)
        else:
            return 0.
                
    @cached_property()
    def bias_to_sigma_relation(self):
        """
        The relationship between bias and velocity dispersion, using the 
        Tinker et al. relation for halo mass and bias, and the virial theorem
        scaling between mass and velocity dispersion
        """
        return BiasToSigmaRelation(self.z, self.cosmo, interpolate=self.interpolate)
             
    @cached_property("sigma_v", "sigma_lin", "vel_disp_from_sims", "_ib1", "sigma8_z")
    def sigmav_halo(self):
        """
        The velocity dispersion for halos, possibly as a function of bias
        """
        if self.vel_disp_from_sims:
            return self.vel_disp_fitter(b1=self._ib1, sigma8_z=self.sigma8_z)
        else:
            return self.sigma_v
            
    @cached_property("sigma_v", "sigma_lin", "vel_disp_from_sims", "_ib1_bar", "sigma8_z")
    def sigmav_halo_bar(self):
        """
        The velocity dispersion for halos, possibly as a function of bias
        """
        if self.vel_disp_from_sims:
            return self.vel_disp_fitter(b1=self._ib1_bar, sigma8_z=self.sigma8_z)
        else:
            return self.sigma_v
        
    def sigmav_from_bias(self, s8_z, bias):
        """
        Return the velocity dispersion `sigmav` value for the specified linear
        bias value
        """
        try:
            return self.bias_to_sigma_relation(s8_z, bias)
        except Exception as e:
            msg = "Warning: error in computing sigmav from (s8_z, b1) = (%.2f, %.2f); original msg = %s" %(s8_z, bias, e)
            b1s = self.bias_to_sigma_relation.interpolation_grid['b1']
            if bias < np.amin(b1s):
                toret = self.bias_to_sigma_relation(s8_z, np.amin(b1s))
            elif bias > np.amax(b1s):
                toret = self.bias_to_sigma_relation(s8_z, np.amax(b1s))
            else:
                toret = 0.
            print(msg + "; returning sigmav = %.4e" %toret)
            return toret
                        
    #---------------------------------------------------------------------------
    # power term attributes
    #---------------------------------------------------------------------------                
    @cached_property("_ib1", "P00", "use_Phm_model", "sigma8_z", "b2_00_a")
    def Phm(self):
        """
        The halo - matter cross correlation for the 1st tracer
        """
        Phm = PowerTerm()
        
        if self.use_Phm_model:
            Phm.total.mu0 = self.Phm_hzpt_model(self._ib1, self.k)
        else:
            # the bias values to use
            b1, b2_00 = self._ib1, self.b2_00_a(self._ib1)
            
            term1 = b1*self.P00.total.mu0
            term2 = b2_00*self.K00(self.k)
            term3 = self.bs*self.K00s(self.k)
            Phm.total.mu0 = term1 + term2 + term3
            
        return Phm
        
    @cached_property("_ib1_bar", "P00", "use_Phm_model", "sigma8_z", "b2_00_a")
    def Phm_bar(self):
        """
        The halo - matter cross correlation for the 2nd tracer
        """
        Phm = PowerTerm()
        
        if self.use_Phm_model:
            Phm.total.mu0 = self.Phm_hzpt_model(self._ib1_bar, self.k)
        else:
            # the bias values to use
            b1_bar, b2_00_bar = self._ib1_bar, self.b2_00_a(self._ib1_bar)
            
            term1 = b1_bar*self.P00.total.mu0
            term2 = b2_00_bar*self.K00(self.k)
            term3 = self.bs_bar*self.K00s(self.k)
            Phm.total.mu0 = term1 + term2 + term3

        return Phm
        
    @cached_property("k", "_ib1", "_ib1_bar", "z", "sigma8_z")
    def stochasticity(self):
        """
        The isotropic (type B) stochasticity term due to the discreteness of the 
        halos, i.e., Poisson noise at 1st order.
        
        Notes
        -----
        *   The model for the (type B) stochasticity, interpolated as a function 
            of sigma8(z), b1, and k using a Gaussian process
        """
        params = {'sigma8_z' : self.sigma8_z, 'k' : self.k}    
        if self._ib1 != self._ib1_bar:
            b1_1, b1_2 = sorted([self._ib1, self._ib1_bar])
            return self.cross_stochasticity_fits(b1_1=b1_1, b1_2=b1_2, **params)
        else:
            return self.auto_stochasticity_fits(b1=self._ib1, **params)
                              
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
            
    @cached_property("P00", "Phm", "Phm_bar")
    def P00_ss_no_stoch(self):
        """
        The isotropic, halo-halo power spectrum, without any stochasticity term.
        """    
        P00_ss_no_stoch = PowerTerm()
        b1_k = self.Phm.total.mu0  / self.P00.total.mu0
        b1_bar_k = self.Phm_bar.total.mu0 / self.P00.total.mu0
        P00_ss_no_stoch.total.mu0 = b1_k * b1_bar_k * self.P00.total.mu0
        	       
        return P00_ss_no_stoch
            
    @cached_property("_ib1", "_ib1_bar", "max_mu", "Pdv", "P01", "b2_01_a")
    def P01_ss(self):
        """
        The correlation of the halo density and halo momentum fields, which 
        contributes mu^2 terms to the power expansion.
        """ 
        P01_ss = PowerTerm()

        # do mu^2 terms?
        if self.max_mu >= 2:    
            P01_ss.total.mu2 = P01_mu2(self, self.b2_01_a)
            
        return P01_ss
        
    @cached_property("_ib1", "_ib1_bar", "max_mu", "P02", "P00",
                     "sigmav_halo", "sigmav_halo_bar", "b2_00_b", "b2_00_c")
    def P02_ss(self):
        """
        The correlation of the halo density and halo kinetic energy, which 
        contributes mu^2 and mu^4 terms to the power expansion.
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        bs, bs_bar = self.bs, self.bs_bar
        
        P02_ss = PowerTerm()
        
        # do mu^2 terms?
        if self.max_mu >= 2:
            
            # the nonlinear bias to use
            b2_00, b2_00_bar = self.b2_00_c(b1), self.b2_00_c(b1_bar)
            
            # the velocities
            sigsq = self.sigmav_halo**2
            sigsq_bar = self.sigmav_halo_bar**2
        
            # the PT integrals
            K20_a = self.K20_a(self.k)
            K20s_a = self.K20s_a(self.k)
        
            P00_ss = b1*b1_bar * self.P00.total.mu0 + (b1*b2_00 + b1_bar*b2_00_bar)*self.K00(self.k)
            term1_mu2 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu2            
            term2_mu2 =  -0.5*(self.f*self.k)**2 * (sigsq + sigsq_bar) * P00_ss
            term3_mu2 = 0.5*self.f**2 * ( (b2_00 + b2_00_bar)*K20_a + (bs + bs_bar)*K20s_a )
            P02_ss.total.mu2 = term1_mu2 + term2_mu2 + term3_mu2
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # the nonlinear bias to use
                b2_00, b2_00_bar = self.b2_00_b(b1), self.b2_00_b(b1_bar)
                                    
                # the PT integrals
                K20_b = self.K20_b(self.k)
                K20s_b = self.K20s_b(self.k)
            
                term1_mu4 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu4
                term2_mu4 = self.f**2 * ( (b2_00 + b2_00_bar)*K20_b + (bs + bs_bar)*K20s_b )
                P02_ss.total.mu4 = term1_mu4 + term2_mu4
        
        return P02_ss
            
    @cached_property("_ib1", "_ib1_bar", "max_mu", "P11")
    def P11_ss(self):
        """
        The auto-correlation of the halo momentum field, which 
        contributes mu^2 and mu^4 terms to the power expansion. The mu^2 terms
        come from the vector part, while the mu^4 dependence is dominated by
        the scalar part on large scales (linear term) and the vector part
        on small scales.
        """
        b1, b1_bar = self._ib1, self._ib1_bar
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
                term1_mu4 = 0.5*(b1 + b1_bar)*self.P11.total.mu4
                
                # second term is B11 coming from P11
                term2_mu4 = -0.5*((b1 - 1) + (b1_bar-1)) * self.Pvv_jennings(self.k)
                
                # third term is mu^4 part of C11 (at 2-loop)
                I1 = self.Ivvdd_h02(self.k)
                I2 = self.Idvdv_h04(self.k)
                term3_mu4 =  self.f**2 * (I1 + I2) * (b1*b1_bar - 0.5*(b1 + b1_bar))

                P11_ss.total.mu4 = term1_mu4 + term2_mu4 + term3_mu4
        
        return P11_ss
        
    @cached_property("_ib1", "_ib1_bar", "max_mu", "Pdv", "P01", "sigmav_halo", "sigmav_halo_bar", "b2_01_b")
    def P03_ss(self):
        """
        The cross-corelation of halo density with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 terms.
        """            
        b1, b1_bar  = self._ib1, self._ib1_bar    
        P03_ss = PowerTerm()
    
        # do mu^4 term?
        if self.max_mu >= 4:
                                
            # velocites
            sigsq = self.sigmav_halo**2
            sigsq_bar = self.sigmav_halo_bar**2
            P03_ss.total.mu4 = -0.25*(self.f*self.k)**2 * (sigsq + sigsq_bar) * P01_mu2(self, self.b2_01_b)
                
        return P03_ss
            
    @cached_property("_ib1", "_ib1_bar", "max_mu", "Pdv", "P01", "sigmav_halo", "sigmav_halo_bar", "b2_01_b")
    def P12_ss(self):
        """
        The correlation of halo momentum and halo kinetic energy density, which 
        contributes mu^4 and mu^6 terms to the power expansion.
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        P12_ss = PowerTerm()
    
        # do mu^4 terms?
        if self.max_mu >= 4:
            Plin = self.normed_power_lin(self.k)
                                            
            # the velocities
            sigsq = self.sigmav_halo**2
            sigsq_bar = self.sigmav_halo_bar**2
        
            term1_mu4 = self.P12.total.mu4
            term2_mu4 = -0.5*((b1 - 1) + (b1_bar - 1))*self.f**3*self.I03(self.k)
            term3_mu4 = -0.25*(self.f*self.k)**2 * (sigsq + sigsq_bar) * (P01_mu2(self, self.b2_01_b) - self.P01.total.mu2)
            P12_ss.total.mu4 = term1_mu4 + term2_mu4 + term3_mu4
                    
            # do mu^6 terms?
            if self.max_mu >= 6:
            
                # get the integral attributes
                I21 = self.I21(self.k)
                I30 = self.I30(self.k)
                J20 = self.J20(self.k)
            
                P12_ss.total.mu6 = self.f**3 * (I21 - 0.5*(b1+b1_bar)*I30 + 2*self.k**2*J20*Plin)
        
        return P12_ss
            
    @cached_property("P11_ss", "sigmav_halo", "sigmav_halo_bar")
    def P13_ss(self):
        """
        The cross-correlation of halo momentum with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 and mu^6 terms.
        """
        P13_ss = PowerTerm()
        
        # velocities
        sigsq = self.sigmav_halo**2
        sigsq_bar = self.sigmav_halo_bar**2
    
        # the amplitude
        A = -(self.f*self.k)**2
    
        # do mu^4 terms?
        if self.max_mu >= 4:
        
            # using P11_ss mu^2 terms 
            P13_ss.total.mu4 = 0.5*A*(sigsq + sigsq_bar)*self.P11_ss.total.mu2
    
            # do mu^6 terms?
            if self.max_mu >= 6:
            
                # using P11_ss mu^4 terms
                P13_ss.total.mu6 = 0.5*A*(sigsq + sigsq_bar)*self.P11_ss.total.mu4
        
        return P13_ss
            
    @cached_property("_ib1", "_ib1_bar", "P00", "P02", "P22", "sigmav_halo", "sigmav_halo_bar", "b2_00_d")
    def P22_ss(self):
        """
        The auto-corelation of halo kinetic energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear 
        contributions here. 
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        b2_00, b2_00_bar = self.b2_00_d(b1), self.b2_00_d(b1_bar)
        P22_ss = PowerTerm()
    
        # do mu^4 terms?
        if self.max_mu >= 4:
        
            # velocities in units of Mpc/h
            sigsq = self.sigmav_halo**2
            sigsq_bar = self.sigmav_halo_bar**2

            # 1-loop P22bar
            term1 = self.P22.no_velocity.mu4
        
            # add convolution to P22bar
            term2 = 0.5*(self.f*self.k)**4 * (b1*b1_bar * self.P00.total.mu0) * self.sigmasq_k(self.k)**2
        
            # b1 * P02_bar
            term3 = -0.25*(self.k*self.f)**2 * (sigsq + sigsq_bar) * ( 0.5*(b1 + b1_bar)*self.P02.no_velocity.mu2)
        
            # sigma^4 x P00_ss
            P00_ss = b1*b1_bar * self.P00.total.mu0 + (b1*b2_00 + b1_bar*b2_00_bar)*self.K00(self.k)
            term4 = 0.125*(self.k*self.f)**4 * (sigsq**2 + sigsq_bar**2) * P00_ss
        
            P22_ss.total.mu4 = term1 + term2 + term3 + term4
        
            # do mu^6 terms?
            if self.max_mu >= 6:
            
                term1 = self.P22.no_velocity.mu6
                term2 = -0.25*(self.k*self.f)**2 * (sigsq + sigsq_bar) * (0.5*(b1 + b1_bar)*self.P02.no_velocity.mu4)
                P22_ss.total.mu6 = term1 + term2
                
        return P22_ss
            
    @cached_property("_ib1", "_ib1_bar", "P00", "P02", "sigmav_halo", "sigmav_halo_bar", "b2_00_d")
    def P04_ss(self):
        """
        The cross-correlation of halo density with the rank four tensor field
        ((1+delta_h)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        b2_00, b2_00_bar = self.b2_00_d(b1), self.b2_00_d(b1_bar)
        P04_ss = PowerTerm()
    
        # do mu^4 terms?
        if self.max_mu >= 4:
        
            # velocities in Mpc/h
            sigsq = self.sigmav_halo**2
            sigsq_bar = self.sigmav_halo_bar**2
        
            # contribution from P02[mu^2]
            term1 = -0.125*(b1 + b1_bar)*(self.f*self.k)**2 * (sigsq + sigsq_bar) * self.P02.no_velocity.mu2
        
            # contribution here from P00_ss * vel^4
            P00_ss = b1*b1_bar * self.P00.total.mu0 + (b1*b2_00 + b1_bar*b2_00_bar)*self.K00(self.k)
            A = (1./12)*(self.f*self.k)**4 * P00_ss
            term2 = A*(3.*0.5*(sigsq**2 + sigsq_bar**2) + self.velocity_kurtosis)
        
            P04_ss.total.mu4 = term1 + term2
    
            # do mu^6 terms?
            if self.max_mu >= 6:
                P04_ss.total.mu6 = -0.125*(b1 + b1_bar)*(self.f*self.k)**2 * (sigsq + sigsq_bar) * self.P02.no_velocity.mu4
        
        return P04_ss
            
    @cached_property("k", "correct_mu2", "_ib1", "_ib1_bar", "sigma8_z", "f")
    def mu2_model_correction(self):
        """
        The mu2 correction to the model evaluated at `k`
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        corr = PowerTerm()
        
        if self.correct_mu2:
            mean_bias = (b1*b1_bar)**0.5
            params = {'b1':mean_bias, 'sigma8_z':self.sigma8_z, 'k':self.k, 'f':self.f}
            corr.total.mu2 = self.Pmu2_correction(**params)
        return corr
        
    @cached_property("k", "correct_mu4", "_ib1", "_ib1_bar", "sigma8_z", "f")
    def mu4_model_correction(self):
        """
        The mu4 correction to the model evaluated at `k`
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        corr = PowerTerm()
        
        if self.correct_mu4:
            mean_bias = (b1*b1_bar)**0.5
            params = {'b1':mean_bias, 'sigma8_z':self.sigma8_z, 'k':self.k, 'f':self.f}
            corr.total.mu4 = self.Pmu4_correction(**params)
        return corr
    
    #---------------------------------------------------------------------------
    # power as a function of mu
    #---------------------------------------------------------------------------
    @interpolated_property("P00_ss", interp="k")
    def P_mu0(self, k):
        """
        The full halo power spectrum term with no angular dependence. Contributions
        from P00_ss.
        """
        return self.P00_ss.total.mu0

    @interpolated_property("P01_ss", "P11_ss", "P02_ss", "mu2_model_correction", interp="k")
    def P_mu2(self, k):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_ss, P11_ss, and P02_ss.
        """
        P_mu2 = self.P01_ss.total.mu2 + self.P11_ss.total.mu2 + self.P02_ss.total.mu2 
        return P_mu2 + self.mu2_model_correction.total.mu2


    @interpolated_property("P11_ss", "P02_ss", "P12_ss", "P22_ss", "P03_ss",
                           "P13_ss", "P04_ss", "mu4_model_correction", interp="k")
    def P_mu4(self, k):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_ss, P02_ss, P12_ss, P03_ss, P13_ss, P22_ss, and P04_ss.
        """
        return self.P11_ss.total.mu4 + self.P02_ss.total.mu4 + \
                self.P12_ss.total.mu4 + self.P03_ss.total.mu4 + \
                self.P22_ss.total.mu4 + self.P13_ss.total.mu4 + \
                self.P04_ss.total.mu4 + self.mu4_model_correction.total.mu4

    
    @interpolated_property("P12_ss", interp="k")
    def P_mu6(self, k):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_ss, P13_ss, P22_ss.
        """
        return self.P12_ss.total.mu6 + 1./8*self.f**4 * self.I32(self.k)

#------------------------------------------------------------------------------
# Power Terms
#------------------------------------------------------------------------------
def P01_mu2(self, b2_01_func):
    """
    The correlation of the halo density and halo momentum fields, which 
    contributes mu^2 terms to the power expansion.
    """ 
    # the bias values to use
    b1, b1_bar = self._ib1, self._ib1_bar
    b2_01, b2_01_bar = b2_01_func(b1), b2_01_func(b1_bar)
    
    # the relevant PT integrals
    K10  = self.K10(self.k)
    K10s = self.K10s(self.k)
    K11  = self.K11(self.k)
    K11s = self.K11s(self.k)

    term1 = (b1*b1_bar) * self.P01.total.mu2
    term2 = -self.Pdv*(b1*(1. - b1_bar) + b1_bar*(1. - b1))
    term3 = self.f*((b2_01 + b2_01_bar)*K10 + (self.bs + self.bs_bar)*K10s )
    term4 = self.f*((b1_bar*b2_01 + b1*b2_01_bar)*K11 + (b1_bar*self.bs + b1*self.bs_bar)*K11s)
    return term1 + term2 + term3 + term4
