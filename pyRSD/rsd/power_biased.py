"""
 power_biased.py
 pyRSD: class implementing the redshift space power spectrum using
        the PT expansion and bias model outlined in Vlah et al. 2013.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/10/2014
"""
from . import power_dm, tools
from .. import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class BiasedSpectrum(power_dm.DMSpectrum):
    
    allowable_kwargs = power_dm.DMSpectrum.allowable_kwargs + ['sigma_from_sims']
    _power_atts = ['_P00_ss', '_P00_ss_no_stoch', '_P01_ss', '_P11_ss', 
                   '_P02_ss', '_P12_ss', '_P22_ss', '_P03_ss', '_P13_ss', 
                   '_P04_ss']
                
    def __init__(self, sigma_from_sims=True, **kwargs):
        
        # initalize the dark matter power spectrum
        super(BiasedSpectrum, self).__init__(**kwargs)
        
        # don't violate galilean invariance.
        self._include_2loop = False
        
        # whether to use sigma_v from simulations
        self.sigma_from_sims = sigma_from_sims
        
    #end __init__
    
    #---------------------------------------------------------------------------
    def _delete_power(self):
        """
        Delete all integral and power spectra attributes.
        """
        # delete the power attributes
        for a in power_dm.DMSpectrum._power_atts + BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
            
        self._delete_splines()
        
    #end _delete_power    
    
    #---------------------------------------------------------------------------
    # SET ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def stoch_model(self):
        """
        Attribute determining the stochasticity model to use.
        """
        try:
            return self._stoch_model
        except:
            return lambda k, *args: args[0] + args[1]*np.log(k)
            
    @stoch_model.setter
    def stoch_model(self, val):
        if not callable(val) and val != 'constant' and val != None:
            raise TypeError("Input stochasticity model must a callable function, 'constant', or None")
        self._stoch_model = val
        
        # delete dependencies
        if hasattr(self, '_P00_ss'): del self._P00_ss
        if hasattr(self, '_stochasticity'): del self._stochasticity
        if hasattr(self, '_P_mu0_spline'): del self._P_mu0_spline
        
    #---------------------------------------------------------------------------
    @property
    def stoch_args(self):
        """
        Any arguments to pass to the stochasticity model function held in 
        ``self.stoch_model``.
        """
        try:
            return self._stoch_args
        except:
            return self.default_stoch_args(np.sqrt(self.b1*self.b1_bar), self.z)
            
    @stoch_args.setter
    def stoch_args(self, val):
        self._stoch_args = val
        
        # delete dependencies
        if hasattr(self, '_P00_ss'): del self._P00_ss
        if hasattr(self, '_stochasticity'): del self._stochasticity
        if hasattr(self, '_P_mu0_spline'): del self._P_mu0_spline
    
    #---------------------------------------------------------------------------
    @property
    def default_stoch_args(self):
        """
        The default stochasticity parameters (constant and sloe), which are
        computed using simulation results to interpolate lambda as a 
        function of bias and redshift
        """
        try:
            return self._default_stoch_args
        except:
            self._default_stoch_args = tools.LambdaStochasticityFits()
            return self._default_stoch_args
        
    #---------------------------------------------------------------------------
    @property
    def stochasticity(self):
        """
        The isotropic stochasticity term due to the discreteness of the halos, 
        i.e., Poisson noise.
        """
        try:
            return self._stochasticity
        except:
            if callable(self.stoch_model):
                return self.stoch_model(self.k, *self.stoch_args)
            else:
                raise ValueError("Stochasticity value must be set, or input function provided.")
    
    @stochasticity.setter
    def stochasticity(self, val):
        
        # don't use a stochasticity model
        if self.stoch_model is None:
            if np.isscalar(val):
                self._stochasticity = np.ones(len(self.k))*val
            else:
                if len(val) != len(self.k):
                    raise ValueError("Input stochasticity has the wrong length.")
                self._stochasticity = val
        
        # use a model constant in k
        elif self.stoch_model == "constant":
            if not np.isscalar(val):
                raise ValueError("Cannot set a 'constant' stochasticity with an array.")
            self._stochasticity = np.ones(len(self.k))*val
        else:
            raise ValueError("Error setting the stochasticity. Check 'stoch_model' attribute.")
            
        # delete P00_ss if it exists
        if hasattr(self, '_P00_ss'): del self._P00_ss
        if hasattr(self, '_P_mu0_spline'): del self._P_mu0_spline
        
    @stochasticity.deleter
    def stochasticity(self):
        if hasattr(self, '_stochasticity'): del self._stochasticity
        if hasattr(self, '_P00_ss'): del self._P00_ss
        if hasattr(self, '_P_mu0_spline'): del self._P_mu0_spline
        
    #---------------------------------------------------------------------------
    def _delete_sigma_depends(self, kind):
        """
        Delete the dependencies of the sigma_v2, sigma_bv2, sigma_bv4
        """
        if kind == 'v2':
            for a in ['_P01', '_P03']:
                if hasattr(self, a): delattr(self, a)
                if hasattr(self, a+"_ss"): delattr(self, a+"_ss")
        elif kind == 'bv2':
            for a in ['_P02', '_P12', '_P13', '_P22']:
                if hasattr(self, a): delattr(self, a)
                if hasattr(self, a+"_ss"): delattr(self, a+"_ss")
        elif kind == 'bv4':
            if hasattr(self, '_P04'): delattr(self, '_P04')
            if hasattr(self, '_P04_ss'): delattr(self, '_P04_ss')
        self._delete_splines()
    
    #---------------------------------------------------------------------------
    @property
    def sigmav_fitter(self):
        """
        Interpolator from simulation data for sigmav
        """
        try:
            return self._sigmav_fitter
        except:
            self._sigmav_fitter = tools.SigmavFits()
            return self._sigmav_fitter
            
    #---------------------------------------------------------------------------
    @property
    def sigma_v(self):
        """
        The velocity dispersion at z = 0. If not provided and ``sigma_from_sims
        = False``, this defaults to the linear theory prediction [units: Mpc/h]
        """
        try: 
            return self._sigma_v
        except AttributeError:
            
            if self.sigma_from_sims:
                mean_bias = np.sqrt(self.b1*self.b1_bar)
                sigma_v = self.sigmav_fitter(mean_bias, self.z)
                sigma_v /= (self.f*self.D*100.) # this should be in Mpc/h now
                
                # need to normalize by the right sigma8
                sigma_v *= (self.sigma8 / 0.807)
                return sigma_v
            else:
                return self.sigma_lin
        
    @sigma_v.setter
    def sigma_v(self, val):
        del self.sigma_v
        self._sigma_v = val
        
            
    @sigma_v.deleter
    def sigma_v(self):
        try:
            del self._sigma_v
        except AttributeError:
            pass
            
        # delete dependencies
        for a in ['_P02', '_P12', '_P22', '_P03', '_P13', '_P04']:
            if hasattr(self, a): delattr(self, a)
            if hasattr(self, a+'_ss'): delattr(self, a+'_ss')
        self._delete_splines()
            
    #---------------------------------------------------------------------------
    @property
    def sigma_from_sims(self):
        """
        Whether to use the velocity dispersion as computed from simulations.
        """
        return self._sigma_from_sims
        
    @sigma_from_sims.setter
    def sigma_from_sims(self, val):

        self._sigma_from_sims = val
        del self.sigma_v      
              
    #---------------------------------------------------------------------------
    # FIRST BIAS TERMS 
    #---------------------------------------------------------------------------
    @property
    def b1(self):
        """
        The linear bias factor of the first tracer.
        """
        try:
            return self._b1
        except:
            raise ValueError("Must specify linear bias 'b1' attribute.")
            
    @b1.setter
    def b1(self, val):
        if hasattr(self, '_b1') and val == self._b1: 
            return
        self._b1 = val
        
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        
        self._delete_splines()
            
        # also delete the other biases, that could depend on b1
        for a in ['_b2_00', '_b2_01', '_bs']:
            if hasattr(self, a): delattr(self, a)
    
    #---------------------------------------------------------------------------
    @property
    def nonlinear_bias_fitter(self):
        """
        Interpolator from simulation data for sigmav
        """
        try:
            return self._nonlinear_bias_fitter
        except:
            self._nonlinear_bias_fitter = tools.NonlinearBiasFits()
            return self._nonlinear_bias_fitter
    
    #---------------------------------------------------------------------------
    @property
    def b2_00(self):
        """
        The quadratic, local bias used for the P00_ss term.
        """
        try:
            return self._b2_00
        except:
            return self.nonlinear_bias_fitter(self.b1, self.z)[0]
            #raise ValueError("Must specify quadratic, local bias 'b2_00' attribute.")
            
    @b2_00.setter
    def b2_00(self, val):
        if hasattr(self, '_b2_00') and val == self._b2_00: 
            return
            
        self._b2_00 = val
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
    
    #---------------------------------------------------------------------------
    @property
    def b2_01(self):
        """
        The quadratic, local bias used for the P01_ss term.
        """
        try:
            return self._b2_01
        except:
            return self.nonlinear_bias_fitter(self.b1, self.z)[1]
            #raise ValueError("Must specify quadratic, local bias 'b2_01' attribute.")
            
    @b2_01.setter
    def b2_01(self, val):
        if hasattr(self, '_b2_01') and val == self._b2_01: 
            return
        
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
            
    #---------------------------------------------------------------------------
    @property
    def bs(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        try:
            return self._bs
        except:
            return -2./7 * (self.b1 - 1.)
            #raise ValueError("Must specify quadratic, nonlocal tidal bias 'bs' attribute.")
            
    @bs.setter
    def bs(self, val):
        if hasattr(self, '_bs') and val == self._bs: 
            return
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
            
    #---------------------------------------------------------------------------
    # THE BIAS TERMS OF THE 2ND (BARRED) TRACER
    #---------------------------------------------------------------------------
    @property
    def b1_bar(self):
        """
        The linear bias factor.
        """
        try:
            return self._b1_bar
        except:
            raise ValueError("Must specify linear bias 'b1_bar' attribute.")
            
    @b1_bar.setter
    def b1_bar(self, val):
        if hasattr(self, '_b1_bar') and val == self._b1_bar: 
            return
        self._b1_bar = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
            
        # also delete the other biases, that could depend on b1
        for a in ['_b2_00_bar', '_b2_01_bar', '_bs_bar']:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
    
    #---------------------------------------------------------------------------
    @property
    def b2_00_bar(self):
        """
        The quadratic, local bias used for the P00_ss term.
        """
        try:
            return self._b2_00_bar
        except:
            return self.nonlinear_bias_fitter(self.b1_bar, self.z)[0]
            #raise ValueError("Must specify quadratic, local bias 'b2_00' attribute.")
            
    @b2_00_bar.setter
    def b2_00_bar(self, val):
        if hasattr(self, '_b2_00_bar') and val == self._b2_00_bar: 
            return
        self._b2_00_bar = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
    
    #---------------------------------------------------------------------------
    @property
    def b2_01_bar(self):
        """
        The quadratic, local bias used for the P01_ss term.
        """
        try:
            return self._b2_01_bar
        except:
            return self.nonlinear_bias_fitter(self.b1_bar, self.z)[1]
            #raise ValueError("Must specify quadratic, local bias 'b2_01' attribute.")
            
    @b2_01_bar.setter
    def b2_01_bar(self, val):
        if hasattr(self, '_b2_01_bar') and val == self._b2_01_bar: 
            return
        self._b2_01_bar = val
        
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
            
    #---------------------------------------------------------------------------
    @property
    def bs_bar(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        try:
            return self._bs_bar
        except:
            return -2./7 * (self.b1_bar - 1.)
            #raise ValueError("Must specify quadratic, nonlocal tidal bias 'bs' attribute.")
            
    @bs_bar.setter
    def bs_bar(self, val):
        if hasattr(self, '_bs_bar') and val == self._bs_bar: 
            return
        self._bs_bar = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
        self._delete_splines()
            
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def P00_ss(self):
        """
        The isotropic, halo-halo power spectrum, including the (possibly
        k-dependent) stochasticity term, as specified by the user.
        """
        try:
            return self._P00_ss
        except:
            # make sure stoch exists first
            stoch = self.stochasticity
            self._P00_ss = power_dm.PowerTerm()
            self._P00_ss.total.mu0 = self.P00_ss_no_stoch.total.mu0 + stoch
            return self._P00_ss
            
    #---------------------------------------------------------------------------
    @property
    def P00_ss_no_stoch(self):
        """
        The isotropic, halo-halo power spectrum, without any stochasticity term.
        """
        try:
            return self._P00_ss_no_stoch
        except:
            # first make sure biases are available
            b1, b1_bar = self.b1, self.b1_bar
            b2_00, b2_00_bar = self.b2_00, self.b2_00_bar
            bs, bs_bar = self.bs, self.bs_bar
            
            # get the integral attributes
            K00 = self.integrals.K00(self.k)
            K00s = self.integrals.K00s(self.k)
            
            self._P00_ss_no_stoch = power_dm.PowerTerm()
            term1 = (b1*b1_bar) * self.P00.total.mu0
            term2 = (b1*b2_00_bar + b1_bar*b2_00)*K00
            term3 = (b1*bs_bar + b1_bar*bs)*K00s
            self._P00_ss_no_stoch.total.mu0 = term1 + term2 + term3
            
            return self._P00_ss_no_stoch
            
    #---------------------------------------------------------------------------
    @property 
    def P01_ss(self):
        """
        The correlation of the halo density and halo momentum fields, which 
        contributes mu^2 terms to the power expansion.
        """
        try:
            return self._P01_ss
        except:
            # first make sure biases are available
            b1, b1_bar = self.b1, self.b1_bar
            b2_01, b2_01_bar = self.b2_01, self.b2_01_bar
            bs, bs_bar = self.bs, self.bs_bar
            
            self._P01_ss = power_dm.PowerTerm()

            # do mu^2 terms?
            if self.max_mu >= 2:
                
                # get the integral attributes
                K10  = self.integrals.K10(self.k)
                K10s = self.integrals.K10s(self.k)
                K11  = self.integrals.K11(self.k)
                K11s = self.integrals.K11s(self.k)
                
                term1 = (b1*b1_bar) * self.P01.total.mu2
                term2 = -self.Pdv * ( b1*(1. - b1_bar) + b1_bar*(1. - b1) )
                term3 = self.f * ( (b2_01 + b2_01_bar)*K10 + (bs + bs_bar)*K10s )
                term4 = self.f * ( (b1_bar*b2_01 + b1*b2_01_bar)*K11 + (b1_bar*bs + b1*bs_bar)*K11s )
        
                self._P01_ss.total.mu2 = term1 + term2 + term3 + term4
            return self._P01_ss
            
    #---------------------------------------------------------------------------
    @property 
    def P02_ss(self):
        """
        The correlation of the halo density and halo kinetic energy, which 
        contributes mu^2 and mu^4 terms to the power expansion.
        """
        try:
            return self._P02_ss
        except:
            # first make sure biases are available
            b1, b1_bar = self.b1, self.b1_bar
            b2_00, b2_00_bar = self.b2_00, self.b2_00_bar
            bs, bs_bar = self.bs, self.bs_bar
            
            self._P02_ss = power_dm.PowerTerm()
            
            # do mu^2 terms?
            if self.max_mu >= 2:
                
                # the mu^2 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_v
                sigma_02  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
                sigsq_eff = sigma_lin**2 + sigma_02**2
                
                # get the integral attributes
                K20_a = self.integrals.K20_a(self.k)
                K20s_a = self.integrals.K20s_a(self.k)
                
                term1_mu2 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu2            
                term2_mu2 =  -(self.f*self.D*self.k)**2 * sigsq_eff * self.P00_ss_no_stoch.total.mu0
                term3_mu2 = 0.5*self.f**2 * ( (b2_00 + b2_00_bar)*K20_a + (bs + bs_bar)*K20s_a )
                self._P02_ss.total.mu2 = term1_mu2 + term2_mu2 + term3_mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    # get the integral attributes
                    K20_b = self.integrals.K20_b(self.k)
                    K20s_b = self.integrals.K20s_b(self.k)
                    
                    term1_mu4 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu4
                    term2_mu4 = 0.5*self.f**2 * ( (b2_00 + b2_00_bar)*K20_b + (bs + bs_bar)*K20s_b )
                    self._P02_ss.total.mu4 = term1_mu4 + term2_mu4
            return self._P02_ss
            
    #---------------------------------------------------------------------------
    @property 
    def P11_ss(self):
        """
        The auto-correlation of the halo momentum field, which 
        contributes mu^2 and mu^4 terms to the power expansion. The mu^2 terms
        come from the vector part, while the mu^4 dependence is dominated by
        the scalar part on large scales (linear term) and the vector part
        on small scales.
        """
        try:
            return self._P11_ss
        except:
            # first make sure linear bias is available
            b1, b1_bar = self.b1, self.b1_bar
            
            self._P11_ss = power_dm.PowerTerm()
 
            # do mu^2 terms?
            if self.max_mu >= 2:
                
                # this is C11 at 2-loop order
                I1 = self.integrals.Ivvdd_h01(self.k)
                I2 = self.integrals.Idvdv_h03(self.k)
                self._P11_ss.total.mu2 = (b1*b1_bar)*self.f**2 * (I1 + I2)
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    Plin = self.normed_power_lin(self.k)
                    
                    # get the integral attributes
                    I22 = self.integrals.I22(self.k)
                    J10 = self.integrals.J10(self.k)
                    
                    # first term is mu^4 part of P11
                    term1_mu4 = self.P11.total.mu4
                    
                    # second term is B11 coming from P11
                    term2_mu4 = (b1 + b1_bar - 2.)*self.f**2 * (6.*self.k**2*Plin*J10 + 2*I22)
                    
                    # third term is mu^4 part of C11 (at 2-loop)
                    I1 = self.integrals.Ivvdd_h02(self.k)
                    I2 = self.integrals.Idvdv_h04(self.k)
                    term3_mu4 =  (b1*b1_bar - 1)*self.f**2 * (I1 + I2)

                    self._P11_ss.total.mu4 = term1_mu4 + term2_mu4 + term3_mu4
            return self._P11_ss
            
    #---------------------------------------------------------------------------
    @property
    def P03_ss(self):
        """
        The cross-corelation of halo density with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 terms.
        """
        try:
            return self._P03_ss
        except:
            self._P03_ss = power_dm.PowerTerm()
            
            # do mu^4 term?
            if self.max_mu >= 4:
                
                # optionally add small scale velocity
                sigma_lin = self.sigma_v 
                sigma_03  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
                sigsq_eff = sigma_lin**2 + sigma_03**2
                
                self._P03_ss.total.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01_ss.total.mu2
        
            return self._P03_ss
            
    #---------------------------------------------------------------------------
    @property
    def P12_ss(self):
        """
        The correlation of halo momentum and halo kinetic energy density, which 
        contributes mu^4 and mu^6 terms to the power expansion.
        """
        try:
            return self._P12_ss
        except:
            # first make sure linear bias is available
            b1, b1_bar = self.b1, self.b1_bar
            
            self._P12_ss = power_dm.PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                Plin = self.normed_power_lin(self.k)
                
                # now do mu^4 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_v  
                sigma_12  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_12**2
                
                # get the integral attributes
                I12 = self.integrals.I12(self.k)
                I03 = self.integrals.I03(self.k)
                J02 = self.integrals.J02(self.k)
                
                term1_mu4 = self.f**3 * (I12 - 0.5*(b1 + b1_bar)*I03 + 2*self.k**2 * J02*Plin)
                term2_mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01_ss.total.mu2
                self._P12_ss.total.mu4 = term1_mu4 + term2_mu4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # get the integral attributes
                    I21 = self.integrals.I21(self.k)
                    I30 = self.integrals.I30(self.k)
                    J20 = self.integrals.J20(self.k)
                    
                    self._P12_ss.total.mu6 = self.f**3 * (I21 - 0.5*(b1+b1_bar)*I30 + 2*self.k**2*J20*Plin)
            
            return self._P12_ss
            
    #---------------------------------------------------------------------------
    @property
    def P13_ss(self):
        """
        The cross-correlation of halo momentum with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 and mu^6 terms.
        """
        try:
            return self._P13_ss
        except:
            self._P13_ss = power_dm.PowerTerm()
            
            # vector small scale velocity additions
            sigma_lin = self.sigma_v 
            sigma_13_v  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
            
            # the amplitude
            A = -(self.f*self.D*self.k)**2
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # using P11_ss mu^2 terms 
                self._P13_ss.total.mu4 = A*sigsq_eff_vector*self.P11_ss.total.mu2
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # scalar small scale velocity additions
                    sigma_13_s  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                    sigsq_eff_scalar = sigma_lin**2 + sigma_13_s**2
                    
                    # using P11_ss mu^4 terms
                    self._P13_ss.total.mu6 = A*sigsq_eff_scalar*self.P11_ss.total.mu4
            
            return self._P13_ss
            
    #---------------------------------------------------------------------------
    @property
    def P22_ss(self):
        """
        The auto-corelation of halo kinetic energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear 
        contributions here. 
        """
        try:
            return self._P22_ss
        except:
            # first make sure linear bias is available
            b1, b1_bar = self.b1, self.b1_bar
            
            self._P22_ss = power_dm.PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # velocities in units of Mpc/h
                sigma_lin = self.sigma_v
                sigma_22  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_22**2
            
                # 1-loop P22bar
                term1 = self.P22.no_velocity.mu4
                
                # add convolution to P22bar
                term2 = 0.5*(self.f*self.k)**4 * (b1*b1_bar * self.Pdd) * self.integrals.sigmasq_k(self.k)**2
                
                # b1 * P02_bar
                term3 = -0.5*(self.k*self.f*self.D)**2 * sigsq_eff * ( 0.5*(b1 + b1_bar)*self.P02.no_velocity.mu2)
                
                # sigma^4 x P00_ss
                term4 = 0.25*(self.k*self.f*self.D)**4 * sigsq_eff**2 * self.P00_ss_no_stoch.total.mu0
                
                self._P22_ss.total.mu4 = term1 + term2 + term3 + term4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    term1 = self.P22.no_velocity.mu6
                    term2 = -0.5*(self.k*self.f*self.D)**2 * sigsq_eff * (0.5*(b1 + b1_bar)*self.P02.no_velocity.mu4)
                    self._P22_ss.total.mu6 = term1 + term2
                
                        
            return self._P22_ss
            
    #---------------------------------------------------------------------------
    @property
    def P04_ss(self):
        """
        The cross-correlation of halo density with the rank four tensor field
        ((1+delta_h)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        try:
            return self._P04_ss
        except:
            # first make sure linear bias is available
            b1, b1_bar = self.b1, self.b1_bar
            
            self._P04_ss = power_dm.PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # compute the relevant small-scale + linear velocities in Mpc/h
                sigma_lin = self.sigma_v 
                sigma_04  = self.sigma_bv4 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_04**2
                
                # contribution from P02[mu^2]
                term1 = -0.25*(b1 + b1_bar)*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                
                # contribution here from P00_ss * vel^4
                A = (1./12)*(self.f*self.D*self.k)**4 * self.P00_ss_no_stoch.total.mu0
                vel_kurtosis = self.integrals.velocity_kurtosis / self.D**4
                term2 = A*(3.*sigsq_eff**2 + vel_kurtosis)
                
                self._P04_ss.total.mu4 = term1 + term2
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    self._P04_ss.total.mu6 = -0.25*(b1 + b1_bar)*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                
            return self._P04_ss
            
    #---------------------------------------------------------------------------
    def P_mu0(self, k):
        """
        The full halo power spectrum term with no angular dependence. Contributions
        from P00_ss.
        """
        try:
            return self._P_mu0_spline(k)
        except AttributeError:
            Pk = self.P00_ss.total.mu0
            self._P_mu0_spline = spline(self.k, Pk)
            return self._P_mu0_spline(k)

    #end P_mu0
    #---------------------------------------------------------------------------
    def P_mu2(self, k):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_ss, P11_ss, and P02_ss.
        """
        try:
            return self._P_mu2_spline(k)
        except AttributeError:
            Pk = self.P01_ss.total.mu2 + self.P11_ss.total.mu2 + self.P02_ss.total.mu2
            self._P_mu2_spline = spline(self.k, Pk)
            return self._P_mu2_spline(k)
    
    #end P_mu2
    #---------------------------------------------------------------------------
    def P_mu4(self, k):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_ss, P02_ss, P12_ss, P03_ss, P13_ss, P22_ss, and P04_ss.
        """
        try:
            return self._P_mu4_spline(k)
        except AttributeError:
            Pk = self.P11_ss.total.mu4 + self.P02_ss.total.mu4 + self.P12_ss.total.mu4 + \
                 self.P22_ss.total.mu4 + self.P03_ss.total.mu4 + self.P13_ss.total.mu4 + \
                 self.P04_ss.total.mu4
                 
            self._P_mu4_spline = spline(self.k, Pk)
            return self._P_mu4_spline(k)
            
        return 
                
    #---------------------------------------------------------------------------
    def P_mu6(self, k):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_ss, P13_ss, P22_ss.
        """
        try:
            return self._P_mu6_spline(k)
        except AttributeError:
            Pk = self.P12_ss.total.mu6 + 1./8*self.f**4 * self.integrals.I32(self.k)
            self._P_mu6_spline = spline(self.k, Pk)
            return self._P_mu6_spline(k)
            
    #end P_mu6
    #---------------------------------------------------------------------------
    
#enclass BiasedSpectrum     
#-------------------------------------------------------------------------------  

        
