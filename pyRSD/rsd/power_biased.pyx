"""
 power_biased.pyx
 pyRSD: class implementing the redshift space power spectrum using
        the PT expansion and bias model outlined in Vlah et al. 2013.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/10/2014
"""
from . import power_dm, tools
from pyRSD.cosmology cimport growth, cosmo_tools

import numpy as np

class BiasedSpectrum(power_dm.DMSpectrum):
    
    
    _power_atts = ['_P00_ss', '_P00_ss_no_stoch', '_P01_ss', '_P11_ss', 
                   '_P02_ss', '_P12_ss', '_P22_ss', '_P03_ss', '_P13_ss', 
                   '_P04_ss', '_stochasticity']
                
    def __init__(self, stoch_model='constant', stoch_args=(), **kwargs):
        
        # initalize the dark matter power spectrum
        super(BiasedSpectrum, self).__init__(**kwargs)
        
        # don't violate galilean invariance.
        self._include_2loop = False
        
    #end __init__
    #---------------------------------------------------------------------------
    def _delete_power(self):
        """
        Delete all integral and power spectra attributes.
        """
        atts = power_dm.DMSpectrum._power_atts + BiasedSpectrum._power_atts
        atts = [a for a in atts if a in self.__dict__]
        for a in atts:
            del self.__dict__[a]
    #---------------------------------------------------------------------------
    # SET ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def stoch_model(self):
        """
        Attribute determining the stochasticity model to use.
        """
        return lambda k, *args: args[0] + args[1]*np.log(k)
            
    @stoch_model.setter
    def stoch_model(self, val):
        if not callable(val) and val != 'constant' and val != None:
            raise TypeError("Input stochasticity model must a callable function, 'constant', or None")
        self._stoch_model = val
        
        # delete dependencies
        if hasattr(self, '_P00_ss'): del self._P00_ss
        if hasattr(self, '_stochasticity'): del self._stochasticity
    #---------------------------------------------------------------------------
    @property
    def stoch_args(self):
        """
        Any arguments to pass to the stochasticity model function held in 
        ``self.stoch_model``.
        """
        return tools.stochasticity(np.sqrt(self.b1*self.b1_bar), self.z)
            
    @stoch_args.setter
    def stoch_args(self, val):
        self._stoch_args = val
        
        # delete dependencies
        if hasattr(self, '_P00_ss'): del self._P00_ss
        if hasattr(self, '_stochasticity'): del self._stochasticity
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
                self._stochasticity = self.stoch_model(self.k, *self.stoch_args)
                return self._stochasticity
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
        self._b1 = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
            
        # also delete the other biases, that could depend on b1
        for a in ['_b2_00', '_b2_01', '_bs']:
            if a in self.__dict__: del self.__dict__[a]
    
    #---------------------------------------------------------------------------
    @property
    def b2_00(self):
        """
        The quadratic, local bias used for the P00_ss term.
        """
        try:
            return self._b2_00
        except:
            return tools.b2_00(self.b1, self.z)[0]
            #raise ValueError("Must specify quadratic, local bias 'b2_00' attribute.")
            
    @b2_00.setter
    def b2_00(self, val):
        self._b2_00 = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
    
    #---------------------------------------------------------------------------
    @property
    def b2_01(self):
        """
        The quadratic, local bias used for the P01_ss term.
        """
        try:
            return self._b2_01
        except:
            return tools.b2_01(self.b1, self.z)[0]
            #raise ValueError("Must specify quadratic, local bias 'b2_01' attribute.")
            
    @b2_01.setter
    def b2_01(self, val):
        self._b2_01 = val
        
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
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
        self._bs = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
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
        self._b1_bar = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
            
        # also delete the other biases, that could depend on b1
        for a in ['_b2_00_bar', '_b2_01_bar', '_bs_bar']:
            if a in self.__dict__: del self.__dict__[a]
    
    #---------------------------------------------------------------------------
    @property
    def b2_00_bar(self):
        """
        The quadratic, local bias used for the P00_ss term.
        """
        try:
            return self._b2_00_bar
        except:
            return tools.b2_00(self.b1_bar, self.z)[0]
            #raise ValueError("Must specify quadratic, local bias 'b2_00' attribute.")
            
    @b2_00_bar.setter
    def b2_00_bar(self, val):
        self._b2_00_bar = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
    
    #---------------------------------------------------------------------------
    @property
    def b2_01_bar(self):
        """
        The quadratic, local bias used for the P01_ss term.
        """
        try:
            return self._b2_01_bar
        except:
            return tools.b2_01(self.b1_bar, self.z)[0]
            #raise ValueError("Must specify quadratic, local bias 'b2_01' attribute.")
            
    @b2_01_bar.setter
    def b2_01_bar(self, val):
        self._b2_01_bar = val
        
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
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
        self._bs_bar = val
            
        # delete terms depending on the bias
        for a in BiasedSpectrum._power_atts:
            if a in self.__dict__: del self.__dict__[a]
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
            K00 = self.integrals.I('k00', 0)
            K00s = self.integrals.I('k00s', 0)
            
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
                K10 = self.integrals.I('k10', 0)
                K10s = self.integrals.I('k10s', 0)
                K11 = self.integrals.I('k11', 0)
                K11s = self.integrals.I('k11s', 0)
                
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
                
                # get the integral attributes
                K20_a = self.integrals.I('k20_a', 0)
                K20s_a = self.integrals.I('k20s_a', 0)
                
                term1_mu2 = 0.5*(b1 + b1_bar) * self.P02.no_velocity.mu2            
                term2_mu2 =  -(self.f*self.D*self.k*self.sigma_lin)**2 * self.P00_ss_no_stoch.total.mu0
                term3_mu2 = 0.5*self.f**2 * ( (b2_00 + b2_00_bar)*K20_a + (bs + bs_bar)*K20s_a )
                self._P02_ss.total.mu2 = term1_mu2 + term2_mu2 + term3_mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    # get the integral attributes
                    K20_b = self.integrals.I('k20_b', 0)
                    K20s_b = self.integrals.I('k20s_b', 0)
                    
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
                I1 = self.integrals.I('h01', 1, ('vv', 'dd'))
                I2 = self.integrals.I('h03', 1, ('dv', 'dv'))
                self._P11_ss.total.mu2 = (b1*b1_bar)*self.f**2 * (I1 + I2)
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    Plin = self.D**2 * self.power_lin.power
                    
                    # get the integral attributes
                    I22 = self.integrals.I('f22', 0)
                    J10 = self.integrals.J('g10')
                    
                    # first term is mu^4 part of P11
                    term1_mu4 = self.P11.total.mu4
                    
                    # second term is B11 coming from P11
                    term2_mu4 = (b1 + b1_bar - 2.)*self.f**2 * (6.*self.k**2*Plin*J10 + 2*I22)
                    
                    # third term is mu^4 part of C11 (at 2-loop)
                    I1 = self.integrals.I('h02', 1, ('vv', 'dd'))
                    I2 = self.integrals.I('h04', 1, ('dv', 'dv'))
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
                self._P03_ss.total.mu4 = -0.5*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P01_ss.total.mu2
        
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
                Plin = self.D**2 * self.power_lin.power
                
                # get the integral attributes
                I12 = self.integrals.I('f12', 0)
                I03 = self.integrals.I('f03', 0)
                J02 = self.integrals.J('g02')
                
                term1_mu4 = self.f**3 * (I12 - 0.5*(b1 + b1_bar)*I03 + 2*self.k**2 * J02*Plin)
                term2_mu4 = -0.5*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P01_ss.total.mu2
                self._P12_ss.total.mu4 = term1_mu4 + term2_mu4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # get the integral attributes
                    I21 = self.integrals.I('f21', 0)
                    I30 = self.integrals.I('f30', 0)
                    J20 = self.integrals.J('g20')
                    
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
            
            A = -(self.f*self.D*self.k*self.sigma_lin)**2 
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # using P11_ss mu^2 terms 
                self._P13_ss.total.mu4 = A*self.P11_ss.total.mu2
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # using P11_ss mu^4 terms
                    self._P13_ss.total.mu6 = A*self.P11_ss.total.mu4
            
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
            
                # 1-loop P22bar
                term1 = self.P22.no_velocity.mu4
                
                # add convolution to P22bar
                term2 = 0.5*(self.f*self.k)**4 * (b1*b1_bar * self.Pdd) * self.integrals.sigmasq_k**2
                
                # b1 * P02_bar
                term3 = -0.5*(self.k*self.f*self.D*self.sigma_lin)**2 * ( 0.5*(b1 + b1_bar)*self.P02.no_velocity.mu2)
                
                # sigma^4 x P00_ss
                term4 = 0.25*(self.k*self.f*self.D*self.sigma_lin)**4 * self.P00_ss_no_stoch.total.mu0
                
                self._P22_ss.total.mu4 = term1 + term2 + term3 + term4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    term1 = self.P22.no_velocity.mu6
                    term2 = -0.5*(self.k*self.f*self.D*self.sigma_lin)**2 * (0.5*(b1 + b1_bar)*self.P02.no_velocity.mu4)
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
                
                # contribution from P02[mu^2]
                term1 = -0.25*(b1 + b1_bar)*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P02.no_velocity.mu2
                
                # contribution here from P00_ss * vel^4
                A = (1./12)*(self.f*self.D*self.k)**4 * self.P00_ss_no_stoch.total.mu0
                vel_kurtosis = self.integrals.vel_kurtosis / self.D**4
                term2 = A*(3.*self.sigma_lin**4 + vel_kurtosis)
                
                self._P04_ss.total.mu4 = term1 + term2
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    self._P04_ss.total.mu6 = -0.25*(b1 + b1_bar)*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P02.no_velocity.mu4
                
            return self._P04_ss
    #---------------------------------------------------------------------------
    @property
    def P_mu0(self):
        """
        The full halo power spectrum term with no angular dependence. Contributions
        from P00_ss.
        """
        return self.P00_ss.total.mu0
    #---------------------------------------------------------------------------
    @property
    def P_mu2(self):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_ss, P11_ss, and P02_ss.
        """
        return self.P01_ss.total.mu2 + self.P11_ss.total.mu2 + self.P02_ss.total.mu2
    #---------------------------------------------------------------------------
    @property
    def P_mu4(self):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_ss, P02_ss, P12_ss, P03_ss, P13_ss, P22_ss, and P04_ss.
        """
        return self.P11_ss.total.mu4 + self.P02_ss.total.mu4 + self.P12_ss.total.mu4 + \
                self.P22_ss.total.mu4 + self.P03_ss.total.mu4 + self.P13_ss.total.mu4 + \
                self.P04_ss.total.mu4
    #---------------------------------------------------------------------------
    @property
    def P_mu6(self):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_ss, P13_ss, P22_ss.
        """
        return self.P12_ss.total.mu6 + 1./8*self.f**4 * self.integrals.I('f32', 0)
    #---------------------------------------------------------------------------
#enclass BiasedSpectrum       

        
