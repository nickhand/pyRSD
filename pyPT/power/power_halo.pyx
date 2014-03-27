#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
 power_halo.pyx
 pyPT: class implementing the redshift space halo power spectrum using
       the PT expansion outlined in Vlah et al. 2013.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/10/2014
"""
from pyPT.power import integralsPT
from pyPT.cosmology cimport growth, cosmo_tools
from pyPT.power import power_dm

import re
import numpy as np
cimport numpy as np

class HaloSpectrum(power_dm.DMSpectrum):
    
    def __init__(self, stoch_model='constant', 
                        stoch_args=(),
                        **kwargs):
        
        # initalize the dark matter power spectrum
        super(HaloSpectrum, self).__init__(**kwargs)
        
        # don't violate galilean invariance
        self.include_2loop = False
        self.stoch_model = stoch_model
        self.stoch_args  = stoch_args
        
    #end __init__
    #---------------------------------------------------------------------------
    def _delete_all(self):
        """
        Delete all integral and power spectra attributes.
        """
        pattern = re.compile("_([IJKP]([0-9]*[_a-z]*|lin|_mu[0-9]+)|stochasticity)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    # SET ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def stoch_model(self):
        """
        Attribute determining the stochasticity model to use.
        """
        return self._stoch_model
            
    @stoch_model.setter
    def stoch_model(self, val):
        if not callable(val) and val != 'constant' and val != None:
            raise TypeError("Input stochasticity model must a callable function, 'constant', or None")
        self._stoch_model = val
        
        # delete dependencies
        if hasattr(self, '_P00_hh'): del self._P00_hh
        if hasattr(self, '_stochasticity'): del self._stochasticity
    #---------------------------------------------------------------------------
    @property
    def stoch_args(self):
        """
        Any arguments to pass to the stochasticity model function held in 
        ``self.stoch_model``.
        """
        return self._stoch_args
            
    @stoch_args.setter
    def stoch_args(self, val):
        self._stoch_args = val
        
        # delete dependencies
        if hasattr(self, '_P00_hh'): del self._P00_hh
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
            
        # delete P00_hh if it exists
        if hasattr(self, '_P00_hh'): del self._P00_hh
    #---------------------------------------------------------------------------
    # BIAS TERMS 
    #---------------------------------------------------------------------------
    @property
    def b1(self):
        """
        The linear bias factor.
        """
        try:
            return self._b1
        except:
            raise ValueError("Must specify linear bias 'b1' attribute.")
            
    @b1.setter
    def b1(self, val):
        self._b1 = val
            
        # delete terms depending on the bias
        pattern = re.compile("_P[01234]{2}_hh[a-z_]*")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    #---------------------------------------------------------------------------
    @property
    def b2_00(self):
        """
        The quadratic, local bias used for the P00_hh term.
        """
        try:
            return self._b2_00
        except:
            raise ValueError("Must specify quadratic, local bias 'b2_00' attribute.")
            
    @b2_00.setter
    def b2_00(self, val):
        self._b2_00 = val
            
        # delete terms depending on the bias
        pattern = re.compile("_P[01234]{2}_hh[a-z_]*")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    #---------------------------------------------------------------------------
    @property
    def b2_01(self):
        """
        The quadratic, local bias used for the P01_hh term.
        """
        try:
            return self._b2_01
        except:
            raise ValueError("Must specify quadratic, local bias 'b2_01' attribute.")
            
    @b2_01.setter
    def b2_01(self, val):
        self._b2_01 = val
        
        # delete terms depending on the bias
        pattern = re.compile("_P[01234]{2}_hh[a-z_]*")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def bs(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        try:
            return self._bs
        except:
            raise ValueError("Must specify quadratic, nonlocal tidal bias 'bs' attribute.")
            
    @bs.setter
    def bs(self, val):
        self._bs = val
            
        # delete terms depending on the bias
        pattern = re.compile("_P[01234]{2}_hh[a-z_]*")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    # INTEGRAL ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def K00(self):
        try:
            return self._K00
        except:
            self._K00 = self.integrals.I('k00', 0)
            return self._K00
    #---------------------------------------------------------------------------
    @property
    def K00s(self):
        try:
            return self._K00s
        except:
            self._K00s = self.integrals.I('k00s', 0)
            return self._K00s
    #---------------------------------------------------------------------------
    @property
    def K01(self):
        try:
            return self._K01
        except:
            self._K01 = self.integrals.I('k01', 0)
            return self._K01
    #---------------------------------------------------------------------------
    @property
    def K01s(self):
        try:
            return self._K01s
        except:
            self._K01s = self.integrals.I('k01s', 0)
            return self._K01s
    #---------------------------------------------------------------------------
    @property
    def K02s(self):
        try:
            return self._K02s
        except:
            self._K02s = self.integrals.I('k02s', 0)
            return self._K02s
    #---------------------------------------------------------------------------
    @property
    def K10(self):
        try:
            return self._K10
        except:
            self._K10 = self.integrals.I('k10', 0)
            return self._K10
    #---------------------------------------------------------------------------
    @property
    def K10s(self):
        try:
            return self._K10s
        except:
            self._K10s = self.integrals.I('k10s', 0)
            return self._K10s
    #---------------------------------------------------------------------------
    @property
    def K11(self):
        try:
            return self._K11
        except:
            self._K11 = self.integrals.I('k11', 0)
            return self._K11
    #---------------------------------------------------------------------------
    @property
    def K11s(self):
        try:
            return self._K11s
        except:
            self._K00 = self.integrals.I('k00', 0)
            return self._K00
    #---------------------------------------------------------------------------
    @property
    def K20_a(self):
        try:
            return self._K20_a
        except:
            self._K20_a = self.integrals.I('k20_a', 0)
            return self._K20_a
    #---------------------------------------------------------------------------
    @property
    def K20s_a(self):
        try:
            return self._K20s_a
        except:
            self._K20s_a = self.integrals.I('k20s_a', 0)
            return self._K20s_a
    #---------------------------------------------------------------------------
    @property
    def K20_b(self):
        try:
            return self._K20_b
        except:
            self._K20_b = self.integrals.I('k20_b', 0)
            return self._K20_b
    #---------------------------------------------------------------------------
    @property
    def K20s_b(self):
        try:
            return self._K20s_b
        except:
            self._K20s_b = self.integrals.I('k20s_b', 0)
            return self._K20s_b
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def P00_hm(self):
        """
        The isotropic, halo-matter power spectrum.
        """
        try:
            return self._P00_hm
        except:
            # first make sure biases are available
            b1    = self.b1
            b2_00 = self.b2_00
            bs    = self.bs
            
            self._P00_hm = power_dm.PowerTerm()
            self._P00_hm.total.mu0 = b1*self.P00.total.mu0 + (b2_00*self.K00 + bs*self.K00s)
            return self._P00_hm
    #---------------------------------------------------------------------------
    @property
    def P00_hh(self):
        """
        The isotropic, halo-halo power spectrum, including the (possibly
        k-dependent) stochasticity term, as specified by the user.
        """
        try:
            return self._P00_hh
        except:
            # make sure stoch exists first
            stoch = self.stochasticity
            self._P00_hh = power_dm.PowerTerm()
            self._P00_hh.total.mu0 = self.P00_hh_no_stoch.total.mu0 + stoch
            return self._P00_hh
    #---------------------------------------------------------------------------
    @property
    def P00_hh_no_stoch(self):
        """
        The isotropic, halo-halo power spectrum, without any stochasticity term.
        """
        try:
            return self._P00_hh_no_stoch
        except:
            # first make sure biases are available
            b1    = self.b1
            b2_00 = self.b2_00
            bs    = self.bs
            
            self._P00_hh_no_stoch = power_dm.PowerTerm()
            term1 = b1**2 * self.P00.total.mu0
            term2 = 2*b1*(b2_00*self.K00 + bs*self.K00s)
            self._P00_hh_no_stoch.total.mu0 = term1 + term2
            
            return self._P00_hh_no_stoch
    #---------------------------------------------------------------------------
    @property 
    def P01_hh(self):
        """
        The correlation of the halo density and halo momentum fields, which 
        contributes mu^2 terms to the power expansion.
        """
        try:
            return self._P01_hh
        except:
            # first make sure biases are available
            b1    = self.b1
            b2_01 = self.b2_01
            bs    = self.bs
            
            self._P01_hh = power_dm.PowerTerm()

            # do mu^2 terms?
            if self.max_mu >= 2:
                
                term1 = b1**2 * self.P01.total.mu2
                term2 = -2*b1*(1. - b1)*self.Pdv
                term3 =  2.*self.f*(b2_01*self.K10 + bs*self.K10s)
                term4 = 2.*self.f*b1*(b2_01*self.K11 + bs*self.K11s)
        
                self._P01_hh.total.mu2 = term1 + term2 + term3 + term4
            return self._P01_hh
    #---------------------------------------------------------------------------
    @property 
    def P02_hh(self):
        """
        The correlation of the halo density and halo kinetic energy, which 
        contributes mu^2 and mu^4 terms to the power expansion.
        """
        try:
            return self._P02_hh
        except:
            # first make sure biases are available
            b1    = self.b1
            b2_00 = self.b2_00
            bs    = self.bs
            
            self._P02_hh = power_dm.PowerTerm()
            
            # do mu^2 terms?
            if self.max_mu >= 2:
                
                term1_mu2 = b1*self.P02.no_velocity.mu2            
                term2_mu2 =  -(self.f*self.D*self.k*self.sigma_lin)**2 * self.P00_hh_no_stoch.total.mu0
                term3_mu2 = self.f**2 * (b2_00*self.K20_a + bs*self.K20s_a)
                self._P02_hh.total.mu2 = term1_mu2 + term2_mu2 + term3_mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    term1_mu4 = b1*self.P02.no_velocity.mu4
                    term2_mu4 = self.f**2 * (b2_00*self.K20_b + bs*self.K20s_b)
                    self._P02_hh.total.mu4 = term1_mu4 + term2_mu4
            return self._P02_hh
    #---------------------------------------------------------------------------
    @property 
    def P11_hh(self):
        """
        The auto-correlation of the halo momentum field, which 
        contributes mu^2 and mu^4 terms to the power expansion. The mu^2 terms
        come from the vector part, while the mu^4 dependence is dominated by
        the scalar part on large scales (linear term) and the vector part
        on small scales.
        """
        try:
            return self._P11_hh
        except:
            # first make sure linear bias is available
            b1 = self.b1
            
            self._P11_hh = power_dm.PowerTerm()
 
            # do mu^2 terms?
            if self.max_mu >= 2:
                
                # this is C11 at 2-loop order
                I1 = self.integrals.I('h01', 1, ('dd', 'vv'))
                I2 = self.integrals.I('h03', 1, ('dv', 'dv'))
                self._P11_hh.total.mu2 = (self.f*b1)**2 * (I1 + I2)
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    Plin = self.D**2 * self.Plin
                    
                    # first term is mu^4 part of P11
                    term1_mu4 = self.P11.total.mu4
                    
                    # second term is B11 coming from P11
                    term2_mu4 = 2*(b1-1)*self.f**2 * (6.*self.k**2*Plin*self.J10 + 2*self.I22)
                    
                    # third term is mu^4 part of C11 (at 2-loop)
                    I1 = self.integrals.I('h02', 1, ('dd', 'vv'))
                    I2 = self.integrals.I('h04', 1, ('dv', 'dv'))
                    term3_mu4 =  (b1**2 - 1)*self.f**2 * (I1 + I2)

                    self._P11_hh.total.mu4 = term1_mu4 + term2_mu4 + term3_mu4
            return self._P11_hh
    #---------------------------------------------------------------------------
    @property
    def P03_hh(self):
        """
        The cross-corelation of halo density with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 terms.
        """
        try:
            return self._P03_hh
        except:
            self._P03_hh = power_dm.PowerTerm()
            
            # do mu^4 term?
            if self.max_mu >= 4:
                self._P03_hh.total.mu4 = -0.5*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P01_hh.total.mu2
        
            return self._P03_hh
    #---------------------------------------------------------------------------
    @property
    def P12_hh(self):
        """
        The correlation of halo momentum and halo kineticenergy density, which 
        contributes mu^4 and mu^6 terms to the power expansion.
        """
        try:
            return self._P12_hh
        except:
            # first make sure linear bias is available
            b1 = self.b1
            
            self._P12_hh = power_dm.PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                Plin = self.D**2 * self.Plin
                term1_mu4 = self.f**3 * (self.I12 - b1*self.I03 + 2*self.k**2 * self.J02*Plin)
                term2_mu4 = -0.5*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P01_hh.total.mu2
                self._P12_hh.total.mu4 = term1_mu4 + term2_mu4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    self._P12_hh.total.mu6 = self.f**3 * (self.I21 - b1*self.I30 + 2*self.k**2*self.J20*Plin)
            
            return self._P12_hh
    #---------------------------------------------------------------------------
    @property
    def P13_hh(self):
        """
        The cross-correlation of halo momentum with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 and mu^6 terms.
        """
        try:
            return self._P13_hh
        except:
            self._P13_hh = power_dm.PowerTerm()
            
            A = -(self.f*self.D*self.k*self.sigma_lin)**2 
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # using P11_hh mu^2 terms 
                self._P13_hh.total.mu4 = A*self.P11_hh.total.mu2
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # using P11_hh mu^4 terms
                    self._P13_hh.total.mu6 = A*self.P11_hh.total.mu4
            
            return self._P13_hh
    #---------------------------------------------------------------------------
    @property
    def P22_hh(self):
        """
        The auto-corelation of halo kinetic energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear 
        contributions here. 
        """
        try:
            return self._P22_hh
        except:
            # first make sure linear bias is available
            b1 = self.b1
            
            self._P22_hh = power_dm.PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
            
                # 1-loop P22bar
                term1 = self.P22.no_velocity.mu4
                
                # add convolution to P22bar
                term2 = 0.5*(self.f*self.k)**4 * (b1**2 * self.Pdd) * self.integrals.sigmasq_k**2
                
                # b1 * P02_bar
                term3 = -0.5*(self.k*self.f*self.D*self.sigma_lin)**2 * (b1*self.P02.no_velocity.mu2)
                
                # sigma^4 x P00_hh
                term4 = 0.25*(self.k*self.f*self.D*self.sigma_lin)**4 * self.P00_hh_no_stoch.total.mu0
                
                self._P22_hh.total.mu4 = term1 + term2 + term3 + term4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    term1 = self.P22.no_velocity.mu6
                    term2 = -0.5*b1*(self.k*self.f*self.D*self.sigma_lin)**2 * self.P02.no_velocity.mu4
                    self._P22_hh.total.mu6 = term1 + term2
                
                    # do mu^8 terms?
                    if self.max_mu >= 8:
                        self._P22_hh.total.mu8 = self.P22.no_velocity.mu8
                        
            return self._P22_hh
    #---------------------------------------------------------------------------
    @property
    def P04_hh(self):
        """
        The cross-correlation of halo density with the rank four tensor field
        ((1+delta_h)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        try:
            return self._P04_hh
        except:
            # first make sure linear bias is available
            b1 = self.b1
            
            self._P04_hh = power_dm.PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # contribution from P02[mu^2]
                term1 = -0.5*b1*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P02.no_velocity.mu2
                
                # contribution here from P00_hh * vel^4
                A = (1./12)*(self.f*self.D*self.k)**4 * self.P00_hh_no_stoch.total.mu0
                vel_kurtosis = self.integrals.vel_kurtosis / self.D**4
                term2 = A*(3*self.sigma_lin**4 + vel_kurtosis)
                
                self._P04_hh.total.mu4 = term1 + term2
            
                # do mu^6 terms?
                if self.max_mu >= 6:
                    self._P04_hh.total.mu6 = -0.5*b1*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P02.no_velocity.mu4
                
            return self._P04_hh
    #---------------------------------------------------------------------------
    @property
    def P_mu0(self):
        """
        The full halo power spectrum term with no angular dependence. Contributions
        from P00_hh.
        """
        return self._P_mu0
    #---------------------------------------------------------------------------
    @property
    def P_mu2(self):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_hh, P11_hh, and P02_hh.
        """
        return self.P01_hh.total.mu2 + self.P11_hh.total.mu2 + self.P02_hh.total.mu2
    #---------------------------------------------------------------------------
    @property
    def P_mu4(self):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_hh, P02_hh, P12_hh, P03_hh, P13_hh, P22_hh, and P04_hh.
        """
        return self.P11_hh.total.mu4 + self.P02_hh.total.mu4 + self.P12_hh.total.mu4 + \
                self.P22_hh.total.mu4 + self.P03_hh.total.mu4 + self.P13_hh.total.mu4 + \
                self.P04_hh.total.mu4
    #---------------------------------------------------------------------------
    @property
    def P_mu6(self):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_hh, P13_hh, P22_hh.
        """
        return self.P12_hh.total.mu6 + self.P22_hh.total.mu6 + self.P13_hh.total.mu6
    #---------------------------------------------------------------------------
    @property
    def P_mu8(self):
        """
        The full power spectrum term with mu^8 angular dependence. Contributions
        from P22_hh. 
        """
        try:
            return self._P_mu8
        except:
            if self.include_2loop:
                self._P_mu8 = self.P22_hh.total.mu8
            else:
                self._P_mu8 = np.zeros(len(self.k))
            return self._P_mu8
    #---------------------------------------------------------------------------
#enclass HaloPowerSpectrum       

        
