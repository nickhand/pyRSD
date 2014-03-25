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
from pyPT.power cimport integralsK, integralsIJ
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
    @property
    def base_spectra(self):
        try:
            return self._base_spectra
        except:
            self._base_spectra = HaloBaseSpectra(self.b1, self.b2_00, self.bs, 
                                                self.z, self._kmin, self._kmax, 
                                                self.cosmo, self.num_threads)
            return self._base_spectra
            
    @base_spectra.deleter
    def base_spectra(self):
        try:
            del self._base_spectra
        except:
            pass
    #---------------------------------------------------------------------------
    # INTEGRAL ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def K00(self):
        try:
            return self._K00
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 0, 0
            self.base_spectra.K.s = False
            self._K00 = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K00
    #---------------------------------------------------------------------------
    @property
    def K00s(self):
        try:
            return self._K00s
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 0, 0
            self.base_spectra.K.s = True
            self._K00s = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K00s
    #---------------------------------------------------------------------------
    @property
    def K01(self):
        try:
            return self._K01
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 0, 1
            self.base_spectra.K.s = False
            self._K01 = self.base_spectra.K.evaluate(self.k,self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K01
    #---------------------------------------------------------------------------
    @property
    def K01s(self):
        try:
            return self._K01s
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 0, 1
            self.base_spectra.K.s = True
            self._K01s = self.base_spectra.K.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._K01s
    #---------------------------------------------------------------------------
    @property
    def K02s(self):
        try:
            return self._K02s
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 0, 2
            self.base_spectra.K.s = True
            self._K02s = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K02s
    #---------------------------------------------------------------------------
    @property
    def K10(self):
        try:
            return self._K10
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 1, 0
            self.base_spectra.K.s = False
            self._K10 = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K10
    #---------------------------------------------------------------------------
    @property
    def K10s(self):
        try:
            return self._K10s
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 1, 0
            self.base_spectra.K.s = True
            self._K10s = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K10s
    #---------------------------------------------------------------------------
    @property
    def K11(self):
        try:
            return self._K11
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 1, 1
            self.base_spectra.K.s = False
            self._K11 = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K11
    #---------------------------------------------------------------------------
    @property
    def K11s(self):
        try:
            return self._K11s
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 1, 1
            self.base_spectra.K.s = True
            self._K11s = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K11s
    #---------------------------------------------------------------------------
    @property
    def K20_a(self):
        try:
            return self._K20_a
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 2, 0
            self.base_spectra.K.s = False
            self._K20_a = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K20_a
    #---------------------------------------------------------------------------
    @property
    def K20s_a(self):
        try:
            return self._K20s_a
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 2, 0
            self.base_spectra.K.s = True
            self._K20s_a = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K20s_a
    #---------------------------------------------------------------------------
    @property
    def K20_b(self):
        try:
            return self._K20_b
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 2, 1
            self.base_spectra.K.s = False
            self._K20_b = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._K20_b
    #---------------------------------------------------------------------------
    @property
    def K20s_b(self):
        try:
            return self._K20s_b
        except:
            self.base_spectra.K.n, self.base_spectra.K.m = 2, 1
            self.base_spectra.K.s = True
            self._K20s_b = self.base_spectra.K.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
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
            self._P00_hm.total.mu0 = b1*self.P00.total.mu0 + self.D**4 * (b2_00*self.K00 + bs*self.K00s)
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
            term2 = 2*b1*self.D**4 * (b2_00*self.K00 + bs*self.K00s)
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
                
                A = 2.*self.f*self.D**4
                term1 = b1**2 * self.P01.total.mu2
                term2 = -2*b1*(1. - b1)*self.Pdv
                term3 =  A*(b2_01*self.K10 + bs*self.K10s)
                term4 = A*b1*(b2_01*self.K11 + bs*self.K11s)
        
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
                term3_mu2 = (self.D**2*self.f)**2 * (b2_00*self.K20_a + bs*self.K20s_a)
                self._P02_hh.total.mu2 = term1_mu2 + term2_mu2 + term3_mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    term1_mu4 = b1*self.P02.no_velocity.mu4
                    term2_mu4 = (self.D**2*self.f)**2 * (b2_00*self.K20_b + bs*self.K20s_b)
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
                
                self._P11_hh.total.mu2 = b1**2 * (self.I04 + self.I40)
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    term1_mu4 = self.P11.total.mu4
                    term2_mu4 = 2.*(b1-1.)* (self.f*self.D**2)**2 * (6.*self.k**2*self.Plin*self.J10 + 2.*self.I22)
                    term3_mu4 =  (b1**2 - 1.)*(self.I14 + self.I41)

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
                term1_mu4 = self.f**3 * self.D**4 * (self.I12 - b1*self.I03 + 2*self.k**2 * self.J02*self.Plin)
                term2_mu4 = -0.5*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P01_hh.total.mu2
                self._P12_hh.total.mu4 = term1_mu4 + term2_mu4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    self._P12_hh.total.mu6 = self.f**3 * self.D**4 * (self.I21 - b1*self.I30 + 2*self.k**2*self.J20*self.Plin)
            
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
            
                term1 = self.P22.no_velocity.mu4
                term2 = -0.25*(self.k*self.f*self.D*self.sigma_lin)**2 * (b1*self.P02.no_velocity.mu2 + self.P02_hh.total.mu2)
                
                A = 0.25 * self.k**4
                convolve_mu0 = self.base_spectra.convolve('k1loop', 'P22bar_mu0', 'k1loop', 'P00_hh', self.k)
                convolve_mu2 = self.base_spectra.convolve('k1loop', 'P22bar_mu2', 'k1loop', 'P00_hh', self.k)
                convolve_mu4 = self.base_spectra.convolve('k1loop', 'P22bar_mu4', 'k1loop', 'P00_hh', self.k)
                term3 = A*(convolve_mu0 + (1./3)*convolve_mu2 + (1./5)*convolve_mu4)
                
                self._P22_hh.total.mu4 = term1 + term2 + term3
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    term1 = self.P22.no_velocity.mu6
                    term2 = -0.25*(self.k*self.f*self.D*self.sigma_lin)**2 * (b1*self.P02.no_velocity.mu4 + self.P02_hh.total.mu4)
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
                
                term1 = -0.5*b1*(self.f*self.D*self.k*self.sigma_lin)**2 * self.P02.no_velocity.mu2
                A = (1./12)*( b1*(self.f*self.D*self.k)**2 )**2 * self.P00_hh_no_stoch.total.mu0
                
                # compute velocity kurtosis (in Mpc/h)
                vel_kurtosis = self.base_spectra.vel_kurtosis / (self.f*self.D)**4
                term2 = A*(3*self.sigma_lin**4 + vel_kurtosis)
                
                # save the total
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
        from P00_hh
        """
        try:
            return self._P_mu0
        except:
            self._P_mu0 = self.P00_hh.total.mu0
            return self._P_mu0
    #---------------------------------------------------------------------------
    @property
    def P_mu2(self):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_hh, P11_hh, and P02_hh.
        """
        try:
            return self._P_mu2
        except:
            self._P_mu2 = self.P01_hh.total.mu2 + self.P11_hh.total.mu2 + self.P02_hh.total.mu2
            return self._P_mu2
    #---------------------------------------------------------------------------
    @property
    def P_mu4(self):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_hh, P02_hh, P12_hh, P03_hh, P13_hh, P22_hh, and P04_hh
        """
        try:
            return self._P_mu4
        except:
            self._P_mu4 = self.P11_hh.total.mu4 + self.P02_hh.total.mu4 + \
                            self.P12_hh.total.mu4 + self.P22_hh.total.mu4 + \
                            self.P03_hh.total.mu4 + self.P13_hh.total.mu4 + self.P04_hh.total.mu4
            return self._P_mu4
    #---------------------------------------------------------------------------
    @property
    def P_mu6(self):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_hh, P22_hh, P13_hh, P04_hh
        """
        try:
            return self._P_mu6
        except:
            self._P_mu6 = self.P12_hh.total.mu6 + self.P22_hh.total.mu6 + \
                            self.P13_hh.total.mu6 + self.P04_hh.total.mu6
            return self._P_mu6
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

#-------------------------------------------------------------------------------
class HaloBaseSpectra(power_dm.DMBaseSpectra):
    """
    Class to holding base power spectra to be used for splines in the 
    computation of PT integrals.
    """
    def __init__(self, b1, b2_00, bs, *args, **kwargs):
        
        self.b1    = b1
        self.b2_00 = b2_00
        self.bs    = bs
        
        # initalize the dark matter power spectrum
        super(HaloBaseSpectra, self).__init__(*args, **kwargs)
    #---------------------------------------------------------------------------
    @property
    def K(self):
        """
        The instance of the ``K`` integrals.
        """
        try:
            return self._K
        except:
            self._K = integralsK.K_nm(0, 0, False, self.klin, self.Plin, k2=None, P2=None)
            return self._K
    #---------------------------------------------------------------------------
    @property
    def P00_hh(self):
        """
        The isotropic halo power term, with no stochasticity.
        """
        try:
            return self._P00_hh
        except:    
            b1    = self.b1
            b2_00 = self.b2_00
            bs    = self.bs
            
            # compute K00
            self.K.n, self.K.m = 0, 0
            self.K.s = False
            K00 = self.K.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)
            
            # compute K00s
            self.K.n, self.K.m = 0, 0
            self.K.s = True
            K00s = self.K.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)
            
            term1 = b1**2 * self.Pdd
            term2 = 2*b1*self.D**4 * (b2_00*K00 + bs*K00s)
            self._P00_hh = term1 + term2

            return self._P00_hh
    #---------------------------------------------------------------------------
        
