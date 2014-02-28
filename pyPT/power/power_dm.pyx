#!python
#cython: boundscheck=False
#cython: wraparound=False
"""
 power_dm.pyx
 pyPT: class implementing the redshift space dark matter power spectrum using
       the PT expansion outlined in Vlah et al. 2012.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
from pyPT.power cimport integralsIJ
from pyPT.cosmology cimport cosmo_tools, linear_growth 
from ..cosmology import cosmo, velocity, hmf

from scipy.signal import convolve
import numpy as np
cimport numpy as np

#-------------------------------------------------------------------------------
cdef class Spectrum:
        
    def __init__(self, z, 
                      kmin=1e-3, 
                      kmax=10., 
                      num_threads=1, 
                      cosmo_params="Planck1_lens_WP_highL", 
                      mass_function_kwargs={'mf_fit' : 'Tinker'}, 
                      bias_model='Tinker',
                      include_2loop=False):
        """
        Parameters
        ----------
        z : float
            The redshift to compute the power spectrum at
        kmin : float, optional
            The wavenumber in h/Mpc defining the minimum value to compute 
            integrals to. Default is kmin = 1e-3 h/Mpc.
        kmax : float, optional
            The wavenumber in h/Mpc defining the maximum value to compute 
            integrals to. Default is 10 h/Mpc.
        num_threads : int, optional
            The number of threads to use in parallel computation. Default = 1.
        cosmo_params : {str, dict, cosmo.Cosmology}
            The cosmological parameters to use. Default is Planck 2013 parameters.
        mass_function_kwargs : dict, optional
            The keyword arguments to pass to the hmf.HaloMassFunction object
            for use in computing small-scale velocity dispersions in the halo
            model
        bias_model : {'Tinker', 'SMT', 'PS'}
            The name of the halo bias model to use for any halo model 
            calculations
        include_2loop : bool, optional
            If ``True``, include 2-loop contributions in the model terms.
        """
        if not isinstance(cosmo_params, cosmo.Cosmology):
            self.cosmo = cosmo.Cosmology(cosmo_params)
        else:
            self.cosmo = cosmo_params
            
        self.kmin, self.kmax = kmin, kmax
        self.num_threads     = num_threads
        self.include_2loop   = include_2loop
        
        # initialze the linear power spectrum module
        self.klin = np.logspace(np.log(1e-8), np.log(1000.), 10000, base=np.e)
        self.Plin = linear_growth.Pk_full(self.klin, 0., params=self.cosmo)
        
        # now compute useful quantities for later use
        self.z          = z
        self.D          = linear_growth.growth_function(z, normed=True, params=self.cosmo)
        self.f          = linear_growth.growth_rate(z, params=self.cosmo)
        self.conformalH = cosmo_tools.H(z,  params=self.cosmo)/(1.+z)
        
        # initialize the halo mass function
        mass_function_kwargs['z'] = z
        self.hmf                  = hmf.HaloMassFunction(**mass_function_kwargs)
        self.bias_model           = bias_model
        
        # initialize velocity dispersions
        self.__sigma_lin = 0.
        self.__sigma_v2  = 0.
        self.__sigma_bv2 = 0.
        self.__sigma_bv4 = 0.

    #end __init__
    #---------------------------------------------------------------------------
    property sigma_lin:
        """
        The dark matter velocity dispersion, as evaluated in linear theory 
        [units: km/s]
        """
        def __get__(self):
            if self.__sigma_lin > 0.:
                return self.__sigma_lin
            else:
                self.__sigma_lin = velocity.sigmav_lin(self.z, cosmo_params=self.cosmo)
                return self.__sigma_lin
    #---------------------------------------------------------------------------
    property sigma_v2:
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by the velocity squared. [units: km/s]
        """
        def __get__(self):
            if self.__sigma_v2 > 0.:
                return self.__sigma_v2
            else:
                self.__sigma_v2 = velocity.sigma_v2(self.hmf)
                return self.__sigma_v2
    
        def __set__(self, val):
            self.__sigma_v2 = val
    #---------------------------------------------------------------------------
    property sigma_bv2:
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by the bias times velocity squared. [units: km/s]
        """
        def __get__(self):
            if self.__sigma_bv2 > 0.:
                return self.__sigma_bv2
            else:
                self.__sigma_bv2 = velocity.sigma_bv2(self.hmf, self.bias_model)
                return self.__sigma_bv2
    
        def __set__(self, val):
            self.__sigma_bv2 = val
    #---------------------------------------------------------------------------
    property sigma_bv4:
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by the velocity squared. [units: km/s]
        """
        def __get__(self):
            if self.__sigma_bv4 > 0.:
                return self.__sigma_bv4
            else:
                self.__sigma_bv4 = velocity.sigma_bv4(self.hmf, self.bias_model)
                return self.__sigma_bv4
    
        def __set__(self, val):
            self.__sigma_bv4 = val
    
    #---------------------------------------------------------------------------
    def _vectorize(self, x):
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        return x   
    #---------------------------------------------------------------------------
    cpdef P_dv(self, k_hMpc):
        """
        The 1-loop correlation between density and velocity divergence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        # compute the Inm integrals in parallel
        I01s = self.Inm_parallel(0, 1, k)
        
        J01 = integralsIJ.J_nm(0, 1, self.klin, self.Plin)
        J01s = np.array([J01.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        P11  = -self.f*self.conformalH*Plin
        P22  = -2*self.f*self.conformalH*I01s
        P13  = -3*self.f*self.conformalH*k*k*Plin*J01s
        
        return self.D**2*P11 + self.D**4*(P22 + 2*P13)
    #end P_dv
    
    #---------------------------------------------------------------------------
    cpdef P_vv(self, k_hMpc):
        """
        The 1-loop autocorrelation of velocity divergence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        # compute the Inm integrals in parallel
        I11s = self.Inm_parallel(1, 1, k)
        
        J11 = integralsIJ.J_nm(1, 1, self.klin, self.Plin)
        J11s = np.array([J11.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        fact = (self.f*self.conformalH)**2
        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        P11  = fact*Plin
        P22  = 2*fact*I11s
        P13  = 3*fact*k*k*Plin*J11s
        
        return self.D**2*P11 + self.D**4*(P22 + 2*P13)
    #end P_v
    
    #---------------------------------------------------------------------------    
    cpdef P00(self, k_hMpc):
        """
        The isotropic, zero-order term in the power expansion, corresponding
        to the density field autocorrelation
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        # compute the I00 integrals in parallel
        I00s = self.Inm_parallel(0, 0, k)
        
        # compute the J00 integrals in serial
        J00 = integralsIJ.J_nm(0, 0, self.klin, self.Plin)
        J00s = np.array([J00.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        # compute each term separately
        P11 = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        P22 = 2*I00s
        P13 = 3*k**2*P11*J00s
        
        return self.D**2*P11 + self.D**4*(P22 + 2*P13)
    #end P00
    #---------------------------------------------------------------------------
    cpdef P01(self, k_hMpc):
        """
        The correlation of density and momentum density, which contributes
        mu^2 terms to the power expansion.
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        # compute the I00 integrals in parallel
        I00s = self.Inm_parallel(0, 0, k)
        
        # compute the J00 integrals in serial
        J00 = integralsIJ.J_nm(0, 0, self.klin, self.Plin)
        J00s = np.array([J00.evaluate(ik, self.kmin, self.kmax) for ik in k])

        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        fact = 2.*self.f*self.D**2
        return fact*(Plin + 4.*self.D**2*(I00s + 3*k*k*J00s*Plin))
    #end P01
    
    #---------------------------------------------------------------------------
    cpdef P11(self, k_hMpc):
        """
        The autocorrelation of momentum density, which has a scalar portion 
        which contributes mu^4 terms and a vector term which contributes
        mu^2*(1-mu^2) terms to the power expansion. This is the last term to
        contain a linear contribution.
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        P11_scalar = self.P11_scalar(k)
        P11_vector = self.P11_vector(k)
        
        return P11_scalar, P11_vector
          
    #---------------------------------------------------------------------------      
    cpdef P11_scalar(self, k_hMpc):
        """
        Scalar part of P11, contributing mu^4 terms.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)

        # compute the I00 integrals in parallel
        I00s = self.Inm_parallel(0, 0, k)
        
        # compute the J00 integrals in serial
        J00 = integralsIJ.J_nm(0, 0, self.klin, self.Plin)
        J00s = np.array([J00.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        # compute each term separately
        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        return self.f**2*self.D**2*(Plin + self.D**2*(8.*I00s + 18.*J00s*k*k*Plin))
    #end P11_scalar
    
    #---------------------------------------------------------------------------
    cpdef P11_vector(self, k_hMpc):
        """
        Vector part of P11, contributing mu^2 * (1 + mu^2) terms.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        # compute the Inm integrals in parallel
        I31s = self.Inm_parallel(3, 1, k)
        
        # compute each term separately
        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        return self.f**2*self.D**4*I31s
    #end P11_vector
    
    #---------------------------------------------------------------------------
    cpdef P02(self, k_hMpc):
        """
        The correlation of density and energy density, which contributes
        mu^2 and mu^4 terms to the power expansion. No linear contribution. 
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        P02_vel = self.P02_with_vel(k_hMpc)
        P02_no_vel = self.P02_no_vel(k_hMpc)
        
        return P02_no_vel, P02_vel
    #end P02
  
    #---------------------------------------------------------------------------
    cpdef P02_no_vel(self, k_hMpc):
        """
        Part of P02 that does not contain a velocity dispersion factor. The
        first term returned has mu^2 dependence and the second has mu^4
        dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        # compute the Inm integrals in parallel
        I02s = self.Inm_parallel(0, 2, k)
        I20s = self.Inm_parallel(2, 0, k)
        
        # compute the J00 integrals in serial
        J02 = integralsIJ.J_nm(0, 2, self.klin, self.Plin)
        J02s = np.array([J02.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        J20 = integralsIJ.J_nm(2, 0, self.klin, self.Plin)
        J20s = np.array([J20.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        # compute each term separately
        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        P02_mu2 = (self.f*self.D**2)**2 * (I02s + 2*k*k*J02s*Plin)
        P02_mu4 = (self.f*self.D**2)**2 * (I20s + 2*k*k*J20s*Plin)
        
        return P02_mu2, P02_mu4
    #end P02_no_vel
    
    #---------------------------------------------------------------------------
    cpdef P02_with_vel(self, k_hMpc):
        """
        Part of P02 that does contain a velocity dispersion factor. It has a
        mu^2 angular dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        sigma_lin = self.sigma_lin
        sigma_02  = self.sigma_bv2
        P00       = self.P00(k)
        vel_terms = sigma_lin**2 + (sigma_02/(self.f*self.conformalH*self.D))**2
        
        return -(self.f*self.D*k)**2 * vel_terms * P00
    #end P02_with_vel
    
    #---------------------------------------------------------------------------
    cpdef P12(self, k_hMpc):
        """
        The correlation of momentum density and energy density, which contributes
        mu^4 and mu^6 terms to the power expansion. No linear contribution. 
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        P12_vel    = self.P12_with_vel(k_hMpc)
        P12_no_vel = self.P12_no_vel(k_hMpc)
        
        return P12_no_vel, P12_vel
    #end P12
  
    #---------------------------------------------------------------------------
    cpdef P12_no_vel(self, k_hMpc):
        """
        Part of P12 that does not contain a velocity dispersion factor. The
        first term returned has mu^4 dependence and the second has mu^6
        dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        # compute the Inm integrals in parallel
        I12s = self.Inm_parallel(1, 2, k)
        I21s = self.Inm_parallel(2, 1, k)
        
        I03s = self.Inm_parallel(0, 3, k)
        I30s = self.Inm_parallel(3, 0, k)
        
        # compute the J00 integrals in serial
        J02 = integralsIJ.J_nm(0, 2, self.klin, self.Plin)
        J02s = np.array([J02.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        J20 = integralsIJ.J_nm(2, 0, self.klin, self.Plin)
        J20s = np.array([J20.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        # compute each term separately
        Plin = linear_growth.Pk_full(k, 0., params=self.cosmo) 
        P12_mu4 = self.f**3 * self.D**4 * (I12s - I03s + 2*k*k*J02s*Plin)
        P12_mu6 = self.f**3 * self.D**4 * (I21s - I30s + 2*k*k*J20s*Plin)
        
        return P12_mu4, P12_mu6
    #end P12_no_vel
    
    #---------------------------------------------------------------------------
    cpdef P12_with_vel(self, k_hMpc):
        """
        Part of P12 that does contain a velocity dispersion factor. It has a
        mu^4 angular dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        sigma_lin = self.sigma_lin
        sigma_12  = self.sigma_bv2
        P01       = self.P01(k)
        vel_terms = sigma_lin**2 + (sigma_12/(self.f*self.conformalH*self.D))**2
        
        return -0.5*self.f**2*self.D**2*k*k * vel_terms * P01
    #end P12_with_vel
    
    #---------------------------------------------------------------------------
    cpdef P22(self, k_hMpc):
        """
        The autocorelation of energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. No linear contribution. 
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        P22_no_vel = self.P22_no_vel(k_hMpc)
        if self.include_2loop:
            #P22_vel    = self.P22_with_vel(k_hMpc)
            P22_vel = None
        
        if self.include_2loop:
            return P22_no_vel, P22_vel
        else:
            return P22_no_vel
    #end P22
  
    #---------------------------------------------------------------------------
    cpdef P22_no_vel(self, k_hMpc):
        """
        Part of P22 that does not contain a velocity dispersion factor. The
        terms return have angular dependences of mu^4, mu^6, mu^8, respectively.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        # compute the Inm integrals in parallel
        I23s = self.Inm_parallel(2, 3, k)
        I32s = self.Inm_parallel(3, 2, k)
        I33s = self.Inm_parallel(0, 3, k)
        
        # compute each term separately
        A       = 1./16 * (self.f*self.D)**4
        P22_mu4 = A*I23s
        P22_mu6 = A*2.*I32s
        P22_mu8 = A*I33s
        
        return P22_mu4, P22_mu6, P22_mu8
    #end P22_no_vel
    
    #---------------------------------------------------------------------------
    cpdef P22_with_vel(self, k_hMpc):
        """
        Part of P22 that does contain a velocity dispersion factor. It has a
        no angular dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        sigma_lin  = self.sigma_lin
        sigma_22   = self.sigma_bv2
        P00        = self.P00(k)
        P02_no_vel = self.P02_no_vel(k)
        P22_no_vel = self.P22_no_vel(k)
        vel_terms  = sigma_lin**2 + (sigma_22/(self.f*self.conformalH*self.D))**2
        
        # do the convolution
        P22_conv = np.convolve(P22_no_vel, P00, mode='same')
        
        A = self.f*self.conformalH*self.D
        return -2*A**2 * vel_terms * P02_no_vel + A**4 * vel_terms**2 * P00  + P22_conv
    #end P22_with_vel
    
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    cpdef P03(self, k_hMpc):
        """
        The autocorelation of density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^4 and mu^6.
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        P03_vel = self.P03_with_vel(k_hMpc)
        if self.include_2loop:
            #P22_vel    = self.P22_with_vel(k_hMpc)
            P22_vel = None
        
        if self.include_2loop:
            return P22_no_vel, P22_vel
        else:
            return P22_no_vel
    #end P22
  
    #---------------------------------------------------------------------------
    cpdef P03_no_vel(self, k_hMpc):
        """
        Part of P22 that does not contain a velocity dispersion factor. The
        terms return have angular dependences of mu^4, mu^6, mu^8, respectively.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        # compute the Inm integrals in parallel
        I23s = self.Inm_parallel(2, 3, k)
        I32s = self.Inm_parallel(3, 2, k)
        I33s = self.Inm_parallel(0, 3, k)
        
        # compute each term separately
        A       = 1./16 * (self.f*self.D)**4
        P22_mu4 = A*I23s
        P22_mu6 = A*2.*I32s
        P22_mu8 = A*I33s
        
        return P22_mu4, P22_mu6, P22_mu8
    #end P22_no_vel
    
    #---------------------------------------------------------------------------
    cpdef P03_with_vel(self, k_hMpc):
        """
        Part of P03 that does contain a velocity dispersion factor. It has a
        no angular dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        sigma_lin  = self.sigma_lin
        sigma_03   = self.sigma_v2
        vel_terms  = sigma_lin**2 + (sigma_03/(self.f*self.conformalH*self.D))**2
        if self.include_2loop:
            P = self.P01(k)
        else:
            P = linear_growth.Pk_full(k, 0., params=self.cosmo) 
            
            
    #end P03_with_vel
    
    #---------------------------------------------------------------------------
    cdef np.ndarray Inm_parallel(self, int n, int m, np.ndarray[double, ndim=1] k):
        """
        Compute the I_nm integrals at each k, in parallel
        """
        I = integralsIJ.I_nm(n, m, self.klin, self.Plin)
        return I.evaluate(k, self.kmin, self.kmax, self.num_threads)
    #end Inm_parallel
#endclass spectrum    
