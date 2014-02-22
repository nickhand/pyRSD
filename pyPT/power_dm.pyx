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
cimport integralsIJ
import numpy as np
cimport numpy as np
from cosmology import power_eh


cdef class spectrum:
        
    def __cinit__(self, z, kmin=1e-3, kmax=1., num_threads=1, cosmo_params='Planck13'):
        """
        Parameters
        ----------
        z : float
            the redshift to compute the power spectrum at
        kmin : float, optional
            the wavenumber defining the minimum value to compute integrals to. 
            Linear power spectrum must be defined to at least this value. Default
            is kmin = 1e-3 1/Mpc.
        kmax : float, optional
            the wavenumber defining the minimum value to compute integrals to. 
            Linear power spectrum must be defined to at least this value. Default
            is 1 1/Mpc.
        num_threads : int, optional
            the number of threads to use in parallel computation. Default = 1.
        cosmo_params : dict or str, optional
            dictionary of cosmological parameters or string defining a
            pre-defined cosmology from parameters module. Default is Planck 2013.
        """
        self.kmin, self.kmax = kmin, kmax
        self.num_threads = num_threads
        
        # initialze the linear power spectrum module
        p = power_eh(cosmo_params)
        self.klin = np.logspace(-5, 1, 10000)
        self.Plin = p.Pk_full(self.klin, 0.)
        
        self.D = p.growth_factor(z)
        self.f = p.growth_rate(z)
        self.conformalH = p.H(z)/(1.+z)
        self.Plin_func = p.Pk_full
    #end __init__
    #---------------------------------------------------------------------------
    def _vectorize(self, x):
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        return x   
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
        P11 = self.Plin_func(k, 0.) 
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
          
        # compute the Inm integrals in parallel
        I01s = self.Inm_parallel(0, 1, k)
        I10s = self.Inm_parallel(1, 0, k)
        
        # compute the J00 integrals in serial
        J01 = integralsIJ.J_nm(0, 1, self.klin, self.Plin)
        J01s = np.array([J01.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        J10 = integralsIJ.J_nm(1, 0, self.klin, self.Plin)
        J10s = np.array([J10.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        # compute each term separately
        Plin = self.Plin_func(k, 0.) 
        fact = 2.*self.f*self.D**2
        return fact*(Plin + 2.*self.D**2*(I01s + I10s + 3*k*k*(J01s + J10s)*Plin))
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
        Plin = self.Plin_func(k, 0.) 
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
        Plin = self.Plin_func(k, 0.) 
        return self.f**2*self.D**4*I31s
    #end P11_vector
    
    #---------------------------------------------------------------------------
    cpdef P02(self, k_hMpc, sigma_02):
        """
        The correlation of density and energy density, which contributes
        mu^2 and mu^4 terms to the power expansion. No linear contribution. 
        
        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber(s) in h/Mpc to compute the power at
        sigma_02 : float
            redshift-dependent, small-scale addition to the velocity dispersion
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
        
        P02_vel = self.P02_with_vel(k_hMpc, sigma_02)
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
        Plin = self.Plin_func(k, 0.) 
        P02_mu2 = self.f**2*self.D**4*(I02s + 2*k*k*J02s*Plin)
        P02_mu4 = self.f**2*self.D**4*(I20s + 2*k*k*J20s*Plin)
        
        return P02_mu2, P02_mu4
    #end P02_no_vel
    
    #---------------------------------------------------------------------------
    cpdef P02_with_vel(self, k_hMpc, sigma_02):
        """
        Part of P02 that does contain a velocity dispersion factor. It has a
        mu^2 angular dependence.
        """
        # handle both scalar and array inputs
        k = self._vectorize(k_hMpc)
          
        sigma_v = 1.0
        P00 = self.P00(k)
        vel_terms = sigma_v**2 + (sigma_02/(self.f*self.conformalH*self.D))**2
        
        return -self.f**2*self.D**2*k*k*vel_terms*P00
    #end P02_with_vel
    
    
    #---------------------------------------------------------------------------
    cdef np.ndarray Inm_parallel(self, int n, int m, np.ndarray[double, ndim=1] k):
        """
        Compute the I_nm integrals at each k, in parallel
        """
        I = integralsIJ.I_nm(n, m, self.klin, self.Plin)
        return I.evaluate(k, self.kmin, self.kmax, self.num_threads)
    #end Inm_parallel
#endclass spectrum    