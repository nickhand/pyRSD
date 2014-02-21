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
from cosmology import linear_power


cdef class spectrum:
        
    def __init__(self, z, kmin=1e-3, kmax=1., num_threads=1, cosmo_params='Planck13'):
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
        p = linear_power(tf='EH_full', cosmo_params=cosmo_params)
        self.klin = np.logspace(-5, 1, 10000)
        self.Plin = p.P_k(self.klin, 0.)
        self.D = p.growth_factor(z)
        self.f = p.growth_rate(z)
        self.Plin_func = p.P_k
    #end __init__
        
    #---------------------------------------------------------------------------    
    cpdef P00(self, k):
        """
        The isotropic, zero-order term in the power expansion, corresponding
        to the density field autocorrelation
        
        Parameters
        ----------
        k : {float, np.ndarray}
            the wavenumber(s) in 1/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        if np.isscalar(k):
            k = np.array([k])
        else:
            k = np.array(k)
          
        # compute the I00 integrals in parallel
        I00s = self.Inm_parallel(0, 0, k)
        
        # compute the J00 integrals in serial
        J00 = integralsIJ.J_nm(0, 0, self.klin, self.Plin)
        J00s = np.array([J00.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        # compute each term separately
        P11 = self.Plin_func(k, 0.) 
        P22 = 2*I00s
        P13 = 3*k**2*P11*J00s
        
        return self.D**2*P11 + self.D**4 * (P22 + 2*P13)
    #end P00
    #---------------------------------------------------------------------------
    cpdef P01(self, k):
        """
        The correlation of density and momentum density, which contributes
        mu^2 terms to the power expansion.
        
        Parameters
        ----------
        k : {float, np.ndarray}
            the wavenumber(s) in 1/Mpc to compute the power at
        """
        # handle both scalar and array inputs
        if np.isscalar(k):
            k = np.array([k])
        else:
            k = np.array(k)
          
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
        return fact*Plin, fact*2.*self.D**2*(I01s + I10s + 3*k**2*(J01s + J10s)*Plin)
    #end P01
    
    #---------------------------------------------------------------------------
    cdef np.ndarray Inm_parallel(self, int n, int m, np.ndarray[double, ndim=1] k):
        """
        Compute the I_nm integrals at each k, in parallel
        """
        I = integralsIJ.I_nm(n, m, self.klin, self.Plin)
        return I.evaluate(k, self.kmin, self.kmax, self.num_threads)
    #end Inm_parallel
#endclass spectrum    