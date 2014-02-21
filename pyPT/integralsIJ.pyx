#!python
#cython: boundscheck=False
#cython: wraparound=False
"""
 integralsIJ.pyx
 pyPT: Cython code for the integrals I_nm(k), J_nm(k) from Appendix D 
       of Vlah et al. 2012
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
#import kernels_cy
from libc.math cimport exp, sqrt, log, M_PI
from cython_gsl cimport *
from kernels cimport *
from cython import parallel
import cython

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

# define the function parameters the Inm integral
cdef struct fparams: 
    double kmin
    double kmax
    double k
    double x
    int n
    int m
    gsl_spline * spline
    gsl_interp_accel * acc
    
#-------------------------------------------------------------------------------
cdef double Inm_inner(double lnq,  void * params) nogil:
    """
    Compute the inner integral for I_nm(k)
    """
    cdef double q, k_minus_q, kern
    cdef double power1 = 0.
    cdef double power2 = 0.

    # read the extra arguments
    cdef fparams * p = (<fparams *> params)
        
    q = exp(lnq)
    k_minus_q = sqrt(q*q + p.k*p.k - 2.*p.k*q*p.x)
    kern = f_kernel(p.n, p.m, q/p.k, p.x)
    
    # get the spline values for the linear power spectrum, making sure
    # we are within the interpolation domain
    if q >= p.kmin and q <= p.kmax:
        power1 = gsl_spline_eval(p.spline, q, p.acc)
    
    if k_minus_q >= p.kmin and k_minus_q <= p.kmax:
        power2 = gsl_spline_eval(p.spline, k_minus_q, p.acc)
        
    return q*q*q*kern*power1*power2
#end Inm_inner
    
#-------------------------------------------------------------------------------
cdef double Inm_outer(double x,  void * params) nogil:
    """
    Compute the outer integral for I_nm(k).
    """
    # define the variables and initialize the integration
    cdef gsl_integration_cquad_workspace * w
    cdef double result, error
    w = gsl_integration_cquad_workspace_alloc(1000)

    # update the value of x in the function parameters
    cdef fparams * p = (<fparams *> params)
    p.x = x
    
    # set up the gsl function for the inner integral
    cdef gsl_function F
    F.function = &Inm_inner
    F.params = p

    # do the integration
    gsl_integration_cquad(&F, log(p.kmin), log(p.kmax), 0, 1e-5, w, &result, &error, NULL)

    # free the integration workspace
    gsl_integration_cquad_workspace_free(w)
    return result
#end Inm_outer

#-------------------------------------------------------------------------------
cdef class I_nm:
    
    def __cinit__(self, n, m, klin, Plin):
        cdef np.ndarray xarr, yarr
        
        # check the input values
        if n > 3 or m > 3: 
            raise ValueError("kernel f_nm must have n, m < 4")
            
        self.acc = NULL
        self.spline = NULL
            
        # make the data arrays contiguous in memory for splining
        xarr = np.ascontiguousarray(klin, dtype=np.double)
        yarr = np.ascontiguousarray(Plin, dtype=np.double)

        # set up the spline
        self.spline = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
        
        # store the index
        self.n = n
        self.m = m
    
    def __dealloc__(self):
        if self.spline != NULL:
            gsl_spline_free(self.spline)
        if self.acc != NULL:
            gsl_interp_accel_free(self.acc)
            
    cpdef evaluate(self, np.ndarray[double, ndim=1] k, double kmin, double kmax, int num_threads):
        
        cdef double result = 0.
        cdef double error = 0.
        cdef int N = k.shape[0]
        cdef int i
        cdef np.ndarray[double, ndim=1] output = np.empty(N)
        
        # set up the pointers that we need to allocate explicitly
        cdef fparams *params
        cdef gsl_integration_workspace *w
        cdef gsl_function *F
        
        # do the integration for all k in parallel
        for i in parallel.prange(N, schedule='static', num_threads=num_threads, nogil=True):
        
            # allocate and set up the function parameters to pass
            params = <fparams *>malloc(sizeof(fparams))
            params.kmin = kmin
            params.kmax = kmax
            params.k = k[i]
            params.x = 0.
            params.n = self.n
            params.m = self.m        
            params.spline = self.spline
            params.acc = self.acc
            
            # allocate memory for the integration workspace
            w = gsl_integration_workspace_alloc(1000)
            
            # allocate and set up the function to pass to the integrator
            F = <gsl_function *>malloc(sizeof(gsl_function))
            F.function = &Inm_outer
            F.params = params

            # do the integration and store the output
            gsl_integration_qags(F, -1., 1., 0, 1e-5, 1000, w, &result, &error)
            output[i] = result / (2.*M_PI)**2

            # free all of the memory we allocated
            gsl_integration_workspace_free(w)
            free(params)
            free(F)
        
        return output
    #end evaluate
#endclass I_nm

#-------------------------------------------------------------------------------
cdef double Jnm_integrand(double lnq,  void * params) nogil:
    """
    Compute the integral for J_nm(k).
    """
    cdef double q, kern
    cdef double power = 0.
    cdef fparams * p = (<fparams *> params)
    q = exp(lnq)
    kern = g_kernel(p.n, p.m, q/p.k)

    # get the spline values for the linear power spectrum, making sure
    # we are within the interpolation domain
    if q >= p.kmin and q <= p.kmax:
        power = gsl_spline_eval(p.spline, q, p.acc)

    return q*kern*power
#end Jnm_integand
    
#-------------------------------------------------------------------------------
cdef class J_nm:
    
    def __cinit__(self, n, m, klin, Plin):
        cdef np.ndarray xarr, yarr
        
        # check the input values
        if (n + m > 2): 
            raise ValueError("kernel g_nm must have n, m such that n + m < 3")
            
        self.acc = NULL
        self.spline = NULL
        self.w = NULL
        
        # make the data arrays contiguous in memory for splining
        xarr = np.ascontiguousarray(klin, dtype=np.double)
        yarr = np.ascontiguousarray(Plin, dtype=np.double)

        # set up the spline
        self.spline = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
    
        # set up the integration workspace
        self.w = gsl_integration_workspace_alloc(1000)
        
        # store the index
        self.n = n
        self.m = m
    
    def __dealloc__(self):
        if self.spline != NULL:
            gsl_spline_free(self.spline)
        if self.acc != NULL:
            gsl_interp_accel_free(self.acc)
        if self.w != NULL:
            gsl_integration_workspace_free(self.w)
            
    cpdef evaluate(self, double k, double kmin, double kmax):
        cdef double result, error

        # set up the params to pass and the pointer
        cdef fparams params
        params.kmin = kmin
        params.kmax = kmax
        params.k = k
        params.x = 0. # this is not needed here
        params.n = self.n
        params.m = self.m        
        params.spline = self.spline
        params.acc = self.acc
        cdef fparams * param_ptr = &params

        # now set up the gsl integrating function
        cdef gsl_function F
        F.function = &Jnm_integrand
        F.params = param_ptr

        gsl_integration_qags(&F, log(kmin), log(kmax), 0, 1e-5, 1000, self.w, &result, &error)
        return result / (2.*M_PI**2)
    #end evaluate
#endclass J_nm  
 
