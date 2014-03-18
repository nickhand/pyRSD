#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
 integralsK.pyx
 pyPT: classes to compute the integrals K_nm(k) from Vlah et al. 2013
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/10/2014
"""
from pyPT.power cimport kernelsK
from libc.math cimport exp, sqrt, log, M_PI
from libc.stdlib cimport malloc, free
from cython import parallel

from cython_gsl cimport *
import numpy as np
cimport numpy as np

# define the function parameters the Inm integral
cdef struct fparams: 
    double k
    double lnq
    int n
    int m
    bint s
    gsl_spline * spline1, *spline2
    gsl_interp_accel * acc1, *acc2
    
#-------------------------------------------------------------------------------
cdef double Knm_inner(double x,  void * params) nogil:
    """
    The integrand of the inner integral for K_nm(k), which is the
    integral over x = cos(theta)
    """
    cdef double k_minus_q, kern, q
    cdef double power1 
    cdef double power2
    cdef double result
    
    # read the extra arguments
    cdef fparams * p = (<fparams *> params)
    
    q = exp(p.lnq)
    k_minus_q = sqrt(q*q + p.k*p.k - 2.*p.k*q*x)
    kern = kernelsK.kernel(p.n, p.m, p.s, q/p.k, x)
        
    # get the spline values for the linear power spectrum, making sure
    # we are within the interpolation domain
    power1 = gsl_spline_eval(p.spline1, q, p.acc1)
    if gsl_isnan(power1):
        power1 = 0.
    
    power2 = gsl_spline_eval(p.spline2, k_minus_q, p.acc2)
    if gsl_isnan(power2):
        power2 = 0.
    
    return q*q*q*kern*power1*power2
#end Knm_inner
    
#-------------------------------------------------------------------------------
cdef double Knm_outer(double lnq,  void * params) nogil:
    """
    Compute the outer integral for K_nm(k).
    """
    # define the variables and initialize the integration
    cdef gsl_integration_cquad_workspace * w
    cdef double result, error
    cdef int status
    cdef const char * reason
    w = gsl_integration_cquad_workspace_alloc(1000)

    # update the value of x in the function parameters
    cdef fparams * p = (<fparams *> params)
    p.lnq = lnq
    
    # set up the gsl function for the inner integral
    cdef gsl_function F 
    F.function = &Knm_inner
    F.params = p
    
    # do the integration
    status = gsl_integration_cquad(&F, -1., 1., 0., 1e-4, w, &result, &error, NULL)
    if status and status != GSL_EDIVERGE:
        reason = gsl_strerror(status)
        with gil:
            print "Warning: %s" %reason
            
    # free the integration workspace
    gsl_integration_cquad_workspace_free(w)
    return result
#end Knm_outer

#-------------------------------------------------------------------------------
cdef class K_nm:
    
    def __cinit__(self, n, m, s, k1, P1, k2=None, P2=None):
        cdef np.ndarray xarr, yarr
        self.acc1, self.acc2 = NULL, NULL
        self.spline1, self.spline2 = NULL, NULL
            
        #-----------------------------------------------------------------------
        # set up the two power splines
        #-----------------------------------------------------------------------        
        xarr = np.ascontiguousarray(k1, dtype=np.double)
        yarr = np.ascontiguousarray(P1, dtype=np.double)

        # set up the spline
        self.spline1 = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc1 = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline1, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
        
        if k2 is None or P2 is None:
            k2 = k1
            P2 = P1
        xarr = np.ascontiguousarray(k2, dtype=np.double)
        yarr = np.ascontiguousarray(P2, dtype=np.double)

        # set up the spline
        self.spline2 = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc2 = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline2, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
        
        # store the index
        self.n = n
        self.m = m
        self.s = s
        
        # turn off the error handler
        gsl_set_error_handler_off()
    #end __cinit__
    
    #---------------------------------------------------------------------------
    def __dealloc__(self):
        if self.spline1 != NULL:
            gsl_spline_free(self.spline1)
        if self.acc1 != NULL:
            gsl_interp_accel_free(self.acc1)

        if self.spline2 != NULL:
            gsl_spline_free(self.spline2)
        if self.acc2 != NULL:
            gsl_interp_accel_free(self.acc2)
    #end __dealloc__
    
    #---------------------------------------------------------------------------        
    cpdef evaluate(self, np.ndarray[double, ndim=1] k, double kmin, double kmax, int num_threads):
        
        cdef double result = 0.
        cdef double error = 0.
        cdef int N = k.shape[0]
        cdef int i, status
        cdef const char * reason
        cdef np.ndarray[double, ndim=1] output = np.empty(N)
        cdef gsl_function * F
        cdef gsl_integration_cquad_workspace * w
        cdef fparams * params
        
        # do the integration for all k in parallel
        with nogil, parallel.parallel(num_threads=num_threads):
        
            # allocate and set up the function parameters to pass
            params         = <fparams *>malloc(sizeof(fparams))
            params.lnq     = 0.
            params.n       = self.n
            params.m       = self.m   
            params.s       = self.s     
            params.spline1 = self.spline1
            params.acc1    = self.acc1
            params.spline2 = self.spline2
            params.acc2    = self.acc2
        
            # allocate memory for the integration workspace
            w = <gsl_integration_cquad_workspace *>malloc(sizeof(gsl_integration_cquad_workspace))
            w = gsl_integration_cquad_workspace_alloc(1000)
        
            # allocate and set up the function to pass to the integrator
            F = <gsl_function *>malloc(sizeof(gsl_function))
            F.function = &Knm_outer
        
            for i in parallel.prange(N):
                
                params.k = k[i]
                F.params = params
        
                # do the integration and store the output
                status = gsl_integration_cquad(F, log(kmin), log(kmax), 0., 1e-4, w, &result, &error, NULL)
                
                if status:
                    reason = gsl_strerror(status)
                    with gil:
                        print "Warning: %s" %reason
                
                output[i] = result / (2.*M_PI)**2

            # free all of the memory we allocated
            gsl_integration_cquad_workspace_free(w)
            free(params)
            free(F)
        
        return output
    #end evaluate
    #---------------------------------------------------------------------------
    
#endclass K_nm
#-------------------------------------------------------------------------------
