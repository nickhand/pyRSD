#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
 integralsIJ.pyx
 pyPT: Cython code for the integrals I_nm(k), J_nm(k) from Appendix D 
       of Vlah et al. 2012
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
from pyPT.power cimport kernels
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
    const char* kernel_name
    gsl_spline * spline1, *spline2
    gsl_interp_accel * acc1, *acc2
    
#-------------------------------------------------------------------------------
cdef double inner2D(double x,  void * params) nogil:
    """
    The inner integrand of the 2D integral which is the
    integral over x = cos(theta), weighted by the specified kernel.
    """
    cdef double k_minus_q, kern, q
    cdef double power1 
    cdef double power2
    cdef double result
    
    # read the extra arguments
    cdef fparams * p = (<fparams *> params)
    
    q = exp(p.lnq)
    k_minus_q = sqrt(q*q + p.k*p.k - 2.*p.k*q*x)
    kern = kernels.kernel(p.kernel_name, q/p.k, x)
        
    # get the spline values for the linear power spectrum, making sure
    # we are within the interpolation domain
    power1 = gsl_spline_eval(p.spline1, q, p.acc1)
    if gsl_isnan(power1):
        power1 = 0.
    
    power2 = gsl_spline_eval(p.spline2, k_minus_q, p.acc2)
    if gsl_isnan(power2):
        power2 = 0.
    
    return q*q*q*kern*power1*power2
#end inner2D
    
#-------------------------------------------------------------------------------
cdef double outer2D(double lnq,  void * params) nogil:
    """
    Compute the outer integral integrating over k
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
    F.function = &inner2D
    F.params = p
    
    # do the integration
    status = gsl_integration_cquad(&F, -1., 1., 0., 1e-4, w, &result, &error, NULL)
    if status and status != GSL_EDIVERGE:
        reason = gsl_strerror(status)
        with gil:
            print "Warning: %s" %(reason)
            
    # free the integration workspace
    gsl_integration_cquad_workspace_free(w)
    return result
#end outer2D

#-------------------------------------------------------------------------------
cdef class integral2D:
    
    def __cinit__(self, kernel_name, kmin, kmax, num_threads, k1, P1, k2=None, P2=None):
        
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
            P2 = np.ones(len(P1))
        xarr = np.ascontiguousarray(k2, dtype=np.double)
        yarr = np.ascontiguousarray(P2, dtype=np.double)

        # set up the spline
        self.spline2 = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc2 = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline2, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
        
        # store the kernel
        self.kernel_name = kernel_name
        
        # store the integral specifics
        self.kmin = kmin
        self.kmax = kmax
        self.num_threads = num_threads
        
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
    cpdef evaluate(self, np.ndarray[double, ndim=1] k):
        
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
        with nogil, parallel.parallel(num_threads=self.num_threads):
        
            # allocate and set up the function parameters to pass
            params             = <fparams *>malloc(sizeof(fparams))
            params.lnq         = 0.
            params.kernel_name = self.kernel_name
            params.spline1     = self.spline1
            params.acc1        = self.acc1
            params.spline2     = self.spline2
            params.acc2        = self.acc2
        
            # allocate memory for the integration workspace
            w = <gsl_integration_cquad_workspace *>malloc(sizeof(gsl_integration_cquad_workspace))
            w = gsl_integration_cquad_workspace_alloc(1000)
        
            # allocate and set up the function to pass to the integrator
            F = <gsl_function *>malloc(sizeof(gsl_function))
            F.function = &outer2D
        
            for i in parallel.prange(N):
                
                params.k = k[i]
                F.params = params
        
                # do the integration and store the output
                status = gsl_integration_cquad(F, log(self.kmin), log(self.kmax), 0., 1e-4, w, &result, &error, NULL)
                
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
    
#endclass integral2D

#-------------------------------------------------------------------------------
cdef double integrand1D(double lnq,  void * params) nogil:
    """
    The integrand of the 1D integral, weighted by the specified kernel.
    """
    cdef double q, kern
    cdef double power
    cdef fparams * p = (<fparams *> params)
    q = exp(lnq)
    kern = kernels.kernel(p.kernel_name, q/p.k, 0.)

    # get the spline values for the linear power spectrum
    power = gsl_spline_eval(p.spline1, q, p.acc1)
    if gsl_isnan(power):
        power = 0.
        
    return q*kern*power
#end integrand1D
    
#-------------------------------------------------------------------------------
cdef class integral1D:
    
    def __cinit__(self, kernel_name, kmin, kmax, k1, P1):
        cdef np.ndarray xarr, yarr            
        
        self.acc = NULL
        self.spline = NULL
        self.w = NULL
        
        # set up the power spline
        xarr = np.ascontiguousarray(k1, dtype=np.double)
        yarr = np.ascontiguousarray(P1, dtype=np.double)

        # set up the spline
        self.spline = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
    
        # set up the integration workspace
        self.w = gsl_integration_cquad_workspace_alloc(1000)
        
        # store the kernel
        self.kernel_name = kernel_name
        
        # store the integral specifics
        self.kmin = kmin
        self.kmax = kmax
        
        # turn off the error handler
        gsl_set_error_handler_off()
    #end __cinit__
    
    #---------------------------------------------------------------------------
    def __dealloc__(self):
        if self.spline != NULL:
            gsl_spline_free(self.spline)
        if self.acc != NULL:
            gsl_interp_accel_free(self.acc)
        if self.w != NULL:
            gsl_integration_cquad_workspace_free(self.w)
    #end __dealloc__        
    
    #---------------------------------------------------------------------------
    cpdef evaluate(self, np.ndarray[double, ndim=1] k):
        
        cdef double result, error
        cdef int N = k.shape[0]
        cdef int i, status
        cdef const char * reason
        cdef np.ndarray[double, ndim=1] output = np.empty(N)
        cdef fparams params
        cdef gsl_function F
        
        # set up the params to pass and the pointer
        params.lnq = 0. # this is not needed here
        params.kernel_name = self.kernel_name  
        params.spline1 = self.spline
        params.acc1 = self.acc
        params.spline2 = NULL
        params.acc2 = NULL

        # now set up the gsl integrating function
        F.function = &integrand1D
        
        for i in xrange(N):
            params.k = k[i]
            F.params = &params
            
            status = gsl_integration_cquad(&F, log(self.kmin), log(self.kmax), 0, 1e-4, self.w, &result, &error, NULL)
            if status:
                reason = gsl_strerror(status)
                print "Warning: %s" %reason
            
            output[i] = result / (2.*M_PI**2)
        return output
    #end evaluate
    #----------------------------------------------------------------------------
    
#endclass integral1D 
#-------------------------------------------------------------------------------
