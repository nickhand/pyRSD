#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
"""
 _fourier_integrals.pyx
 pyRSD: compute Fourier integrals for use in computing configuration space
        correlation functions from power spectra
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2014
"""
from pyRSD.rsd cimport _kernels
from libc.math cimport exp, M_PI
from cython_gsl cimport *
import numpy as np
cimport numpy as np

# define the function parameterss
cdef struct fparams: 
    double s
    double R
    double (*kernel)(double) nogil
    gsl_spline *spline
    gsl_interp_accel *acc
    
#-------------------------------------------------------------------------------
cdef double integrand1D(double q,  void * params) nogil:
    """
    The integrand of the 1D Fourier integral.
    """
    cdef double kern, power
    cdef fparams * p = (<fparams *> params)
    kern = p.kernel(p.s*q)

    # get the spline values for the linear power spectrum
    power = gsl_spline_eval(p.spline, q, p.acc)
    if gsl_isnan(power):
        power = 0.
        
    return q*q*exp(-q*q*p.R*p.R)*kern*power
#end integrand1D
    
#-------------------------------------------------------------------------------
cdef class Fourier1D:
    
    def __cinit__(self, multipole, kmin, kmax, smoothing_radius, k1, P1):
        cdef np.ndarray xarr, yarr            
        
        self.acc             = NULL
        self.spline          = NULL
        self.integ_table_sin = NULL
        self.integ_table_cos = NULL
        self.w               = NULL
        
        # set up the power spline
        xarr = np.ascontiguousarray(k1, dtype=np.double)
        yarr = np.ascontiguousarray(P1, dtype=np.double)

        # set up the spline
        self.spline = gsl_spline_alloc(gsl_interp_cspline, xarr.shape[0])
        self.acc    = gsl_interp_accel_alloc()
        gsl_spline_init(self.spline, <double*>xarr.data, <double*>yarr.data, xarr.shape[0])
    
        # set up the integration workspace and tables
        self.integ_table_sin = gsl_integration_qawo_table_alloc(1., kmax-kmin, GSL_INTEG_SINE, 1000)
        self.integ_table_cos = gsl_integration_qawo_table_alloc(1., kmax-kmin, GSL_INTEG_COSINE, 1000)
        self.w = gsl_integration_workspace_alloc(1000)
        
        # store the integral specifics
        self.kmin            = kmin
        self.kmax            = kmax
        self.smoothing_radius = smoothing_radius
        self.multipole       = multipole 
        
        # turn off the error handler
        gsl_set_error_handler_off()
    #end __cinit__
    
    #---------------------------------------------------------------------------
    def __dealloc__(self):
        if self.spline != NULL:
            gsl_spline_free(self.spline)
        if self.acc != NULL:
            gsl_interp_accel_free(self.acc)
        if self.integ_table_sin != NULL:
            gsl_integration_qawo_table_free(self.integ_table_sin)
        if self.integ_table_cos != NULL:
            gsl_integration_qawo_table_free(self.integ_table_cos)
        if self.w != NULL:
            gsl_integration_workspace_free(self.w)
    #end __dealloc__        
    
    #---------------------------------------------------------------------------
    cpdef evaluate(self, np.ndarray[double, ndim=1] s):
        
        cdef double result1 = 0.
        cdef double result2 = 0.
        cdef double error1, error2
        cdef int N = s.shape[0]
        cdef int i, status
        cdef const char * reason
        cdef np.ndarray[double, ndim=1] output = np.empty(N)
        cdef fparams params
        cdef gsl_function F

        # set up the params to pass and the pointer 
        params.spline      = self.spline
        params.acc         = self.acc
        params.R           = self.smoothing_radius

        # now set up the gsl integrating function
        F.function = &integrand1D
        
        # loop over each s value
        for i in xrange(N):
            
            # set up monopole/quadrupole params
            if self.multipole == 0:
                params.kernel = _kernels.j0_sin
            elif self.multipole == 2:
                params.kernel = _kernels.j2_sin
            
            # do the sine integration first
            params.s = s[i]
            F.params = &params
            gsl_integration_qawo_table_set(self.integ_table_sin, s[i], self.kmax-self.kmin, GSL_INTEG_SINE)
            
            # the actual integration
            status = gsl_integration_qawo(&F, self.kmin, 0, 1e-4, 1000, self.w, self.integ_table_sin, &result1, &error1)
            if status:
                reason = gsl_strerror(status)
                print "Warning: %s" %reason
                
            # also do the cosine integration for quadrupole
            if self.multipole == 2:
                params.kernel = _kernels.j2_cos

                # do the sine integration first
                params.s = s[i]
                F.params = &params
                gsl_integration_qawo_table_set(self.integ_table_cos, s[i], self.kmax-self.kmin, GSL_INTEG_COSINE)

                # the actual integration
                status = gsl_integration_qawo(&F, self.kmin, 0, 1e-4, 1000, self.w, self.integ_table_cos, &result2, &error2)
                if status:
                    reason = gsl_strerror(status)
                    print "Warning: %s" %reason
            
            output[i] = (result1 + result2) / (2.*M_PI**2)
        return output
    #end evaluate
    #----------------------------------------------------------------------------
    
#endclass Fourier1D 
#-------------------------------------------------------------------------------
