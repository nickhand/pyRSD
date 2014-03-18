from cython_gsl cimport *
cimport numpy as np

# define the integral K_nm class
cdef class K_nm:
    
    # variables
    cdef gsl_spline *spline1, *spline2
    cdef gsl_interp_accel *acc1, *acc2
    cdef public int n, m
    cdef public bint s
    
    # functions
    cpdef evaluate(self, np.ndarray[double, ndim=1] k, double kmin, double kmax, int num_threads)
    