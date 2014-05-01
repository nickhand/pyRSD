from cython_gsl cimport *
cimport numpy as np

# define the two-dimensional integral class
cdef class integral2D:
    
    # variables
    cdef gsl_spline *spline1
    cdef gsl_spline *spline2
    cdef gsl_interp_accel *acc1
    cdef gsl_interp_accel *acc2
    cdef public const char * kernel_name
    cdef double kmin, kmax
    cdef int num_threads 
    
    # functions
    cpdef evaluate(self, np.ndarray[double, ndim=1] k)
    
# define the one-dimensional integral class
cdef class integral1D:

    # variables
    cdef gsl_spline *spline
    cdef gsl_interp_accel *acc
    cdef gsl_integration_cquad_workspace *w
    cdef public const char * kernel_name
    cdef public double kmin, kmax

    # functions
    cpdef evaluate(self, np.ndarray[double, ndim=1] k)