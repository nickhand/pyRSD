from cython_gsl cimport *
cimport numpy as np


# define the one-dimensional Fourier integral class
cdef class Fourier1D:

    # variables
    cdef gsl_spline *spline
    cdef gsl_interp_accel *acc
    cdef gsl_integration_workspace *w
    cdef gsl_integration_qawo_table *integ_table
    cdef public int multipole
    cdef public double kmin, kmax
    cdef public double smoothing_radius

    # functions
    cpdef evaluate(self, np.ndarray[double, ndim=1] s)