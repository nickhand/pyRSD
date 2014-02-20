from cython_gsl cimport *

# define the integral I_nm class
cdef class I_nm:
    
    # variables
    cdef gsl_spline *spline
    cdef gsl_interp_accel *acc
    cdef gsl_integration_workspace *w
    cdef readonly int n, m
    
    # functions
    cpdef evaluate(self, double k, double kmin, double kmax)

# define the integral J_nm class
cdef class J_nm:

    # variables
    cdef gsl_spline *spline
    cdef gsl_interp_accel *acc
    cdef gsl_integration_workspace *w
    cdef readonly int n, m

    # functions
    cpdef evaluate(self, double k, double kmin, double kmax)