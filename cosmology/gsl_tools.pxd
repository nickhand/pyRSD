cimport numpy as np

cdef extern from "tf_fit.c":
    
    void TFset_parameters(float omega0hh, float f_baryon, float Tcmb) nogil
    float TFfit_onek(float k, float *tf_baryon, float *tf_cdm) nogil
    float TFnowiggles(float omega0, float f_baryon, float hubble, float Tcmb, float k_hmpc) nogil
    

# the integral wrappers    
cpdef np.ndarray compute_sigma_integral(np.ndarray r, object params, object tf)
cpdef np.ndarray compute_growth_integral(np.ndarray z, object params)

# the relevant integrands
cdef double sigma2_integrand_full(double k, void *params) nogil
cdef double sigma2_integrand_nw(double k, void *params) nogil
cdef double growth_function_integrand(double a, void *params) nogil

    

    