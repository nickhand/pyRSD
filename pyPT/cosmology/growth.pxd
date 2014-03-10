cimport numpy as np

cdef extern from "pyPT/cosmology/power.h":
    void D_plus(double *z, int numz, int normed, double *growth) nogil
    void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                    double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                    double W_LAM, int TRANSFER) nogil
    void sigma_r(double *r, double z, int numr, int normed, double *sigma) nogil
    void linear_power(double *k, double z, int numk, double *power) nogil
    void nonlinear_power(double *k, double z, int numk, double *power) nogil
    void dlnsdlnm_integral(double *r, int numr, double *output) nogil
    
cpdef growth_function(object z, bint normed=*, object params=*)   
cpdef growth_rate(object z, object params=*)
cpdef Pk_lin(k_hMpc, z, tf=*, params=*)
cpdef Pk_nonlin(k_hMpc, z, tf=*, params=*)
cpdef mass_variance(object r_Mpch, object z, bint normed=*, object tf=*, object params=*)
cpdef dlnsdlnm(object r_Mpch, object sigma0=*, tf=*, object params=*)