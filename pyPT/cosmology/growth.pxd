cimport numpy as np

cdef extern from "pyPT/cosmology/power_tools.h":
    void D_plus(double *z, int numz, int normed, double *growth) nogil
    void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                    double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                    double W_LAM, int TRANSFER) nogil
    
cpdef growth_function(object z, bint normed=*, object params=*)   
cpdef growth_rate(object z, object params=*)
