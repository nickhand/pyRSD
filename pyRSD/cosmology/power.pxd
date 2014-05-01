cimport numpy as np
from cython_gsl cimport *

cdef extern from "../include/power_tools.h":
    void D_plus(double *z, int numz, int normed, double *growth) nogil
    void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                    double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                    double W_LAM, int TRANSFER) nogil
    void set_CAMB_transfer(double *k, double *Tk, int numk) nogil
    void free_transfer() nogil
    void unnormalized_transfer(double *k, double z, int numk, double *transfer) nogil
    double normalize_power() nogil
    void nonlinear_power(double *k, double z, int numk, double *Delta_L, double *power) nogil
    #void unnormalized_sigma_r(double *r, double z, int numr, double *sigma) nogil
    void dlnsdlnm_integral(double *r, int numr, double *output) nogil