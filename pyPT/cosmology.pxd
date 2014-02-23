cimport numpy as np

cdef extern from "tf_fit.c":
    
    void TFset_parameters(float omega0hh, float f_baryon, float Tcmb) nogil
    float TFfit_onek(float k, float *tf_baryon, float *tf_cdm) nogil

    void TFfit_hmpc(float omega0, float f_baryon, float hubble, float Tcmb,
    	int numk, float *k, float *tf_full, float *tf_baryon, float *tf_cdm) nogil

    float TFsound_horizon_fit(float omega0, float f_baryon, float hubble) nogil
    float TFk_peak(float omega0, float f_baryon, float hubble) nogil
    float TFnowiggles(float omega0, float f_baryon, float hubble,
    		float Tcmb, float k_hmpc) nogil
    float TFzerobaryon(float omega0, float hubble, float Tcmb, float k_hmpc) nogil
    


cpdef H(object z, object cosmo=*)
cpdef E(object z, object cosmo=*)
cpdef omega_m_z(object z, object cosmo=*)
cpdef omega_l_z(object z, object cosmo=*)
cpdef growth_rate(object z, object cosmo=*) 
cpdef growth_factor(object z, object cosmo=*)
cpdef Pk_full(object k_hMpc, object z, object cosmo=*)
cpdef Pk_nowiggles(object k_hMpc, object z, object cosmo=*)
cpdef sigma_r(object r_Mpch, object z, bint normed=*, object tf=*, object cosmo=*)