cimport numpy as np

cdef extern from "tf_fit.c":
    
    void TFset_parameters(float omega0hh, float f_baryon, float Tcmb)
    float TFfit_onek(float k, float *tf_baryon, float *tf_cdm) 

    void TFfit_hmpc(float omega0, float f_baryon, float hubble, float Tcmb,
    	int numk, float *k, float *tf_full, float *tf_baryon, float *tf_cdm)

    float TFsound_horizon_fit(float omega0, float f_baryon, float hubble)
    float TFk_peak(float omega0, float f_baryon, float hubble)
    float TFnowiggles(float omega0, float f_baryon, float hubble,
    		float Tcmb, float k_hmpc)
    float TFzerobaryon(float omega0, float hubble, float Tcmb, float k_hmpc)
    
cdef class power_eh:
    
    cdef readonly object pdict
    cdef double P0_full, P0_nw
    
    cpdef E(self, z)
    cpdef H(self, z)
    cpdef omega_m_z(self, z)
    cpdef omega_l_z(self, z)
    cpdef growth_rate(self, z)
    cpdef growth_factor(self, z)
    cpdef Pk_full(self, k, z)
    cpdef Pk_nowiggles(self, k, z)
    cpdef np.ndarray Tk_full(self, np.ndarray k)
    cpdef np.ndarray Tk_nowiggles(self, np.ndarray k)
    
    