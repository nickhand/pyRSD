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
    
cdef class cosmology:
    
    cdef object pdict
    cdef float f_baryon
    
    cpdef E(self, z)
    cpdef omega_m_z(self, z)
    cpdef omega_l_z(self, z)
    cpdef growth_rate(self, z)
    cpdef growth_factor(self, z)
    
    