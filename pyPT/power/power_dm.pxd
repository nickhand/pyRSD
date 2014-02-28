cimport numpy as np

cdef class Spectrum:
    
    # variables
    cdef readonly int num_threads
    cdef readonly double kmin, kmax 
    cdef readonly bint include_2loop
    cdef np.ndarray klin, Plin
    cdef readonly double D, f, conformalH, z
    cdef readonly object cosmo
    cdef readonly object hmf, bias_model
    cdef double __sigma_lin, __sigma_v2, __sigma_bv2, __sigma_bv4
    
    # functions
    cpdef P00(self, k)
    cpdef P_dv(self, k_hMpc)
    cpdef P_vv(self, k_hMpc)
    cpdef P01(self, k)
    cpdef P11(self, k)
    cpdef P11_scalar(self, k)
    cpdef P11_vector(self, k)
    cpdef P02(self, k_hMpc)
    cpdef P02_no_vel(self, k_hMpc)
    cpdef P02_with_vel(self, k_hMpc)
    cpdef P12(self, k_hMpc)
    cpdef P12_no_vel(self, k_hMpc)
    cpdef P12_with_vel(self, k_hMpc)
    cpdef P22(self, k_hMpc)
    cpdef P22_no_vel(self, k_hMpc)
    cpdef P22_with_vel(self, k_hMpc)
    cdef np.ndarray Inm_parallel(self, int n, int m, np.ndarray[double, ndim=1] k)