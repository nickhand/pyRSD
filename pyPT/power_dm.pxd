cimport numpy as np

cdef class spectrum:
    
    # variables
    cdef readonly int num_threads
    cdef readonly double kmin, kmax
    cdef np.ndarray klin, Plin
    cdef readonly double D, f, conformalH, sigma_v
    cdef object Plin_func
    
    # functions
    cpdef P00(self, k)
    cpdef P_dv(self, k_hMpc)
    cpdef P_vv(self, k_hMpc)
    cpdef P01(self, k)
    cpdef P11(self, k)
    cpdef P11_scalar(self, k)
    cpdef P11_vector(self, k)
    cpdef P02(self, k_hMpc, sigma_02)
    cpdef P02_no_vel(self, k_hMpc)
    cpdef P02_with_vel(self, k_hMpc, sigma_02)
    cdef np.ndarray Inm_parallel(self, int n, int m, np.ndarray[double, ndim=1] k)