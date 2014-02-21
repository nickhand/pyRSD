cimport numpy as np

cdef class spectrum:
    
    # variables
    cdef readonly int num_threads
    cdef readonly double kmin, kmax
    cdef np.ndarray klin, Plin
    cdef readonly double D, f
    cdef object Plin_func
    
    # functions
    cpdef P00(self, k)
    cpdef P01(self, k)
    cdef np.ndarray Inm_parallel(self, int n, int m, np.ndarray[double, ndim=1] k)