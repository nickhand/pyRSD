"""
 kernels.pxd
 pyPT: define the f_nm, g_nm kernels for importing to other modules
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/19/2014
"""
# the f kernels
cdef double f_kernel(int n, int m, double r, double x) nogil

# the g kernels
cdef double g_kernel(int n, int m, double r) nogil