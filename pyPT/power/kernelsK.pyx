#!python
#cython: cdivision=True
"""
 kernelsK.pyx
 pyPT:  Cython module defining the kernels of the integrals K_nm as 
        described in Vlah et al. 2013.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/10/2014
"""

# define wrapper functions for the individual kernels
cdef double kernel(int n, int m, bint s, double r, double x) nogil:
    if n == 0:
        if m == 0:
            if s:
                return F2(r, x)*S2(r, x)
            else:
                return F2(r, x)
        elif m == 1:
            if s:
                return S2(r, x)*S2(r, x)
            else:
                return 1.
        elif m == 2:
            if s:
                return S2(r, x)
            else:
                return 1.
    elif n == 1:
        if m == 0:
            if s:
                return S2(r, x)*G2(r, x)
            else:
                return G2(r, x)
        elif m == 1:
            if s:
                return x/r*S2(r, x)
            else:
                return x/r
    elif n == 2:
        if m == 0:
            if s:
                return S2(r, x)*h03(r, x)
            else:
                return h03(r, x)
        elif m == 1:
            if s:
                return S2(r, x)*h04(r, x)
            else:
                return h04(r, x)
        elif m == 2:
            return kurtosis(r, x)
#-------------------------------------------------------------------------------

cdef double F2(double r, double x) nogil:
    return (7.*x + 3.*r - 10.*r*x*x)/(14.*r*(1. + r*r - 2.*r*x))

cdef double G2(double r, double x) nogil:
    return (7.*x - r - 6.*r*x*x)/(14.*r*(1. + r*r - 2.*r*x))

cdef double S2(double r, double x) nogil:
    return (r-x)*(r-x)/(1. + r*r - 2.*r*x) - 1./3.
    
cdef double h03(double r, double x) nogil:
    return 0.5*(x*x - 1.)/(1. + r*r - 2.*r*x)

cdef double h04(double r, double x) nogil:
    return 0.5*(1. - 3.*x*x + 2.*x/r)/(1. + r*r - 2.*r*x)

cdef double kurtosis(double r, double x) nogil:
    return 1./(r*r*(1. + r*r - 2.*r*x))