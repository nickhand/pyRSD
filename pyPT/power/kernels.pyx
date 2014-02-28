"""
 kernels.pyx
 pyPT: Cython from libc.math cimport sqrtmodule defining the kernels of the 
       relevant integrals in the PT expansion. Kernel and integral expressions 
       given in Appendix D of Vlah et al. 2012.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
from libc.math cimport log, fabs

# define wrapper functions for the individual kernels
cdef double f_kernel(int n, int m, double r, double x) nogil:
    if n == 0:
        if m == 0:
            return f00(r, x)
        elif m == 1:
            return f01(r, x)
        elif m == 2:
            return f02(r, x)
        elif m == 3:  
            return f03(r, x)
    elif n == 1:
        if m == 0:
            return f10(r, x)
        elif m == 1:
            return f11(r, x)
        elif m == 2:
            return f12(r, x)
        elif m == 3:  
            return f13(r, x) 
    elif n == 2:
        if m == 0:
            return f20(r, x)
        elif m == 1:
            return f21(r, x)
        elif m == 2:
            return f22(r, x)
        elif m == 3:  
            return f23(r, x)
    elif n == 3:
        if m == 0:
            return f30(r, x)
        elif m == 1:
            return f31(r, x)
        elif m == 2:
            return f32(r, x)
        elif m == 3:  
            return f33(r, x)
#end f_kernel

#-------------------------------------------------------------------------------
cdef double g_kernel(int n, int m, double r) nogil:
    if n == 0:
        if m == 0:
            return g00(r)
        elif m == 1:
            return g01(r)
        elif m == 2:
            return g02(r)
    elif n == 1:
        if m == 0:
            return g10(r)
        elif m == 1:
            return g11(r)
    elif n == 2:
        if m == 0:
            return g20(r)
#end g_kernel

#-------------------------------------------------------------------------------
# define the f_nm(r, x) kernels

cdef double f00(double r, double x ) nogil:
    return ((7*x+3*r-10*r*x*x)/(14*r*(1+r*r-2*r*x)))**2

cdef double f01(double r, double x) nogil: 
    return (7*x+3*r-10*r*x*x)*(7*x-r-6*r*x*x)/(14*r*(1+r*r-2*r*x))**2

cdef double f10(double r, double x) nogil: 
    return x*(7*x+3*r-10*r*x*x)/(14*r*r*(1+r*r-2*r*x))
    
cdef double f11(double r, double x) nogil: 
    return ((7*x-r-6*r*x*x)/(14*r*(1+r*r-2*r*x)))**2
    
cdef double f02(double r, double x) nogil: 
    return (x*x-1)*(7*x+3*r-10*r*x*x)/(14*r*(1+r*r-2*r*x)**2)

cdef double f20(double r, double x) nogil: 
    return (2*x+r-3*r*x*x)*(7*x+3*r-10*r*x*x)/(14*r*r*(1+r*r-2*r*x)**2)

cdef double f12(double r, double x) nogil: 
    return (x*x-1)*(7*x-r-6*r*x*x)/(14*r*(1+r*r-2*r*x)**2)

cdef double f21(double r, double x) nogil: 
    return (2*x+r-3*r*x*x)*(7*x-r-6*r*x*x)/(14*r*r*(1+r*r-2*r*x)**2)

cdef double f22(double r, double x) nogil: 
    return x*(7*x-r-6*r*x*x)/(14*r*r*(1+r*r-2*r*x))

cdef double f03(double r, double x) nogil: 
    return (1-x*x)*(3*r*x-1)/(r*r*(1+r*r-2*r*x))

cdef double f30(double r, double x) nogil: 
    return (1-3*x*x-3*r*x+5*r*x*x*x)/(r*r*(1+r*r-2*r*x))

cdef double f31(double r, double x) nogil: 
    return (1-2*r*x)*(1-x*x)/(2*r*r*(1+r*r-2*r*x))

cdef double f13(double r, double x) nogil: 
    return (4*r*x+3*x*x-6*r*x*x*x-1)/(2*r*r*(1+r*r-2*r*x))

cdef double f23(double r, double x) nogil: 
    return 3*((1-x*x)/(1+r*r-2*r*x))**2

cdef double f32(double r, double x) nogil: 
    return (1-x*x)*(2-12*r*x-3*r*r+15*r*r*x*x)/(r*r*(1+r*r-2*r*x)**2)

cdef double f33(double r, double x) nogil: 
    return (-4+12*x*x+24*r*x-40*r*x*x*x+3*r*r-30*r*r*x*x+35*r*r*x*x*x*x)/(r*r*(1+r*r-2*r*x)**2)
#-------------------------------------------------------------------------------

# define the g_nm(r) kernels
cdef double g00(double r) nogil: 
    return (1./3024.)*(12./(r*r) - 158. + 100.*r*r - 42.*r*r*r*r + 3./(r*r*r)*(r*r-1.)**3*(7.*r*r+2.)*log((r+1.)/fabs(r-1.)))
 
cdef double g01(double r) nogil:        
    return (1./3024.)*(24./(r*r) - 202. + 56.*r*r - 30.*r*r*r*r + 3./(r*r*r)*(r*r-1.)**3*(5.*r*r+4.)*log((r+1.)/fabs(r-1.))) 
 
cdef double g10(double r) nogil:
    return (1./1008.)*(-38. + 48.*r*r - 18.*r*r*r*r + 9./r*(r*r-1.)**3*log((r+1.)/fabs(r-1.)))
    
cdef double g11(double r) nogil:
    return (1./1008.)*(12./(r*r) - 82. + 4.*r*r - 6.*r*r*r*r + 3./(r*r*r)*(r*r-1.)**3*(r*r+2.)*log((r+1.)/fabs(r-1.))) 

cdef double g02(double r) nogil:
    return (1./224.)*(2./(r*r)*(r*r+1.)*(3.*r*r*r*r - 14.*r*r + 3.) - 3./(r*r*r)*(r*r-1.)**4*log((r+1.)/fabs(r-1.))) 
 
cdef double g20(double r) nogil:
    return (1./672.)*(2./(r*r)*(9. - 109.*r*r + 63.*r**4 - 27.*r**6) + 9./(r*r*r)*(r*r-1.)**3*(3*r*r+1.)*log((r+1.)/fabs(r-1.))) 

#-------------------------------------------------------------------------------
