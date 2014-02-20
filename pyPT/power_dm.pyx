"""
 power_dm.pyx
 pyPT: class implementing the redshift space dark matter power spectrum using
       the PT expansion outlined in Vlah et al. 2012.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
cimport integralsIJ
import numpy as np
cimport numpy as np
import scipy.interpolate
from cosmology import linear_power
import scipy.interpolate

cdef class P_k:
        
    def __init__(self, z, kmin=1e-3, kmax=1., num_threads=1):
        
        self.kmin, self.kmax = kmin, kmax
        
        p = linear_power(tf='EH_full')
        self.klin = np.logspace(-5, 1, 10000)
        self.Plin = p.P_k(self.klin, 0.)
        self.D = p.growth_factor(z)
        
        self.num_threads = num_threads
        
    cpdef P00(self, k):
        
        Plin_func = scipy.interpolate.InterpolatedUnivariateSpline(self.klin, self.Plin)
        
        if np.isscalar(k):
            k = np.array([k])
        else:
            k = np.array(k)
           
        I00s = self.P00_parallel(k)
        
        J00 = integralsIJ.J_nm(0, 0, self.klin, self.Plin)
        J00s = np.array([J00.evaluate(ik, self.kmin, self.kmax) for ik in k])
        
        P11 = Plin_func(k) 
        P22 = 2*I00s
        P13 = 3*k**2*P11*J00s
        
        return self.D**2*P11 + self.D**4 * (P22 + 2*P13)
        
    cdef np.ndarray P00_parallel(self, np.ndarray[double, ndim=1] k):
        
        I00 = integralsIJ.I_nm(0, 0, self.klin, self.Plin)
        return I00.evaluate(k, self.kmin, self.kmax, self.num_threads)
    

def test(double k, double kmin, double kmax):
    
    klin, Plin = np.loadtxt("/Users/Nick/Desktop/Plin.dat", unpack=True)
    indices = [(n, m) for n in range(4) for m in range(4)]
    
    for (n, m) in indices:
        I = integralsIJ.I_nm(n, m, klin, Plin)
        print "I%d%d:" %(n, m), I.evaluate(k, kmin, kmax)
    
    print 
    for (n, m) in [(0, 0), (0, 1), (1, 0), (1,1), (0,2), (2,0)]:
        J = integralsIJ.J_nm(n, m, klin, Plin)
        print "J%d%d:" %(n, m), J.evaluate(k, kmin, kmax)
    