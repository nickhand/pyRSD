from functools import wraps
from .. import pygcl, numpy as np
from ._cache import parameter, interpolated_property, cached_property
from .tools import RSDSpline as spline
from . import INTERP_KMIN, INTERP_KMAX

#-------------------------------------------------------------------------------
# decorators to properly normalize integrals
#-------------------------------------------------------------------------------
def normalize_Jmn(f):
    """
    Decorator to properly normalize Jmn integrals
    """   
    fget = f.fget
    def wrapper(self):
        def normalized_spline(k):
            return self._power_norm * fget(self)(k)
        return normalized_spline
        
    f.fget = wrapper
    return f

def normalize_Imn(f):
    """
    Decorator to properly normalize Imn integrals
    """   
    fget = f.fget
    def wrapper(self):
        def normalized_spline(k):
            return self._power_norm**2 * fget(self)(k)
        return normalized_spline
        
    f.fget = wrapper
    return f

normalize_Kmn = normalize_Imn
    
def normalize_ImnOneLoop(f):
    """
    Decorator to properly normalize one loop Imn integrals
    """
    fget = f.fget
    def wrapper(self):
        norm = self._power_norm
        def normalized_spline(k):
            terms = fget(self)
            return norm**2*terms[0](k) + norm**3*terms[1](k) + norm**4*terms[2](k)
        return normalized_spline
        
    f.fget = wrapper
    return f

class PTIntegralsMixin(object):
    """
    A mixin class to compute and store the necessary PT integrals for the dark 
    matter redshift space power spectrum. 
    
    Notes
    -----
    The class is written such that the computationally-expensive parts do not 
    depend on changes in sigma8(z) so the integrals can be renormalized to 
    the correct sigma8(z) with an overall scaling
    """        
    #---------------------------------------------------------------------------
    # one-loop power spectra
    #---------------------------------------------------------------------------
    @cached_property("power_lin")
    def _Pdd_0(self):
        """
        The 1-loop density auto spectrum 
        """
        return pygcl.OneLoopPdd(self.power_lin)
    
    @cached_property("power_lin")
    def _Pdv_0(self):
        """
        The 1-loop density-velocity cross spectrum
        """
        return pygcl.OneLoopPdv(self.power_lin)
        
    @cached_property("power_lin")
    def _Pvv_0(self):
        """
        The 1-loop velocity auto spectrum
        """
        return pygcl.OneLoopPvv(self.power_lin)

    @cached_property("power_lin")
    def _P22bar_0(self):
        """
        The 1-loop P22 power spectrum
        """
        return pygcl.OneLoopP22Bar(self.power_lin)
            
    #---------------------------------------------------------------------------
    # drivers for the various PT integrals -- depend on Plin
    #---------------------------------------------------------------------------
    @cached_property("power_lin")
    def _Imn(self):
        """
        The internal driver class to compute the I(m, n) integrals
        """
        return pygcl.Imn(self.power_lin)
        
    @cached_property("power_lin")
    def _Jmn(self):
        """
        The internal driver class to compute the J(m, n) integrals
        """
        return pygcl.Jmn(self.power_lin)

    @cached_property("power_lin")
    def _Kmn(self):
        """
        The internal driver class to compute the J(m, n) integrals
        """
        return pygcl.Kmn(self.power_lin)

    @cached_property("_Pdv_0")
    def _Imn1Loop_dvdv(self):
        """
        The internal driver class to compute the 1-loop I(m, n) integrals, 
        which integrate over `P_dv(q) P_dv(|k-q|)`
        """
        return pygcl.ImnOneLoop(self._Pdv_0)

    @cached_property("_Pvv_0", "_Pdd_0")
    def _Imn1Loop_vvdd(self):
        """
        The internal driver class to compute the 1-loop I(m, n) integrals, 
        which integrate over `P_vv(q) P_dd(|k-q|)`
        """
        return pygcl.ImnOneLoop(self._Pvv_0, self._Pdd_0, 1e-4)

    @cached_property("_Pvv_0")
    def _Imn1Loop_vvvv(self):
        """
        The internal driver class to compute the 1-loop I(m, n) integrals, 
        which integrate over `P_vv(q) P_vv(|k-q|)`
        """
        return pygcl.ImnOneLoop(self._Pvv_0)

    #---------------------------------------------------------------------------
    # Jmn integrals as a function of input k
    #---------------------------------------------------------------------------
    @normalize_Jmn
    @interpolated_property("_Jmn")
    def J00(self, k):
        """J(m=0,n=0) perturbation theory integral"""
        return self._Jmn(k, 0, 0)

    @normalize_Jmn
    @interpolated_property("_Jmn")
    def J01(self, k):
        """J(m=0,n=1) perturbation theory integral"""
        return self._Jmn(k, 0, 1)

    @normalize_Jmn
    @interpolated_property("_Jmn")
    def J10(self, k):
        """J(m=1,n=0) perturbation theory integral"""
        return self._Jmn(k, 1, 0)

    @normalize_Jmn
    @interpolated_property("_Jmn")
    def J11(self, k):
        """J(m=1,n=1) perturbation theory integral"""
        return self._Jmn(k, 1, 1)
           
    @normalize_Jmn
    @interpolated_property("_Jmn")
    def J02(self, k):
        """J(m=0,n=2) perturbation theory integral"""
        return self._Jmn(k, 0, 2)
    
    @normalize_Jmn
    @interpolated_property("_Jmn")        
    def J20(self, k):
        """J(m=2,n=0) perturbation theory integral"""
        return self._Jmn(k, 2, 0)
            
    #---------------------------------------------------------------------------
    # Imn integrals as a function of k
    #---------------------------------------------------------------------------   
    @normalize_Imn
    @interpolated_property("_Imn")
    def I00(self, k):
        """I(m=0,n=0) perturbation theory integral"""
        return self._Imn(k, 0, 0)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I01(self, k):
        """I(m=0,n=1) perturbation theory integral"""
        return self._Imn(k, 0, 1)

    @normalize_Imn
    @interpolated_property("_Imn")
    def I02(self, k):
        """I(m=0,n=2) perturbation theory integral"""
        return self._Imn(k, 0, 2)
        
    @normalize_Imn
    @interpolated_property("_Imn")    
    def I03(self, k):
        """I(m=0,n=3) perturbation theory integral"""
        return self._Imn(k, 0, 3)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I10(self, k):
        """I(m=1,n=0) perturbation theory integral"""
        return self._Imn(k, 1, 0)

    @normalize_Imn
    @interpolated_property("_Imn")    
    def I11(self, k):
        """I(m=1,n=1) perturbation theory integral"""
        return self._Imn(k, 1, 1)
    
    @normalize_Imn
    @interpolated_property("_Imn")    
    def I12(self, k):
        """I(m=1,n=2) perturbation theory integral"""
        return self._Imn(k, 1, 2)

    @normalize_Imn
    @interpolated_property("_Imn")
    def I13(self, k):
        """I(m=1,n=3) perturbation theory integral"""
        return self._Imn(k, 1, 3)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I20(self, k):
        """I(m=2,n=0) perturbation theory integral"""
        return self._Imn(k, 2, 0)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I21(self, k):
        """I(m=2,n=1) perturbation theory integral"""
        return self._Imn(k, 2, 1)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I22(self, k):
        """I(m=2,n=2) perturbation theory integral"""
        return self._Imn(k, 2, 2)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I23(self, k):
        """I(m=2,n=3) perturbation theory integral"""
        return self._Imn(k, 2, 3)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I30(self, k):
        """I(m=3,n=0) perturbation theory integral"""
        return self._Imn(k, 3, 0)
        
    @normalize_Imn
    @interpolated_property("_Imn")
    def I31(self, k):
        """I(m=3,n=1) perturbation theory integral"""
        return self._Imn(k, 3, 1)
    
    @normalize_Imn
    @interpolated_property("_Imn")
    def I32(self, k):
        """I(m=3,n=2) perturbation theory integral"""
        return self._Imn(k, 3, 2)
        
    @normalize_Imn
    @interpolated_property("_Imn")
    def I33(self, k):
        """I(m=3,n=3) perturbation theory integral"""
        return self._Imn(k, 3, 3)

    #---------------------------------------------------------------------------
    # Kmn integrals
    #---------------------------------------------------------------------------
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K00(self, k):
        """K(m=0,n=0) perturbation theory integral"""
        return self._Kmn(k, 0, 0)
    
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K00s(self, k):
        """K(m=0,n=0,s=True) perturbation theory integral"""
        return self._Kmn(k, 0, 0, True)
     
    @normalize_Kmn
    @interpolated_property("_Kmn")         
    def K01(self, k):
        """K(m=0,n=1) perturbation theory integral"""
        return self._Kmn(k, 0, 1)

    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K01s(self, k):
        """K(m=0,n=1,s=True) perturbation theory integral"""
        return self._Kmn(k, 0, 1, True)
            
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K02s(self, k):
        """K(m=0,n=2,s=True) perturbation theory integral"""
        return self._Kmn(k, 0, 2, True)
       
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K10(self, k):
        """K(m=1,n=0) perturbation theory integral"""
        return self._Kmn(k, 1, 0)

    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K10s(self, k):
        """K(m=1,n=0,s=True) perturbation theory integral"""
        return self._Kmn(k, 1, 0, True)
        
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K11(self, k):
        """K(m=1,n=1) perturbation theory integral"""
        return self._Kmn(k, 1, 1)
    
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K11s(self, k):
        """K(m=1,n=1,s=True) perturbation theory integral"""
        return self._Kmn(k, 1, 1, True)

    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K20_a(self, k):
        """K(m=2,n=0) mu^2 perturbation theory integral"""
        return self._Kmn(k, 2, 0, False, 0)
       
    @normalize_Kmn
    @interpolated_property("_Kmn")         
    def K20_b(self, k):
        """K(m=2,n=0) mu^4 perturbation theory integral"""
        return self._Kmn(k, 2, 0, False, 1)

    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K20s_a(self, k):
        """K(m=2,n=0,s=True) mu^2 perturbation theory integral"""
        return self._Kmn(k, 2, 0, True, 0)
        
    @normalize_Kmn
    @interpolated_property("_Kmn")
    def K20s_b(self, k):
        """K(m=2,n=0,s=True) mu^4 perturbation theory integral"""
        return self._Kmn(k, 2, 0, True, 1)
            
    #---------------------------------------------------------------------------
    # full 2-loop integrals
    #---------------------------------------------------------------------------
    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_vvdd")
    def Ivvdd_h01(self, k):        
        I_lin   = self._Imn1Loop_vvdd.EvaluateLinear(k, 0, 1)
        I_cross = self._Imn1Loop_vvdd.EvaluateCross(k, 0, 1)
        I_1loop = self._Imn1Loop_vvdd.EvaluateOneLoop(k, 0, 1)
        
        return I_lin, I_cross, I_1loop

    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_vvdd")
    def Ivvdd_h02(self, k):
        I_lin   = self._Imn1Loop_vvdd.EvaluateLinear(k, 0, 2)
        I_cross = self._Imn1Loop_vvdd.EvaluateCross(k, 0, 2)
        I_1loop = self._Imn1Loop_vvdd.EvaluateOneLoop(k, 0, 2)
        
        return I_lin, I_cross, I_1loop
    
    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_dvdv")
    def Idvdv_h03(self, k):
        I_lin   = self._Imn1Loop_dvdv.EvaluateLinear(k, 0, 3)
        I_cross = self._Imn1Loop_dvdv.EvaluateCross(k, 0, 3)
        I_1loop = self._Imn1Loop_dvdv.EvaluateOneLoop(k, 0, 3)
        
        return I_lin, I_cross, I_1loop
        
    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_dvdv")
    def Idvdv_h04(self, k):
        I_lin   = self._Imn1Loop_dvdv.EvaluateLinear(k, 0, 4)
        I_cross = self._Imn1Loop_dvdv.EvaluateCross(k, 0, 4)
        I_1loop = self._Imn1Loop_dvdv.EvaluateOneLoop(k, 0, 4)
        
        return I_lin, I_cross, I_1loop
        
    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_vvvv")        
    def Ivvvv_f23(self, k):
        I_lin   = self._Imn1Loop_vvvv.EvaluateLinear(k, 2, 3)
        I_cross = self._Imn1Loop_vvvv.EvaluateCross(k, 2, 3)
        I_1loop = self._Imn1Loop_vvvv.EvaluateOneLoop(k, 2, 3)
        
        return I_lin, I_cross, I_1loop

    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_vvvv")
    def Ivvvv_f32(self, k):
        I_lin   = self._Imn1Loop_vvvv.EvaluateLinear(k, 3, 2)
        I_cross = self._Imn1Loop_vvvv.EvaluateCross(k, 3, 2)
        I_1loop = self._Imn1Loop_vvvv.EvaluateOneLoop(k, 3, 2)
        
        return I_lin, I_cross, I_1loop
    
    @normalize_ImnOneLoop
    @interpolated_property("_Imn1Loop_vvvv")
    def Ivvvv_f33(self, k):
        I_lin   = self._Imn1Loop_vvvv.EvaluateLinear(k, 3, 3)
        I_cross = self._Imn1Loop_vvvv.EvaluateCross(k, 3, 3)
        I_1loop = self._Imn1Loop_vvvv.EvaluateOneLoop(k, 3, 3)
        
        return I_lin, I_cross, I_1loop
        
    #---------------------------------------------------------------------------
    # velocity-related quantities
    #---------------------------------------------------------------------------
    @cached_property('_P22bar_0')
    def _unnormed_velocity_kurtosis(self):
        """
        The unnormalized velocity kurtosis
        """
        return self._P22bar_0.VelocityKurtosis()
        
    @cached_property('_unnormed_velocity_kurtosis', '_power_norm')
    def velocity_kurtosis(self):
        """
        The velocity kurtosis <v_parallel^4>, computed using the 1-loop divergence
        auto spectra Pvv, aka P22bar
        """
        return self._power_norm**2 * self._unnormed_velocity_kurtosis
            
    @normalize_Jmn
    @interpolated_property("power_lin")
    def sigmasq_k(self, k):
        """
        The dark matter velocity dispersion at z, as a function of k, 
        ``\sigma^2_v(k)`` [units: `(Mpc/h)^2`]
        """
        # integrate up to 0.5 * kmax
        return self.power_lin.VelocityDispersion(k, 0.5)