from functools import wraps
from .. import pygcl, numpy as np
from ._cache import parameter, interpolated_function, cached_property
from .tools import RSDSpline as spline
from . import INTERP_KMIN, INTERP_KMAX

#-------------------------------------------------------------------------------
# decorators to properly normalize integrals
#-------------------------------------------------------------------------------
def normalize_Jmn(f):
    """
    Decorator to properly normalize Jmn integrals
    """
    @wraps(f)
    def wrapper(self, k):
        return self._power_norm * f(self, k)

    return wrapper

def normalize_Imn(f):
    """
    Decorator to properly normalize Imn integrals
    """
    @wraps(f)
    def wrapper(self, k):
        return self._power_norm**2 * f(self, k)
    return wrapper

normalize_Kmn = normalize_Imn

def normalize_ImnOneLoop(f):
    """
    Decorator to properly normalize one loop Imn integrals
    """
    @wraps(f)
    def wrapper(self, k):
        norm = self._power_norm
        terms = f(self, k)
        return norm**2*terms[0] + norm**3*terms[1] + norm**4*terms[2]
    return wrapper

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
    def __init__(self):

        # make sure power spectrum redshift is 0
        msg = "Integrals: input linear power spectrum must be defined at z = 0"
        assert self.power_lin.GetRedshift() == 0., msg

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
    @interpolated_function("_Jmn")
    def _unnormalized_J00(self, k):
        """J(m=0,n=0) perturbation theory integral"""
        return self._Jmn(k, 0, 0)
    J00 = normalize_Jmn(_unnormalized_J00)

    @interpolated_function("_Jmn")
    def _unnormalized_J01(self, k):
        """J(m=0,n=1) perturbation theory integral"""
        return self._Jmn(k, 0, 1)
    J01 = normalize_Jmn(_unnormalized_J01)

    @interpolated_function("_Jmn")
    def _unnormalized_J10(self, k):
        """J(m=1,n=0) perturbation theory integral"""
        return self._Jmn(k, 1, 0)
    J10 = normalize_Jmn(_unnormalized_J10)

    @interpolated_function("_Jmn")
    def _unnormalized_J11(self, k):
        """J(m=1,n=1) perturbation theory integral"""
        return self._Jmn(k, 1, 1)
    J11 = normalize_Jmn(_unnormalized_J11)

    @interpolated_function("_Jmn")
    def _unnormalized_J02(self, k):
        """J(m=0,n=2) perturbation theory integral"""
        return self._Jmn(k, 0, 2)
    J02 = normalize_Jmn(_unnormalized_J02)

    @interpolated_function("_Jmn")
    def _unnormalized_J20(self, k):
        """J(m=2,n=0) perturbation theory integral"""
        return self._Jmn(k, 2, 0)
    J20 = normalize_Jmn(_unnormalized_J20)

    #---------------------------------------------------------------------------
    # Imn integrals as a function of k
    #---------------------------------------------------------------------------
    @interpolated_function("_Imn")
    def _unnormalized_I00(self, k):
        """I(m=0,n=0) perturbation theory integral"""
        return self._Imn(k, 0, 0)
    I00 = normalize_Imn(_unnormalized_I00)

    @interpolated_function("_Imn")
    def _unnormalized_I01(self, k):
        """I(m=0,n=1) perturbation theory integral"""
        return self._Imn(k, 0, 1)
    I01 = normalize_Imn(_unnormalized_I01)

    @interpolated_function("_Imn")
    def _unnormalized_I02(self, k):
        """I(m=0,n=2) perturbation theory integral"""
        return self._Imn(k, 0, 2)
    I02 = normalize_Imn(_unnormalized_I02)

    @interpolated_function("_Imn")
    def _unnormalized_I03(self, k):
        """I(m=0,n=3) perturbation theory integral"""
        return self._Imn(k, 0, 3)
    I03 = normalize_Imn(_unnormalized_I03)

    @interpolated_function("_Imn")
    def _unnormalized_I10(self, k):
        """I(m=1,n=0) perturbation theory integral"""
        return self._Imn(k, 1, 0)
    I10 = normalize_Imn(_unnormalized_I10)

    @interpolated_function("_Imn")
    def _unnormalized_I11(self, k):
        """I(m=1,n=1) perturbation theory integral"""
        return self._Imn(k, 1, 1)
    I11 = normalize_Imn(_unnormalized_I11)

    @interpolated_function("_Imn")
    def _unnormalized_I12(self, k):
        """I(m=1,n=2) perturbation theory integral"""
        return self._Imn(k, 1, 2)
    I12 = normalize_Imn(_unnormalized_I12)

    @interpolated_function("_Imn")
    def _unnormalized_I13(self, k):
        """I(m=1,n=3) perturbation theory integral"""
        return self._Imn(k, 1, 3)
    I13 = normalize_Imn(_unnormalized_I13)

    @interpolated_function("_Imn")
    def _unnormalized_I20(self, k):
        """I(m=2,n=0) perturbation theory integral"""
        return self._Imn(k, 2, 0)
    I20 = normalize_Imn(_unnormalized_I20)

    @interpolated_function("_Imn")
    def _unnormalized_I21(self, k):
        """I(m=2,n=1) perturbation theory integral"""
        return self._Imn(k, 2, 1)
    I21 = normalize_Imn(_unnormalized_I21)

    @interpolated_function("_Imn")
    def _unnormalized_I22(self, k):
        """I(m=2,n=2) perturbation theory integral"""
        return self._Imn(k, 2, 2)
    I22 = normalize_Imn(_unnormalized_I22)

    @interpolated_function("_Imn")
    def _unnormalized_I23(self, k):
        """I(m=2,n=3) perturbation theory integral"""
        return self._Imn(k, 2, 3)
    I23 = normalize_Imn(_unnormalized_I23)

    @interpolated_function("_Imn")
    def _unnormalized_I30(self, k):
        """I(m=3,n=0) perturbation theory integral"""
        return self._Imn(k, 3, 0)
    I30 = normalize_Imn(_unnormalized_I30)

    @interpolated_function("_Imn")
    def _unnormalized_I31(self, k):
        """I(m=3,n=1) perturbation theory integral"""
        return self._Imn(k, 3, 1)
    I31 = normalize_Imn(_unnormalized_I31)

    @interpolated_function("_Imn")
    def _unnormalized_I32(self, k):
        """I(m=3,n=2) perturbation theory integral"""
        return self._Imn(k, 3, 2)
    I32 = normalize_Imn(_unnormalized_I32)

    @interpolated_function("_Imn")
    def _unnormalized_I33(self, k):
        """I(m=3,n=3) perturbation theory integral"""
        return self._Imn(k, 3, 3)
    I33 = normalize_Imn(_unnormalized_I33)

    #---------------------------------------------------------------------------
    # Kmn integrals
    #---------------------------------------------------------------------------
    @interpolated_function("_Kmn")
    def _unnormalized_K00(self, k):
        """K(m=0,n=0) perturbation theory integral"""
        return self._Kmn(k, 0, 0)
    K00 = normalize_Kmn(_unnormalized_K00)

    @interpolated_function("_Kmn")
    def _unnormalized_K00s(self, k):
        """K(m=0,n=0,s=True) perturbation theory integral"""
        return self._Kmn(k, 0, 0, True)
    K00s = normalize_Kmn(_unnormalized_K00s)

    @interpolated_function("_Kmn")
    def _unnormalized_K01(self, k):
        """K(m=0,n=1) perturbation theory integral"""
        return self._Kmn(k, 0, 1)
    K01 = normalize_Kmn(_unnormalized_K01)

    @interpolated_function("_Kmn")
    def _unnormalized_K01s(self, k):
        """K(m=0,n=1,s=True) perturbation theory integral"""
        return self._Kmn(k, 0, 1, True)
    K01s = normalize_Kmn(_unnormalized_K01s)

    @interpolated_function("_Kmn")
    def _unnormalized_K02s(self, k):
        """K(m=0,n=2,s=True) perturbation theory integral"""
        return self._Kmn(k, 0, 2, True)
    K02s = normalize_Kmn(_unnormalized_K02s)

    @interpolated_function("_Kmn")
    def _unnormalized_K10(self, k):
        """K(m=1,n=0) perturbation theory integral"""
        return self._Kmn(k, 1, 0)
    K10 = normalize_Kmn(_unnormalized_K10)


    @interpolated_function("_Kmn")
    def _unnormalized_K10s(self, k):
        """K(m=1,n=0,s=True) perturbation theory integral"""
        return self._Kmn(k, 1, 0, True)
    K10s = normalize_Kmn(_unnormalized_K10s)

    @interpolated_function("_Kmn")
    def _unnormalized_K11(self, k):
        """K(m=1,n=1) perturbation theory integral"""
        return self._Kmn(k, 1, 1)
    K11 = normalize_Kmn(_unnormalized_K11)

    @interpolated_function("_Kmn")
    def _unnormalized_K11s(self, k):
        """K(m=1,n=1,s=True) perturbation theory integral"""
        return self._Kmn(k, 1, 1, True)
    K11s = normalize_Kmn(_unnormalized_K11s)

    @interpolated_function("_Kmn")
    def _unnormalized_K20_a(self, k):
        """K(m=2,n=0) mu^2 perturbation theory integral"""
        return self._Kmn(k, 2, 0, False, 0)
    K20_a = normalize_Kmn(_unnormalized_K20_a)

    @interpolated_function("_Kmn")
    def _unnormalized_K20_b(self, k):
        """K(m=2,n=0) mu^4 perturbation theory integral"""
        return self._Kmn(k, 2, 0, False, 1)
    K20_b = normalize_Kmn(_unnormalized_K20_b)

    @interpolated_function("_Kmn")
    def _unnormalized_K20s_a(self, k):
        """K(m=2,n=0,s=True) mu^2 perturbation theory integral"""
        return self._Kmn(k, 2, 0, True, 0)
    K20s_a = normalize_Kmn(_unnormalized_K20s_a)

    @interpolated_function("_Kmn")
    def _unnormalized_K20s_b(self, k):
        """K(m=2,n=0,s=True) mu^4 perturbation theory integral"""
        return self._Kmn(k, 2, 0, True, 1)
    K20s_b = normalize_Kmn(_unnormalized_K20s_b)

    #---------------------------------------------------------------------------
    # full 2-loop integrals
    #---------------------------------------------------------------------------
    @interpolated_function("_Imn1Loop_vvdd")
    def _unnormalized_Ivvdd_h01(self, k):
        I_lin   = self._Imn1Loop_vvdd.EvaluateLinear(k, 0, 1)
        I_cross = self._Imn1Loop_vvdd.EvaluateCross(k, 0, 1)
        I_1loop = self._Imn1Loop_vvdd.EvaluateOneLoop(k, 0, 1)

        return I_lin, I_cross, I_1loop
    Ivvdd_h01 = normalize_ImnOneLoop(_unnormalized_Ivvdd_h01)

    @interpolated_function("_Imn1Loop_vvdd")
    def _unnormalized_Ivvdd_h02(self, k):
        I_lin   = self._Imn1Loop_vvdd.EvaluateLinear(k, 0, 2)
        I_cross = self._Imn1Loop_vvdd.EvaluateCross(k, 0, 2)
        I_1loop = self._Imn1Loop_vvdd.EvaluateOneLoop(k, 0, 2)

        return I_lin, I_cross, I_1loop
    Ivvdd_h02 = normalize_ImnOneLoop(_unnormalized_Ivvdd_h02)

    @interpolated_function("_Imn1Loop_dvdv")
    def _unnormalized_Idvdv_h03(self, k):
        I_lin   = self._Imn1Loop_dvdv.EvaluateLinear(k, 0, 3)
        I_cross = self._Imn1Loop_dvdv.EvaluateCross(k, 0, 3)
        I_1loop = self._Imn1Loop_dvdv.EvaluateOneLoop(k, 0, 3)

        return I_lin, I_cross, I_1loop
    Idvdv_h03 = normalize_ImnOneLoop(_unnormalized_Idvdv_h03)

    @interpolated_function("_Imn1Loop_dvdv")
    def _unnormalized_Idvdv_h04(self, k):
        I_lin   = self._Imn1Loop_dvdv.EvaluateLinear(k, 0, 4)
        I_cross = self._Imn1Loop_dvdv.EvaluateCross(k, 0, 4)
        I_1loop = self._Imn1Loop_dvdv.EvaluateOneLoop(k, 0, 4)

        return I_lin, I_cross, I_1loop
    Idvdv_h04 = normalize_ImnOneLoop(_unnormalized_Idvdv_h04)

    @interpolated_function("_Imn1Loop_vvvv")
    def _unnormalized_Ivvvv_f23(self, k):
        I_lin   = self._Imn1Loop_vvvv.EvaluateLinear(k, 2, 3)
        I_cross = self._Imn1Loop_vvvv.EvaluateCross(k, 2, 3)
        I_1loop = self._Imn1Loop_vvvv.EvaluateOneLoop(k, 2, 3)

        return I_lin, I_cross, I_1loop
    Ivvvv_f23 = normalize_ImnOneLoop(_unnormalized_Ivvvv_f23)

    @interpolated_function("_Imn1Loop_vvvv")
    def _unnormalized_Ivvvv_f32(self, k):
        I_lin   = self._Imn1Loop_vvvv.EvaluateLinear(k, 3, 2)
        I_cross = self._Imn1Loop_vvvv.EvaluateCross(k, 3, 2)
        I_1loop = self._Imn1Loop_vvvv.EvaluateOneLoop(k, 3, 2)

        return I_lin, I_cross, I_1loop
    Ivvvv_f32 = normalize_ImnOneLoop(_unnormalized_Ivvvv_f32)

    @interpolated_function("_Imn1Loop_vvvv")
    def _unnormalized_Ivvvv_f33(self, k):
        I_lin   = self._Imn1Loop_vvvv.EvaluateLinear(k, 3, 3)
        I_cross = self._Imn1Loop_vvvv.EvaluateCross(k, 3, 3)
        I_1loop = self._Imn1Loop_vvvv.EvaluateOneLoop(k, 3, 3)

        return I_lin, I_cross, I_1loop
    Ivvvv_f33 = normalize_ImnOneLoop(_unnormalized_Ivvvv_f33)

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

    @interpolated_function("power_lin")
    def _unnormalized_sigmasq_k(self, k):
        """
        The dark matter velocity dispersion at z, as a function of k,
        ``\sigma^2_v(k)`` [units: `(Mpc/h)^2`]
        """
        # integrate up to 0.5 * kmax
        return self.power_lin.VelocityDispersion(k, 0.5)
    sigmasq_k = normalize_Jmn(_unnormalized_sigmasq_k)
