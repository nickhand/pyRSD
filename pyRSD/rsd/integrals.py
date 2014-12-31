"""
 integrals.py
 pyRSD: Module built upon pygcl library to compute necessary perturbation 
 theory quantities 
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/30/2014
"""
from .. import pygcl, numpy as np
from .tools import RSDSpline as spline

K_SPLINE = np.logspace(-3, 0, 100)

#-------------------------------------------------------------------------------
class Integrals(object):
    """
    Class to compute and store the necessary PT integrals for the dark 
    matter redshift space power spectrum, in a redshift-independent manner.
    """
           
    def __init__(self, power_lin, z, sigma8):
        
        # store the input arguments
        self._power_lin = power_lin
        self._z         = z
        self._sigma8    = sigma8
        self._cosmo     = self._power_lin.GetCosmology()
        
        # make sure power spectrum redshift is 0
        msg = "Integrals: input linear power spectrum must be defined at z = 0"
        assert self._power_lin.GetRedshift() == 0., msg

    #end __init__
    
    #---------------------------------------------------------------------------
    # INPUT ATTRIBUTES 
    #---------------------------------------------------------------------------
    @property
    def power_lin(self):
        """
        Linear power spectrum object
        """
        return self._power_lin
        
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        """
        The cosmology of the input linear power spectrum
        """
        return self._cosmo
        
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        Redshift to compute the integrals at
        """
        return self._z
    
    @z.setter
    def z(self, val):
        del self.D
        self._z = val
            
    #---------------------------------------------------------------------------
    @property
    def sigma8(self):
        """
        Sigma_8 to compute the integrals at, which gives the normalization of 
        the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        self._sigma8 = val
    
    #---------------------------------------------------------------------------
    @property
    def power_norm(self):
        """
        The factor needed to normalize the integrals to the desired sigma_8, as
        specified by `self.sigma8`
        """
        return (self.sigma8 / self.cosmo.sigma8())**2

    #---------------------------------------------------------------------------
    # DERIVED QUANTITIES
    #---------------------------------------------------------------------------
    @property
    def D(self):
        try:
            return self._D
        except AttributeError:
            self._D = self.cosmo.D_z(self.z)
            return self._D

    @D.deleter
    def D(self):
        try:
            del self._D
        except AttributeError:
            pass
    
    #---------------------------------------------------------------------------
    # ONE LOOP POWER SPECTRA
    #---------------------------------------------------------------------------
    @property
    def Pdd(self):
        try:
            return self._Pdd
        except AttributeError:
            self._Pdd = pygcl.OneLoopPdd(self.power_lin)
            return self._Pdd
    
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        try:
            return self._Pdv
        except AttributeError:
            self._Pdv = pygcl.OneLoopPdv(self.power_lin)
            return self._Pdv
    
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        try:
            return self._Pvv
        except AttributeError:
            self._Pvv = pygcl.OneLoopPvv(self.power_lin)
            return self._Pvv
    
    #---------------------------------------------------------------------------
    @property
    def P22bar(self):
        try:
            return self._P22bar
        except AttributeError:
            self._P22bar = pygcl.OneLoopP22Bar(self.power_lin)
            return self._P22bar
            
    #---------------------------------------------------------------------------
    # INTEGRAL DRIVERS
    #---------------------------------------------------------------------------
    @property
    def _Imn(self):
        """
        The internal driver class to compute the I(m, n) integrals
        """
        try:
            return self.__Imn
        except AttributeError:
            self.__Imn = pygcl.Imn(self.power_lin)
            return self.__Imn
    
    #---------------------------------------------------------------------------
    @property
    def _Jmn(self):
        """
        The internal driver class to compute the J(m, n) integrals
        """
        try:
            return self.__Jmn
        except AttributeError:
            self.__Jmn = pygcl.Jmn(self.power_lin)
            return self.__Jmn

    #---------------------------------------------------------------------------
    @property
    def _Kmn(self):
        """
        The internal driver class to compute the J(m, n) integrals
        """
        try:
            return self.__Kmn
        except AttributeError:
            self.__Kmn = pygcl.Kmn(self.power_lin)
            return self.__Kmn
    
    #---------------------------------------------------------------------------
    @property
    def _Imn1Loop_dvdv(self):
        """
        The internal driver class to compute the 1-loop I(m, n) integrals, 
        which integrate over `P_dv(q) P_dv(|k-q|)`
        """
        try:
            return self.__Imn1Loop_dvdv
        except AttributeError:
            self.__Imn1Loop_dvdv = pygcl.ImnOneLoop(self.Pdv)
            return self.__Imn1Loop_dvdv
    
    #---------------------------------------------------------------------------
    @property
    def _Imn1Loop_vvdd(self):
        """
        The internal driver class to compute the 1-loop I(m, n) integrals, 
        which integrate over `P_vv(q) P_dd(|k-q|)`
        """
        try:
            return self.___Imn1Loop_vvdd
        except AttributeError:
            self.__Imn1Loop_vvdd = pygcl.ImnOneLoop(self.Pvv, self.Pdd)
            return self.__Imn1Loop_vvdd
    
    #---------------------------------------------------------------------------
    @property
    def _Imn1Loop_vvvv(self):
        """
        The internal driver class to compute the 1-loop I(m, n) integrals, 
        which integrate over `P_vv(q) P_vv(|k-q|)`
        """
        try:
            return self.___Imn1Loop_vvvv
        except AttributeError:
            self.__Imn1Loop_vvvv = pygcl.ImnOneLoop(self.Pvv)
            return self.__Imn1Loop_vvvv
    
    #---------------------------------------------------------------------------
    # Jmn integrals as a function of input k
    #---------------------------------------------------------------------------
    def _getattr_Jmn(self, k, att_name, m, n):
        """
        Internal method to return Jmn as a function of k
        """
        spline_name = att_name + "_spline"
        try:
            f = getattr(self, spline_name)
        except AttributeError:
            f = spline(K_SPLINE, self._Jmn(K_SPLINE, m, n), bounds_error=False, fill_value=0)
            setattr(self, spline_name, f)
    
        return (self.power_norm*self.D**2) * f(k)       
    #end _getattr_Jmn
    
    #---------------------------------------------------------------------------
    def J00(self, k):
        return self._getattr_Jmn(k, '_J00', 0, 0)

    def J01(self, k):
         return self._getattr_Jmn(k, '_J01', 0, 1)

    def J10(self, k):
        return self._getattr_Jmn(k, '_J10', 1, 0)

    def J11(self, k):
        return self._getattr_Jmn(k, '_J11', 1, 1)
            
    def J02(self, k):
        return self._getattr_Jmn(k, '_J02', 0, 2)
            
    def J20(self, k):
        return self._getattr_Jmn(k, '_J20', 2, 0)
            
    #---------------------------------------------------------------------------
    # Imn integrals as a function of k
    #---------------------------------------------------------------------------
    def _getattr_Imn(self, k, att_name, m, n):
        """
        Internal method to return Imn as a function of k
        """
        spline_name = att_name + "_spline"
        try:
            f = getattr(self, spline_name)
        except AttributeError:
            f = spline(K_SPLINE, self._Imn(K_SPLINE, m, n), bounds_error=False, fill_value=0)
            setattr(self, spline_name, f)

        return (self.power_norm*self.D**2)**2 * f(k)
    #end _getattr_Imn
    
    #---------------------------------------------------------------------------    
    def I00(self, k):
        return self._getattr_Imn(k, '_I00', 0, 0)
    
    def I01(self, k):
        return self._getattr_Imn(k, '_I01', 0, 1)

    def I02(self, k):
        return self._getattr_Imn(k, '_I02', 0, 2)
    
    def I03(self, k):
        return self._getattr_Imn(k, '_I03', 0, 3)
    
    def I10(self, k):
        return self._getattr_Imn(k, '_I10', 1, 0)
    
    def I11(self, k):
        return self._getattr_Imn(k, '_I11', 1, 1)
    
    def I12(self, k):
        return self._getattr_Imn(k, '_I12', 1, 2)

    def I13(self, k):
        return self._getattr_Imn(k, '_I13', 1, 3)
    
    def I20(self, k):
        return self._getattr_Imn(k, '_I20', 2, 0)
    
    def I21(self, k):
        return self._getattr_Imn(k, '_I21', 2, 1)
    
    def I22(self, k):
        return self._getattr_Imn(k, '_I22', 2, 2)
    
    def I23(self, k):
        return self._getattr_Imn(k, '_I23', 2, 3)
    
    def I30(self, k):
        return self._getattr_Imn(k, '_I30', 3, 0)
    
    def I31(self, k):
        return self._getattr_Imn(k, '_I31', 3, 1) 
    
    def I32(self, k):
        return self._getattr_Imn(k, '_I32', 3, 2)

    def I33(self, k):
        return self._getattr_Imn(k, '_I33', 3, 3)

    #---------------------------------------------------------------------------
    # Kmn integrals
    #---------------------------------------------------------------------------
    def _getattr_Kmn(self, k, att_name, m, n, tidal=False, part=0):
        """
        Internal method to return Kmn as a function of k
        """
        spline_name = att_name + "_spline"
        try:
            f = getattr(self, spline_name)
        except AttributeError:
            f = spline(K_SPLINE, self._Kmn(K_SPLINE, m, n, tidal, part), bounds_error=False, fill_value=0)
            setattr(self, spline_name, f)
            
        return (self.power_norm*self.D**2)**2 * f(k)
    #end _getattr_Kmn
    
    #---------------------------------------------------------------------------
    def K00(self, k):
        return self._getattr_Kmn(k, '_K00', 0, 0)
    
    def K00s(self, k):
        return self._getattr_Kmn(k, '_K00s', 0, 0, tidal=True)
              
    def K01(self, k):
        return self._getattr_Kmn(k, '_K01', 0, 1)

    def K01s(self, k):
        return self._getattr_Kmn(k, '_K01s', 0, 1, tidal=True)
            
    def K02s(self, k):
        return self._getattr_Kmn(k, '_K02s', 0, 2, tidal=True)
                
    def K10(self, k):
        return self._getattr_Kmn(k, '_K10', 1, 0)

    def K10s(self, k):
        return self._getattr_Kmn(k, '_K10s', 1, 0, tidal=True)
        
    def K11(self, k):
        return self._getattr_Kmn(k, '_K11', 1, 1)
    
    def K11s(self, k):
        return self._getattr_Kmn(k, '_K11s', 1, 1, tidal=True)

    def K20_a(self, k):
        return self._getattr_Kmn(k, '_K20_a', 2, 0, tidal=False, part=0)
                
    def K20_b(self, k):
        return self._getattr_Kmn(k, '_K20_b', 2, 0, tidal=False, part=1)

    def K20s_a(self, k):
        return self._getattr_Kmn(k, '_K20s_a', 2, 0, tidal=True, part=0)
        
    def K20s_b(self, k):
        return self._getattr_Kmn(k, '_K20s_b', 2, 0, tidal=True, part=1)
            
    #---------------------------------------------------------------------------
    # 2-LOOP INTEGRALS
    #---------------------------------------------------------------------------
    def _getattr_2loop(self, k, att_name, driver_name, m, n):
        """
        Internal function to get 2-loop integral attributes
        """
        driver = getattr(self, driver_name)
        if not hasattr(self, att_name):
            I_lin   = spline(K_SPLINE, getattr(driver, 'EvaluateLinear')(K_SPLINE, m, n), bounds_error=False, fill_value=0)
            I_cross = spline(K_SPLINE, getattr(driver, 'EvaluateCross')(K_SPLINE, m, n), bounds_error=False, fill_value=0)
            I_1loop = spline(K_SPLINE, getattr(driver, 'EvaluateOneLoop')(K_SPLINE, m, n), bounds_error=False, fill_value=0)

            setattr(self, att_name, [I_lin, I_cross, I_1loop])

        splines = getattr(self, att_name)
        norm = (self.power_norm*self.D**2)
        return norm**2*splines[0](k) + norm**3*splines[1](k) + norm**4*splines[2](k)
            
    #---------------------------------------------------------------------------
    def Ivvdd_h01(self, k):        
        return self._getattr_2loop(k, "_Ivvdd_h01_spline", "_Imn1Loop_vvdd", 0, 1)

    def Ivvdd_h02(self, k):
        return self._getattr_2loop(k, "_Ivvdd_h02_spline", "_Imn1Loop_vvdd", 0, 2)

    def Idvdv_h03(self, k):
        return self._getattr_2loop(k, "_Idvdv_h03_spline", "_Imn1Loop_dvdv", 0, 3)

    def Idvdv_h04(self, k):
        return self._getattr_2loop(k, "_Idvdv_h04_spline", "_Imn1Loop_dvdv", 0, 4)
        
    def Ivvvv_f23(self, k):
        return self._getattr_2loop(k, "_Ivvvv_f23_spline", "_Imn1Loop_vvvv", 2, 3)

    def Ivvvv_f32(self, k):
        return self._getattr_2loop(k, "_Ivvvv_f32_spline", "_Imn1Loop_vvvv", 3, 2)
    
    def Ivvvv_f33(self, k):
        return self._getattr_2loop(k, "_Ivvvv_f33_spline", "_Imn1Loop_vvvv", 3, 3)
        
    #---------------------------------------------------------------------------
    @property
    def velocity_kurtosis(self):
        """
        The velocity kurtosis <v_parallel^4>, computed using the 1-loop divergence
        auto spectra Pvv, aka P22bar
        """
        try:
            return (self.power_norm*self.D**2)**2 * self._velocity_kurtosis
        except AttributeError:
            self._velocity_kurtosis = self.P22bar.VelocityKurtosis()
            return (self.power_norm*self.D**2)**2 * self._velocity_kurtosis
            
    #---------------------------------------------------------------------------
    def sigmasq_k(self, k):
        """
        The dark matter velocity dispersion at z, as a function of k, 
        ``\sigma^2_v(k)`` [units: `(Mpc/h)^2`]
        """
        try:
            return self.power_norm*self.D**2 * self._sigmasq_k(k)
        except AttributeError:
            # integrate up to 0.5*k
            self._sigmasq_k = spline(K_SPLINE, self.power_lin.VelocityDispersion(K_SPLINE, 0.5), bounds_error=False, fill_value=0)
            return self.power_norm*self.D**2 * self._sigmasq_k(k)
    #---------------------------------------------------------------------------
#endclass Integrals

#-------------------------------------------------------------------------------
