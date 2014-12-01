"""
 integrals.py
 pyRSD: Module built upon pygcl library to compute necessary perturbation 
 theory quantities 
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/30/2014
"""
from .. import pygcl
import numpy as np

class Integrals(object):
    """
    Class to compute and store the necessary PT integrals for the dark 
    matter redshift space power spectrum, in a redshift-independent manner.
    """
           
    def __init__(self, k_eval, z, power_lin, sigma8):
        
        # store the input arguments
        self._k_eval    = k_eval
        self._z         = z
        self._power_lin = power_lin
        
        # make sure power spectrum redshift is 0
        assert self._power_lin.GetRedshift() == 0., "Integrals: input linear power spectrum must be defined at z = 0"
        
        # set the initial sigma8
        self._sigma8 = sigma8
        
    #end __init__
    
    #---------------------------------------------------------------------------
    # INPUT ATTRIBUTES 
    #---------------------------------------------------------------------------
    @property
    def k_eval(self):
        """
        Wavenumbers to evaluate the integrals at [units: `h/Mpc`]
        """
        return self._k_eval
    
    @property
    def power_lin(self):
        """
        Linear power spectrum object
        """
        return self._power_lin
        
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
            
    @property
    def sigma8(self):
        """
        Sigma_8 to compute the integrals at, which gives the normalization of the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        self._sigma8 = val
    
    @property
    def power_norm(self):
        """
        The factor needed to normalize the integrals to the desired sigma_8, as
        specified by `self.sigma8`
        """
        return (self.sigma8 / self._power_lin.GetCosmology().sigma8())**2

    #---------------------------------------------------------------------------
    # DERIVED QUANTITIES
    #---------------------------------------------------------------------------
    @property
    def D(self):
        try:
            return self._D
        except AttributeError:
            self._D = self.power_lin.GetCosmology().D_z(self.z)
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
    # Jmn integrals
    #---------------------------------------------------------------------------
    @property
    def J00(self):
        try:
            return self.power_norm*self.D**2 * self._J00
        except AttributeError:
            self._J00 = self._Jmn(self.k_eval, 0, 0)
            return self.power_norm*self.D**2 * self._J00

    @property
    def J01(self):
        try:
            return self.power_norm*self.D**2 * self._J01
        except AttributeError:
            self._J01 = self._Jmn(self.k_eval, 0, 1)
            return self.power_norm*self.D**2 * self._J01            

    @property
    def J10(self):
        try:
            return self.power_norm*self.D**2 * self._J10
        except AttributeError:
            self._J10 = self._Jmn(self.k_eval, 1, 0)
            return self.power_norm*self.D**2 * self._J10

    @property
    def J11(self):
        try:
            return self.power_norm*self.D**2 * self._J11
        except AttributeError:
            self._J11 = self._Jmn(self.k_eval, 1, 1)
            return self.power_norm*self.D**2 * self._J11 
            
    @property
    def J02(self):
        try:
            return self.power_norm*self.D**2 * self._J02
        except AttributeError:
            self._J02 = self._Jmn(self.k_eval, 0, 2)
            return self.power_norm*self.D**2 * self._J02
            
    @property
    def J20(self):
        try:
            return self.power_norm*self.D**2 * self._J20
        except AttributeError:
            self._J20 = self._Jmn(self.k_eval, 2, 0)
            return self.power_norm*self.D**2 * self._J20
    #---------------------------------------------------------------------------
    # Imn integrals
    #---------------------------------------------------------------------------
    @property
    def I00(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I00
        except AttributeError:
            self._I00 = self._Imn(self.k_eval, 0, 0)
            return (self.power_norm*self.D**2)**2 * self._I00

    @property
    def I01(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I01
        except AttributeError:
            self._I01 = self._Imn(self.k_eval, 0, 1)
            return (self.power_norm*self.D**2)**2 * self._I01      

    @property
    def I02(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I02
        except AttributeError:
            self._I02 = self._Imn(self.k_eval, 0, 2)
            return (self.power_norm*self.D**2)**2 * self._I02   

    @property
    def I03(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I03
        except AttributeError:
            self._I03 = self._Imn(self.k_eval, 0, 3)
            return (self.power_norm*self.D**2)**2 * self._I03

    @property
    def I10(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I10
        except AttributeError:
            self._I10 = self._Imn(self.k_eval, 1, 0)
            return (self.power_norm*self.D**2)**2 * self._I10

    @property
    def I11(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I11
        except AttributeError:
            self._I11 = self._Imn(self.k_eval, 1, 1)
            return (self.power_norm*self.D**2)**2 * self._I11      

    @property
    def I12(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I12
        except AttributeError:
            self._I12 = self._Imn(self.k_eval, 1, 2)
            return (self.power_norm*self.D**2)**2 * self._I02   

    @property
    def I13(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I13
        except AttributeError:
            self._I13 = self._Imn(self.k_eval, 1, 3)
            return (self.power_norm*self.D**2)**2 * self._I13

    @property
    def I20(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I20
        except AttributeError:
            self._I20 = self._Imn(self.k_eval, 2, 0)
            return (self.power_norm*self.D**2)**2 * self._I20

    @property
    def I21(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I21
        except AttributeError:
            self._I21 = self._Imn(self.k_eval, 2, 1)
            return (self.power_norm*self.D**2)**2 * self._I21      

    @property
    def I22(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I22
        except AttributeError:
            self._I22 = self._Imn(self.k_eval, 2, 2)
            return (self.power_norm*self.D**2)**2 * self._I22   

    @property
    def I23(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I23
        except AttributeError:
            self._I23 = self._Imn(self.k_eval, 2, 3)
            return (self.power_norm*self.D**2)**2 * self._I23

    @property
    def I30(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I30
        except AttributeError:
            self._I30 = self._Imn(self.k_eval, 3, 0)
            return (self.power_norm*self.D**2)**2 * self._I30

    @property
    def I31(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I31
        except AttributeError:
            self._I31 = self._Imn(self.k_eval, 3, 1)
            return (self.power_norm*self.D**2)**2 * self._I31      

    @property
    def I32(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I32
        except AttributeError:
            self._I32 = self._Imn(self.k_eval, 3, 2)
            return (self.power_norm*self.D**2)**2 * self._I32   

    @property
    def I33(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._I33
        except AttributeError:
            self._I33 = self._Imn(self.k_eval, 3, 3)
            return (self.power_norm*self.D**2)**2 * self._I33                    

    #---------------------------------------------------------------------------
    # Kmn integrals
    #---------------------------------------------------------------------------
    @property
    def K00(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K00
        except AttributeError:
            self._K00 = self._Kmn(self.k_eval, 0, 0, False)
            return (self.power_norm*self.D**2)**2 * self._K00

    @property
    def K00s(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K00s
        except AttributeError:
            self._K00s = self._Kmn(self.k_eval, 0, 0, True)
            return (self.power_norm*self.D**2)**2 * self._K00s            

    @property
    def K01(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K01
        except AttributeError:
            self._K01 = self._Kmn(self.k_eval, 0, 1, False)
            return (self.power_norm*self.D**2)**2 * self._K01

    @property
    def K01s(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K01s
        except AttributeError:
            self._K01s = self._Kmn(self.k_eval, 0, 1, True)
            return (self.power_norm*self.D**2)**2 * self._K01s
            
    @property
    def K02s(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K02s
        except AttributeError:
            self._K02s = self._Kmn(self.k_eval, 0, 2, True)
            return (self.power_norm*self.D**2)**2 * self._K02s
            
    @property
    def K10(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K10
        except AttributeError:
            self._K10 = self._Kmn(self.k_eval, 1, 0, False)
            return (self.power_norm*self.D**2)**2 * self._K10

    @property
    def K10s(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K10s
        except AttributeError:
            self._K10s = self._Kmn(self.k_eval, 1, 0, True)
            return (self.power_norm*self.D**2)**2 * self._K10s

    @property
    def K11(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K11
        except AttributeError:
            self._K11 = self._Kmn(self.k_eval, 1, 1, False)
            return (self.power_norm*self.D**2)**2 * self._K11

    @property
    def K11s(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K11s
        except AttributeError:
            self._K11s = self._Kmn(self.k_eval, 1, 1, True)
            return (self.power_norm*self.D**2)**2 * self._K11s

    @property
    def K20_a(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K20_a
        except AttributeError:
            self._K20_a = self._Kmn(self.k_eval, 2, 0, False, 0)
            return (self.power_norm*self.D**2)**2 * self._K20_a
            
    @property
    def K20_b(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K20_b
        except AttributeError:
            self._K20_b = self._Kmn(self.k_eval, 2, 0, False, 1)
            return (self.power_norm*self.D**2)**2 * self._K20_b

    @property
    def K20s_a(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K20s_a
        except AttributeError:
            self._K20s_a = self._Kmn(self.k_eval, 2, 0, True, 0)
            return (self.power_norm*self.D**2)**2 * self._K20s_a

    @property
    def K20s_b(self):
        try:
            return (self.power_norm*self.D**2)**2 * self._K20s_b
        except AttributeError:
            self._K20s_b = self._Kmn(self.k_eval, 2, 0, True, 1)
            return (self.power_norm*self.D**2)**2 * self._K20s_b
            
    #---------------------------------------------------------------------------
    # 2-LOOP INTEGRALS
    #---------------------------------------------------------------------------
    def _getattr_2loop(self, att_name, driver_name, m, n):
        """
        Internal function to get 2-loop integral attributes
        """
        driver = getattr(self, driver_name)
        if not hasattr(self, att_name):
            I_lin   = getattr(driver, 'EvaluateLinear')(self.k_eval, m, n)
            I_cross = getattr(driver, 'EvaluateCross')(self.k_eval, m, n)
            I_1loop = getattr(driver, 'EvaluateOneLoop')(self.k_eval, m, n)

            setattr(self, att_name, np.vstack((I_lin, I_cross, I_1loop)).T)

        data = getattr(self, att_name)
        norm = (self.power_norm*self.D**2)
        return norm**2*data[:,0] + norm**3*data[:,1] + norm**4*data[:,2]
            
    @property
    def Ivvdd_h01(self):        
        return self._getattr_2loop("_Ivvdd_h01", "_Imn1Loop_vvdd", 0, 1)

    @property
    def Ivvdd_h02(self):
        return self._getattr_2loop("_Ivvdd_h02", "_Imn1Loop_vvdd", 0, 2)

    @property
    def Idvdv_h03(self):
        return self._getattr_2loop("_Idvdv_h03", "_Imn1Loop_dvdv", 0, 3)
        
    @property
    def Idvdv_h04(self):
        return self._getattr_2loop("_Idvdv_h04", "_Imn1Loop_dvdv", 0, 4)
        
    @property
    def Ivvvv_f23(self):
        return self._getattr_2loop("_Ivvdv_f23", "_Imn1Loop_vvvv", 2, 3)
    
    @property
    def Ivvvv_f32(self):
        return self._getattr_2loop("_Ivvdv_f32", "_Imn1Loop_vvvv", 3, 2)
    
    @property
    def Ivvvv_f33(self):
        return self._getattr_2loop("_Ivvdv_f33", "_Imn1Loop_vvvv", 3, 3)
        
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