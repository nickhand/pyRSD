"""
 dm_power_moments.py
 pyRSD: functions for computing dark matter power moments: P00, P01, P11
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/12/2014
"""
from .. import pygcl, numpy as np, os
from ..data import P11_mu4_z_0_000, P11_mu4_z_0_509, P11_mu4_z_0_989
from ..data import Pdv_mu0_z_0_000, Pdv_mu0_z_0_509, Pdv_mu0_z_0_989
from .tools import RSDSpline as spline
from . import INTERP_KMIN, INTERP_KMAX

import collections
import bisect

K_SPLINE = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 100)
SIGMA8_MIN = 0.1
SIGMA8_MAX = 2.5
INTERP_PTS = 50

#-------------------------------------------------------------------------------
class DarkMatterPowerMoment(object):
    """
    A class to compute a generic dark matter power moment
    """
    def __init__(self, power_lin, z, sigma8, model_type='A', interpolate_zeldovich=True):
      
        # store the input arguments
        self.interpolate_zeldovich = interpolate_zeldovich
        self._power_lin = power_lin
        self._cosmo     = self._power_lin.GetCosmology()
        self.model_type = model_type
        
        # make sure power spectrum redshift is 0
        msg = "input linear power spectrum must be defined at z = 0"
        assert self._power_lin.GetRedshift() == 0., msg
        
        # set the initial redshift, sigma8 
        self.z = z
        self.sigma8 = sigma8
                
        # set up the zeldovich power interpolation table
        if self.interpolate_zeldovich:
            self._compute_zeldovich_power_table()
    
    #---------------------------------------------------------------------------
    # HZPT model parameters (model A)
    #---------------------------------------------------------------------------
    @property
    def R1_hzpt(self):
        """
        Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)
        
        Note: the units are length [Mpc/h]
        """
        return 3.33 * (self.sigma8_z/0.8)**0.88

    #---------------------------------------------------------------------------
    @property
    def R1h_hzpt(self):
        """
        Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 3.87 * (self.sigma8_z/0.8)**0.29
        
    #---------------------------------------------------------------------------
    @property
    def R2h_hzpt(self):
        """
        Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 1.69 * (self.sigma8_z/0.8)**0.43
           
    #---------------------------------------------------------------------------
    @property
    def R_hzpt(self):
        """
        Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 26. * (self.sigma8_z/0.8)**0.15
       
    #---------------------------------------------------------------------------
    @property
    def A0_hzpt(self):
        """
        Returns the A0 radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are power [(h/Mpc)^3]
        """
        return 750. * (self.sigma8_z/0.8)**3.75
    
    #---------------------------------------------------------------------------
    # Irshad's model parameters
    #---------------------------------------------------------------------------
    @property
    def A0_irshad(self):
        """
        Returns the A0 parameter for the model presented in Eq. 27 
        of arXiv:1407.0060
        """
        return 1529.87 * self.sigma8_z**3.9
        
    #---------------------------------------------------------------------------
    @property
    def A2_irshad(self):
        """
        Returns the A2 parameter for the model presented in Eq. 28
        of arXiv:1407.0060
        """
        return 1299.75 * self.sigma8_z**3.0
        
    #---------------------------------------------------------------------------
    @property
    def A4_irshad(self):
        """
        Returns the A4 parameter for the model presented in Eq. 29 
        of arXiv:1407.0060
        """
        return 758.31 * self.sigma8_z**2.2
        
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
        self._z = val
        
        # compute new redshift power table
        self.zeldovich_base.SetRedshift(val)
        
        if self.interpolate_zeldovich:
            self._compute_zeldovich_power_table()
    
    #---------------------------------------------------------------------------        
    @property
    def sigma8(self):
        """
        Sigma_8 at `z=0` to compute the spectrum at, which gives the 
        normalization of the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        self._sigma8 = val
                
    #---------------------------------------------------------------------------
    @property
    def model_type(self):
        """
        The type of model to use for the broadband correction; either [`A`, `B`]
        """
        return self._model_type

    @model_type.setter
    def model_type(self, val):

        assert val in ['A', 'B'], "Model type must be either 'A' or 'B'"
        self._model_type = val
            
    #---------------------------------------------------------------------------
    @property
    def sigma8_z(self):
        """
        Sigma_8 at `self.z`
        """
        return self.sigma8 * (self.cosmo.Sigma8_z(self.z) / self.cosmo.sigma8())
            
    #---------------------------------------------------------------------------
    @property
    def zeldovich_base(self):
        """
        The `pygcl.ZeldovichPS` class serving as a base class such that we 
        only compute the `X(q)` and `Y(q)` once. 
        """ 
        try:
            return self._zeldovich_base
        except AttributeError:
            self._zeldovich_base = pygcl.ZeldovichPS(self.power_lin)
            return self._zeldovich_base
            
    #---------------------------------------------------------------------------
    # The power attributes
    #---------------------------------------------------------------------------
    def zeldovich_power(self, k):
        """
        Return the power from the Zel'dovich term
        """
        if self.interpolate_zeldovich:
            # return NaNs if we are out of bounds
            if self.sigma8 < SIGMA8_MIN or self.sigma8 > SIGMA8_MAX: return np.nan*k
        
            keys = self.zeldovich_power_table.keys()
            ihi = bisect.bisect(keys, self.sigma8)
            ilo = ihi - 1
        
            s8_lo = keys[ilo]
            s8_hi = keys[ihi]
            w = (self.sigma8 - s8_lo) / (s8_hi - s8_lo) 
            return (1 - w)*self.zeldovich_power_table[s8_lo](k) + w*self.zeldovich_power_table[s8_hi](k)
        else:
            raise NotImplementedError("whoops")
        
    #---------------------------------------------------------------------------  
    def F(self, k):
        """
        The compensation function F(k) that causes the broadband power to go
        to zero at low k, in order to conserver mass/momentum
        
        Notes
        -----
        For `model A`, the functional form is given by 1 - 1 / (1 + k^2 R^2), 
        where R(z) is given by Eq. 4 in arXiv:1501.07512.
        
        For `model B`, the functional form is a 10th-order polynomial with 
        coefficients given by Table 1 in arXiv:1407.0060
        
        """
        if self.model_type == 'A':
            return 1. - 1./(1. + (k*self.R_hzpt)**2)
            
        elif self.model_type == 'B':
            ans = [0., 21.814, -174.134, 747.369, -2006.792, 3588.808, -4316.241, 
                    3415.525, -1692.839, 474.377, -57.228]
            
            toret = 0.
            for i, an in enumerate(ans):
                toret += an*k**i
            return toret
    #---------------------------------------------------------------------------
#endclass DarkMatterPowerMoment

#-------------------------------------------------------------------------------
class DarkMatterP00(DarkMatterPowerMoment):
    """
    Dark matter model for P00
    """
    
    def __init__(self, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterP00, self).__init__(*args, **kwargs)
 
    #---------------------------------------------------------------------------
    def _compute_zeldovich_power_table(self):
        """
        Compute the interpolation table for the Zeldovich power as a function
        of sigma8
        """
        P00 = pygcl.ZeldovichP00(self.zeldovich_base)
        
        self.zeldovich_power_table = collections.OrderedDict()
        sigma8s = np.linspace(SIGMA8_MIN, SIGMA8_MAX, INTERP_PTS)
        for sigma8 in sigma8s:
            P00.SetSigma8(sigma8)
            self.zeldovich_power_table[sigma8] = spline(K_SPLINE, P00(K_SPLINE))
    
    #---------------------------------------------------------------------------
    def zeldovich_power(self, k):
        """
        Return the power from the Zel'dovich term
        """
        if self.interpolate_zeldovich:
            return DarkMatterPowerMoment.zeldovich_power(self, k)
        else:
            P00 = pygcl.ZeldovichP00(self.zeldovich_base)
            P00.SetSigma8(self.sigma8)
            return P00(k)
    
    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3
        
        Notes
        -----
        For `model A`, the functional form is given by: 
        :math:  P_BB = A0 * F(k) * [ (1 + (k*R1)^2) / (1 + (k*R1h)^2 + (k*R2h)^4) ], 
        as given by Eq. 1 in arXiv:1501.07512.
        
        For `model B`, the functional form is given by:
        :math: (A0 - A2*k^2 + A4^k^4) * F(k),
        as given by Eq 32 in arXiv:1407.0060
        """ 
        if self.model_type == 'A':
            A0  = self.A0_hzpt
            R1  = self.R1_hzpt
            R1h = self.R1h_hzpt
            R2h = self.R2h_hzpt
            return A0*self.F(k)*(1. + (k*R1)**2) / (1 + (k*R1h)**2 + (k*R2h)**4)
            
        else:
            A0 = self.A0_irshad
            A2 = self.A2_irshad
            A4 = self.A4_irshad
            return self.F(k) * (A0 - A2*k**2 + A4*k**4)
        
    #---------------------------------------------------------------------------
    def power(self, k):
        """
        Return the total power in units of (Mpc/h)^3
        """
        return self.zeldovich_power(k) + self.broadband_power(k)
    #---------------------------------------------------------------------------
#endclass DarkMatterP00

#-------------------------------------------------------------------------------

class DarkMatterP01(DarkMatterPowerMoment):
    """
    Dark matter model for P01
    """
    
    def __init__(self, f, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterP01, self).__init__(*args, **kwargs)
        self.f = f
          
    #---------------------------------------------------------------------------      
    @property
    def f(self):
        """
        The growth rate, defined as the `dlnD/dlna`. 
     
        If the parameter has not been explicity set, it defaults to the value
        at `self.z`
        """
        try:
            return self._f
        except AttributeError:
            return self.cosmo.f_z(self.z)
    
    @f.setter
    def f(self, val):
        self._f = val

    #---------------------------------------------------------------------------
    def _compute_zeldovich_power_table(self):
        """
        Compute the interpolation table for the Zeldovich power as a function
        of sigma8
        """
        P01 = pygcl.ZeldovichP01(self.zeldovich_base)
        
        self.zeldovich_power_table = collections.OrderedDict()
        sigma8s = np.linspace(SIGMA8_MIN, SIGMA8_MAX, INTERP_PTS)
        for sigma8 in sigma8s:
            P01.SetSigma8(sigma8)
            self.zeldovich_power_table[sigma8] = spline(K_SPLINE, P01(K_SPLINE))
    
    #---------------------------------------------------------------------------
    def zeldovich_power(self, k):
        """
        Return the power from the Zel'dovich term
        """
        if self.interpolate_zeldovich:
            return DarkMatterPowerMoment.zeldovich_power(self, k)
        else:
            P01 = pygcl.ZeldovichP01(self.zeldovich_base)
            P01.SetSigma8(self.sigma8)
            return P01(k)
            
    #---------------------------------------------------------------------------
    # derivs of HZPT model parameters (model A)
    #---------------------------------------------------------------------------
    @property
    def dR1_hzpt_dlna(self):
        """
        Returns the derivative of `self.R1_hzpt` with respect to the `lna`
        """
        return self.f * 0.88 * self.R1_hzpt
      
    #---------------------------------------------------------------------------
    @property
    def dR1h_hzpt_dlna(self):
        """
        Returns the derivative of `self.R1h_hzpt` with respect to the `lna`
        """
        return self.f * 0.29 * self.R1h_hzpt

    #---------------------------------------------------------------------------
    @property
    def dR2h_hzpt_dlna(self):
        """
        Returns the derivative of `self.R2h_hzpt` with respect to the `lna`
        """
        return self.f * 0.43 * self.R2h_hzpt

    #---------------------------------------------------------------------------
    @property
    def dR_hzpt_dlna(self):
        """
        Returns the derivative of `self.R_hzpt` with respect to the `lna`
        """
        return self.f * 0.15 * self.R_hzpt
    
    #---------------------------------------------------------------------------
    @property
    def dA0_hzpt_dlna(self):
        """
        Returns the derivative of `self.A0_hzpt` with respect to the `lna`
        """
        return self.f * 3.75 * self.A0_hzpt
        
    #---------------------------------------------------------------------------
    # Irshad's model parameters
    #---------------------------------------------------------------------------
    @property
    def dA0_irshad_dlna(self):
        """
        Returns the derivative of `self.A0_irshad` with respect to the `lna`
        """
        return self.f * 3.9 * self.A0_irshad

    #---------------------------------------------------------------------------
    @property
    def dA2_irshad_dlna(self):
        """
        Returns the derivative of `self.A2_irshad` with respect to the `lna`
        """
        return self.f * 3.0 * self.A2_irshad
    
    #---------------------------------------------------------------------------
    @property
    def dA4_irshad_dlna(self):
        """
        Returns the derivative of `self.A4_irshad` with respect to the `lna`
        """
        return self.f * 2.2 * self.A2_irshad
    
    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3
        """ 
        # define the redshift dependent functions, Fk
        if self.model_type == 'A':

            # the P00_BB parameters
            A0  = self.A0_hzpt
            R1  = self.R1_hzpt
            R1h = self.R1h_hzpt
            R2h = self.R2h_hzpt
            R   = self.R_hzpt
            
            # derivs wrt to lna
            dA0  = self.dA0_hzpt_dlna
            dR1  = self.dR1_hzpt_dlna
            dR1h = self.dR1h_hzpt_dlna
            dR2h = self.dR2h_hzpt_dlna
            dR   = self.dR_hzpt_dlna
            
            # store these for convenience
            norm = (1 + (k*R1h)**2 + (k*R2h)**4)
            Ck = (1. + (k*R1)**2) / norm
            Fk = self.F(k)
            
            # first term of tot deriv
            term1 = dA0 * (Fk*Ck)
            
            # 2nd term
            term2 = (A0*Ck) * (2*k**2*R*dR) / (1 + (k*R)**2)**2
            
            # 3rd term
            term3_a = (2*k**2*R1*dR1) / norm
            term3_b = -(1 + (k*R1)**2) / norm**2 * (2*k**2*R1h*dR1h + 4*k**4*R2h**3*dR2h)
            term3 = (A0*Fk) * (term3_a + term3_b)
            
            return term1 + term2 + term3
                
        elif self.model_type == 'B':
            dA0 = self.dA0_irshad_dlna
            dA2 = self.dA2_irshad_dlna
            dA4 = self.dA4_irshad_dlna
            return self.F(k) * (dA0 - dA2*k**2 + dA4*k**4)
        
    #---------------------------------------------------------------------------
    def power(self, k):
        """
        Return the total power in units of (Mpc/h)^3
        """
        return 2*self.f*self.zeldovich_power(k) + self.broadband_power(k)
    #---------------------------------------------------------------------------
    
#endclass DarkMatterP01
#-------------------------------------------------------------------------------


class InterpDarkMatterPowerMoment(object):
    """
    A class to compute the dark matter power moment by interpolating simulation 
    results
    """
    def __init__(self, power_lin, z, sigma8, f):
      
        # store the input arguments
        self._power_lin = power_lin
        self._cosmo     = self._power_lin.GetCosmology()
        
        # make sure power spectrum redshift is 0
        msg = "input linear power spectrum must be defined at z = 0"
        assert self._power_lin.GetRedshift() == 0., msg
        
        # set the initial redshift, sigma8 
        self.z = z
        self.sigma8 = sigma8
        self.f = f
        
        # load the simulation data and store splines
        self._make_interpolation_table()
        
    #end __init__
        
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
        self._z = val
        
        del self.D
    
    #---------------------------------------------------------------------------
    @property
    def D(self):
        """
        The growth function, normalized to unity at z = 0
        """
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
    @property
    def sigma8(self):
        """
        Sigma_8 at `z=0` to compute the spectrum at, which gives the 
        normalization of the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        self._sigma8 = val
        
    #---------------------------------------------------------------------------
    @property
    def power_norm(self):
        """
        Sigma 8 normalization for power spectrum
        """
        return (self.sigma8 / self.cosmo.sigma8())**2
        
    #---------------------------------------------------------------------------
    @property
    def f(self):
        """
        The growth rate, defined as the `dlnD/dlna`. 
     
        If the parameter has not been explicity set, it defaults to the value
        at `self.z`
        """
        try:
            return self._f
        except AttributeError:
            return self.cosmo.f_z(self.z)
    
    @f.setter
    def f(self, val):
        self._f = val
                            
    #---------------------------------------------------------------------------  
    def power(self, k):
        """
        Return the power as computed from the interpolation table
        """        
        keys = self.interpolation_table.keys()
        this_x = self.interpolation_variable
                
        ihi = bisect.bisect(keys, this_x)
        ilo = ihi - 1
        
        if this_x < np.amin(keys) or this_x > np.amax(keys):
            if (ihi == 0): index = ihi
            elif (ihi == len(keys)): index = ihi-1
            
            normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
            factor = self.interpolation_variable * normed_power
            x = keys[index]
            return self.interpolation_table[x](k)*factor
        
        x_lo = keys[ilo]
        x_hi = keys[ihi]
        w = (this_x - x_lo) / (x_hi - x_lo) 
        
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = self.interpolation_variable * normed_power
        y_lo = self.interpolation_table[x_lo](k)
        y_hi = self.interpolation_table[x_hi](k)
        return factor*((1 - w)*y_lo + w*y_hi)
        
    #---------------------------------------------------------------------------
#endclass InterpolatedDarkMatterPowerMoment

#-------------------------------------------------------------------------------
class DarkMatterP11(InterpDarkMatterPowerMoment):
    """
    Dark matter model for mu^4 term of P11. Computed by interpolating 
    simulation data as a function of (f*sigma8)^2
    """
    
    def __init__(self, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterP11, self).__init__(*args, **kwargs)
        
    #---------------------------------------------------------------------------
    @property
    def interpolation_variable(self):
        """
        The interpolation variable to use
        """
        return (self.f*self.sigma8)**2
    
    #---------------------------------------------------------------------------
    def _make_interpolation_table(self):
        """
        Load the simulation data and make the interpolation time
        """
        # cosmology and linear power spectrum for teppei's sims
        cosmo = pygcl.Cosmology("teppei_sims.ini", pygcl.Cosmology.EH_NoWiggle)
        Plin = pygcl.LinearPS(cosmo, 0.)
        
        # the interpolation data
        redshifts = [0., 0.509, 0.989]
        data = [P11_mu4_z_0_000(), P11_mu4_z_0_509(), P11_mu4_z_0_989()]
        interp_vars = [(cosmo.f_z(z)*cosmo.sigma8())**2 for z in redshifts]
        
        # make the interpolation table
        self.interpolation_table = collections.OrderedDict()
        for i, (x, d) in enumerate(zip(interp_vars, data)):
            
            y = d[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(d[:,0]) * cosmo.f_z(redshifts[i])**2)
            self.interpolation_table[x] = spline(d[:,0], y)
            
    #---------------------------------------------------------------------------
#endclass DarkMatterP11

#-------------------------------------------------------------------------------
class DarkMatterPdv(InterpDarkMatterPowerMoment):
    """
    Dark matter model for density -- velocity divergence cross power spectrum
    Pdv. Computed by interpolating simulation data as a function of (f*sigma8)^2
    """
    
    def __init__(self, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterPdv, self).__init__(*args, **kwargs)
        
    #---------------------------------------------------------------------------
    @property
    def interpolation_variable(self):
        """
        The interpolation variable to use
        """
        return self.f*self.sigma8**2
    
    #---------------------------------------------------------------------------
    def _make_interpolation_table(self):
        """
        Load the simulation data and make the interpolation time
        """
        # cosmology and linear power spectrum for teppei's sims
        cosmo = pygcl.Cosmology("teppei_sims.ini", pygcl.Cosmology.EH_NoWiggle)
        Plin = pygcl.LinearPS(cosmo, 0.)
        
        # the interpolation data
        redshifts = [0., 0.509, 0.989]
        data = [Pdv_mu0_z_0_000(), Pdv_mu0_z_0_509(), Pdv_mu0_z_0_989()]
        interp_vars = [cosmo.f_z(z)*cosmo.sigma8()**2 for z in redshifts]
        
        # make the interpolation table
        self.interpolation_table = collections.OrderedDict()
        for i, (x, d) in enumerate(zip(interp_vars, data)):
            
            y = d[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(d[:,0]) * cosmo.f_z(redshifts[i]))
            self.interpolation_table[x] = spline(d[:,0], y, k=2)
            
    #---------------------------------------------------------------------------
    
