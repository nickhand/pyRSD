"""
 analytic.py
 pyRSD: functions for computing analytic power moments
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/12/2014
"""
from .. import pygcl, numpy as np, os
from ..data import P11_mu4_z_0_000, P11_mu4_z_0_509, P11_mu4_z_0_989
from ..data import Pdv_mu0_z_0_000, Pdv_mu0_z_0_509, Pdv_mu0_z_0_989
from ..data import Phh_gp_fits
from .tools import RSDSpline as spline
from . import INTERP_KMIN, INTERP_KMAX

import collections
import bisect

K_SPLINE = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 100)
SIGMA8_MIN = 0.1
SIGMA8_MAX = 2.5
INTERP_PTS = 50

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

#-------------------------------------------------------------------------------
class HaloP00(object):
    """
    Class to compute P00 for halos as a function of linear bias `b1` and 
    redshift `z`
    """
    def __init__(self, P00_model):
        """
        Initialize with a `DarkMatterP00` object
        """
        # doesnt make a copy -- just a reference so that the redshift will 
        # be updated
        self.P00_model = P00_model
        
        # load the data
        gps = Phh_gp_fits()
        self.gp_stoch = gps['stoch']
        self.gp_Phm = gps['Phm']
        
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        The redshift, taken from the `P00_model`; 
        """
        return self.P00_model.z
        
    @z.setter
    def z(self, val):
        self.P00_model.z = z
        
    #---------------------------------------------------------------------------
    def Pmm(self, k):
        """
        The dark matter density auto correlation as computed from 
        `self.P00_model`
        """
        return self.P00_model.power(k)
        
    #---------------------------------------------------------------------------
    def Phm(self, b1, k, return_error=False):
        """
        The halo-matter cross correlation at the bias specified by `self.b1`, 
        as computed from the Gaussian Process fit
        """
        x = np.vstack((np.ones(len(k))*self.z, np.ones(len(k))*b1, k)).T
        if return_error:
            res, sig_sq = self.gp_Phm.predict(x, eval_MSE=True)
        else:
            res = self.gp_Phm.predict(x)
        toret = b1*self.P00_model.zeldovich_power(k) + res
        
        if return_error:
            return toret, sig_sq**0.5
        else:
            return toret
        
    #---------------------------------------------------------------------------
    def stochasticity(self, b1, k, return_error=False):
        """
        The stochasticity as computed from simulations using a Gaussian Process
        fit
        """
        x = np.vstack((np.ones(len(k))*self.z, np.ones(len(k))*b1, k)).T
        if return_error:
            lam, sig_sq = self.gp_stoch.predict(x, eval_MSE=True)
        else:
            lam = self.gp_stoch.predict(x)

        if return_error:
            return lam, sig_sq**0.5
        else:
            return lam
    
    #---------------------------------------------------------------------------
    def power(self, b1, k, return_error=False):
        """
        Return the halo P00 power, optionally returning the error as computed
        from the Gaussian Process fit
        """        
        Pmm = self.Pmm(k)
        if return_error:
            Phm, Phm_err = self.Phm(b1, k, return_error)
            lam, lam_err = self.stochasticity(b1, k, return_error)
            err = np.sqrt((2*b1*Phm_err)**2 + lam_err**2)
        else:
            Phm = self.Phm(b1, k, return_error)
            lam = self.stochasticity(b1, k, return_error)
        
        toret = 2*b1*Phm - b1**2*Pmm + lam
        if return_error:
            return toret, err
        else:
            return toret
        
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------
        
        
    
