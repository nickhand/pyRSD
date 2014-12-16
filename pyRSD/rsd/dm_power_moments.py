"""
 dm_power_moments.py
 pyRSD: functions for computing dark matter power moments: P00, P01, P11
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/12/2014
"""
from .. import pygcl, numpy as np, os
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import collections
import bisect

KSPLINE = np.logspace(-3, 0, 256)
SIGMA8_MIN = 0.5
SIGMA8_MAX = 1.2
INTERP_PTS = 30

#-------------------------------------------------------------------------------
class DarkMatterPowerMoment(object):
    """
    A class to compute a generic dark matter power moment
    """
    def __init__(self, power_lin, z, sigma8, model_type='A'):
      
        # store the input arguments
        self._power_lin = power_lin
        self._cosmo     = self._power_lin.GetCosmology()
        self.model_type = model_type
        
        # make sure power spectrum redshift is 0
        msg = "Integrals: input linear power spectrum must be defined at z = 0"
        assert self._power_lin.GetRedshift() == 0., msg
        
        # set the initial redshift, sigma8 
        self.z = z
        self.sigma8 = sigma8
        
        # initialize splines
        self._initialize_Rparam_splines()
        
        # set up the zeldovich power interpolation table
        self._compute_zeldovich_power_table()
    #end __init__
    
    #---------------------------------------------------------------------------
    # initialization functions
    #---------------------------------------------------------------------------
    def _initialize_Rparam_splines(self):
        """
        Initialize the splines needed for the broadband correction
        """
        # these are the values from Zvonimir
        Ri_zs = np.array([0., 0.5, 1., 2.])
        Ri = np.empty((4, 3))
        Ri[0,:] = [2.8260, 2.30098, 1.3614]
        Ri[1,:] = [2.3670, 1.5930, 0.7370]
        Ri[2,:] = [2.2953, 1.3272, 0.00034365]
        Ri[3,:] = [2.0858, 0.7878, 0.00017]
        
        # now make the splines
        self.R1_spline = spline(Ri_zs, Ri[:,0])
        self.R2_spline = spline(Ri_zs, Ri[:,1])
        self.R3_spline = spline(Ri_zs, Ri[:,2])
        
        # compute R1, R2, R3 over 
        z_spline = np.linspace(0., 2., 1000)
        z_center = 0.5*(z_spline[1:] + z_spline[:-1])
        R1 = self.R1_spline(z_spline)
        R2 = self.R1_spline(z_spline)
        R3 = self.R1_spline(z_spline)

        self.dR1_dlna = spline(z_center, np.diff(R1) / np.diff(np.log(1./(1+z_spline))))
        self.dR2_dlna = spline(z_center, np.diff(R2) / np.diff(np.log(1./(1+z_spline))))
        self.dR3_dlna = spline(z_center, np.diff(R3) / np.diff(np.log(1./(1+z_spline)))) 
    #end _initialize_Rparam_splines

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
        if val < SIGMA8_MIN or val > SIGMA8_MAX:
            raise ValueError("Sigma_8 out of range [%f, %f]" %(SIGMA8_MIN, SIGMA8_MAX))
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
        keys = self.zeldovich_power_table.keys()
        ihi = bisect.bisect(keys, self.sigma8)
        ilo = ihi - 1
        
        s8_lo = keys[ilo]
        s8_hi = keys[ihi]
        w = (self.sigma8 - s8_lo) / (s8_hi - s8_lo) 
        return (1 - w)*self.zeldovich_power_table[s8_lo](k) + w*self.zeldovich_power_table[s8_hi](k)
    
    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        Return the power from the broadband correction term
        """
        return k*0
        
    #---------------------------------------------------------------------------
    def power(self, k):
        """
        Return the total power in units of (Mpc/h)^3
        """
        return k*0
        
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
            self.zeldovich_power_table[sigma8] = spline(KSPLINE, P00(KSPLINE))

    #end _compute_zeldovich_power_table
    
    #---------------------------------------------------------------------------
    @property
    def model_params(self):
        """
        Return the model parameters for P00, needed to evaulate the broadband 
        power correction for the model type specified by `self.model_type`
        """
        # sigma 8 at this redshift, needed by both models
        sigma8 = self.sigma8_z
        
        if self.model_type == 'A':

            # base model params for model A
            A0 = 743.854 * (sigma8/0.81)**3.902
            R1 = self.R1_spline(self.z)
            R2 = self.R2_spline(self.z)
            R3 = self.R3_spline(self.z)
            return A0, R1, R2, R3
        else:
            
            # simple power laws in sigma8 for model B
            A0 = 1529.87 * sigma8**3.9 
            A2 = 1299.75 * sigma8**3.0
            A4 = 758.31 * sigma8**2.2 
            return A0, A2, A4
            
    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3
        """ 
        # define the redshift dependent functions, Fk
        if self.model_type == 'A':
            Fk = lambda k, A0, R1, R2, R3: A0*(1. + (R2*k)**2) / (1. + (R1*k)**2 + (R3*k)**4)
        else:
            Fk = lambda k, A0, A2, A4: A0 - A2*k**2 + A4*k**4

        # the redshift independent piece, C(k)
        Ck = lambda k, R0: 1. - 1./(1. + (R0*k)**2)
        
        return Ck(k, 31.)*Fk(k, *self.model_params)
        
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
    
    def __init__(self, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterP01, self).__init__(*args, **kwargs)
          
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
            self.zeldovich_power_table[sigma8] = spline(KSPLINE, P01(KSPLINE))

    #end _compute_zeldovich_power_table
    
    #---------------------------------------------------------------------------
    @property
    def model_params(self):
        """
        Return the model parameters for P01, needed to evaulate the broadband 
        power correction for the model type specified by `self.model_type`
        """
        # sigma 8 at this redshift, needed by both models
        sigma8 = self.sigma8_z
        
        if self.model_type == 'A':

            # base model params for model A
            A0 = 743.854 * (sigma8/0.81)**3.902
            R1 = self.R1_spline(self.z)
            R2 = self.R2_spline(self.z)
            R3 = self.R3_spline(self.z)
            dR1_dlna = self.dR1_dlna(self.z)
            dR2_dlna = self.dR2_dlna(self.z)
            dR3_dlna = self.dR3_dlna(self.z)
            
            params = (A0, R1, R2, R3)
            derivs = (3.9*self.f*A0, dR1_dlna, dR2_dlna, dR3_dlna)
            return params, derivs
        else:
            
            # simple power laws in sigma8 for model B
            A0 = 1529.87 * sigma8**3.9 
            A2 = 1299.75 * sigma8**3.0
            A4 = 758.31 * sigma8**2.2 
            return 3.9*self.f*A0, 3.*self.f*A2, 2.2*self.f*A4
    
    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3
        """ 
        # define the redshift dependent functions, Fk
        if self.model_type == 'A':
            
            def Fk(k, params, derivs):
                
                A0, R1, R2, R3 = params
                dA0, dR1, dR2, dR3 = derivs
                norm = (1. + (R1*k)**2 + (R3*k)**4)
                
                # each term of the derivative
                term1 = dA0 * (1. + (R2*k)**2) / norm
                term2 = dR2 * A0 * 2. * k**2 * R2 /  norm
                term3 = -dR3 * A0 * (1. + (R2*k)**2) / norm**2 * (4*k**4*R3**3)
                term4 = -dR1 * A0 * (1. + (R2*k)**2) / norm**2 * (2*k**2*R1)
                return term1 + term2 + term3 + term4
        else:
            Fk = lambda k, A0, A2, A4: A0 - A2*k**2 + A4*k**4

        # the redshift independent piece, C(k)
        Ck = lambda k, R0: 1. - 1./(1. + (R0*k)**2)
        
        return Ck(k, 31.)*Fk(k, *self.model_params)
        
    #---------------------------------------------------------------------------
    def power(self, k):
        """
        Return the total power in units of (Mpc/h)^3
        """
        return 2*self.f*self.zeldovich_power(k) + self.broadband_power(k)
    #---------------------------------------------------------------------------
    
#endclass DarkMatterP01
#-------------------------------------------------------------------------------


    
