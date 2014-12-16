"""
 dm_power_moments.py
 pyRSD: functions for computing dark matter power moments: P00, P01, P11
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/12/2014
"""
from .. import pygcl, numpy as np, os
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline as spline

#-------------------------------------------------------------------------------
def dRi_dlna(z):
    """
    These are the derivative of the R_i(z) parameters of model A, with respect
    to the log of the scale factor 
    
    Parameters
    ----------
    z : float, array_like
        The redshift to evaluate the R parameters
        
    Returns
    -------
    dR1_dlna, dR2_dlna, dR3_dlna : float, array_like
        The derivative parameters evaluated at the input redshift
    """
    # compute R1, R2, R3 over 
    z_spline = np.linspace(0., 2., 1000)
    z_center = 0.5*(z_spline[1:] + z_spline[:-1])
    R1, R2, R3 = Ri_z(z_spline)
    
    dR1_dlna = spline(z_center, np.diff(R1) / np.diff(np.log(1./(1+z_spline))))
    dR2_dlna = spline(z_center, np.diff(R2) / np.diff(np.log(1./(1+z_spline))))
    dR3_dlna = spline(z_center, np.diff(R3) / np.diff(np.log(1./(1+z_spline))))
    
    return dR1_dlna(z), dR2_dlna(z), dR3_dlna(z)
#end dRi_dlna

#-------------------------------------------------------------------------------
def Ri_z(z):
    """
    These are the R_i(z) parameters of model A, as given by Zvonimir. They are
    redshift dependent
    
    Parameters
    ----------
    z : float, array_like
        The redshift to evaluate the R parameters
        
    Returns
    -------
    R1, R2, R3 : float, array_like
        The parameters evaluated at the input redshift
    """
    # these are the values from Zvonimir
    Ri_zs = np.array([0., 0.5, 1., 2.])
    Ri = np.empty((4, 3))
    Ri[0,:] = [2.8260, 2.30098, 1.3614]
    Ri[1,:] = [2.3670, 1.5930, 0.7370]
    Ri[2,:] = [2.2953, 1.3272, 0.00034365]
    Ri[3,:] = [2.0858, 0.7878, 0.00017]

    # now make the splines
    R1_spline = spline(Ri_zs, Ri[:,0])
    R2_spline = spline(Ri_zs, Ri[:,1])
    R3_spline = spline(Ri_zs, Ri[:,2])
    
    # return R1, R2, R3 at this z
    return R1_spline(z), R2_spline(z), R3_spline(z)
#end Ri_z

K_SPLINE = np.logspace(-3, np.log10(2.), 500)

#-------------------------------------------------------------------------------
class DarkMatterPowerMoment(object):
    """
    A class to compute a generic dark matter power moment
    """
    def __init__(self, k_eval, power_lin, z, sigma8, model_type='A'):
        
        # check input k_eval values
        if (np.amin(k_eval) < 1e-3 or np.amax(k_eval) > 2.):
            raise ValueError("Probably not a good idea to compute Zel'dovich power for k < 1e-3 or k > 2 h/Mpc")
            
        # store the input arguments
        self._k_eval    = k_eval
        self._power_lin = power_lin
        self.model_type = model_type
        
        # make sure power spectrum redshift is 0
        assert self._power_lin.GetRedshift() == 0., "Integrals: input linear power spectrum must be defined at z = 0"
        
        # set the initial redshift, sigma8 
        self.z = z
        self.sigma8 = sigma8

    #end __init__
    
    #---------------------------------------------------------------------------
    @property
    def k_eval(self):
        """
        Wavenumbers to evaluate the spectrum at [units: `h/Mpc`]
        """
        return self._k_eval
    
    @k_eval.setter
    def k_eval(self, val):
        self._k_eval = val

        # delete dependencies
        del self.zeldovich_power
                    
    #---------------------------------------------------------------------------
    @property
    def power_lin(self):
        """
        Linear power spectrum object
        """
        return self._power_lin
        
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
        
        # delete dependencies
        del self.D, self.sigma8_z, self.zeldovich_power
        self.zeldovich_base.SetRedshift(val)
    
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
        
        # delete dependencies
        del self.zeldovich_power
        self.zeldovich_base.SetSigma8(val)
    
    #---------------------------------------------------------------------------
    @property
    def sigma8_norm(self):
        """
        The factor needed to normalize the spectrum to the desired sigma_8, as
        specified by `self.sigma8`
        """
        return (self.sigma8 / self._power_lin.GetCosmology().sigma8())

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
    # Derived quantities
    #---------------------------------------------------------------------------
    @property
    def D(self):
        """
        Growth function at `self.z`, normalized to unity at z=0
        """
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
            return self.power_lin.GetCosmology().f_z(self.z)

    @f.setter
    def f(self, val):
        self._f = val
        if hasattr(self, '_P01_model'): self.P01_model.f = val

        # delete dependencies 
        self._delete_power()
    
    #---------------------------------------------------------------------------
    @property
    def sigma8_z(self):
        """
        Sigma_8 at `self.z`
        """
        try:
            return self.sigma8_norm * self._sigma8_z
        except AttributeError:
            self._sigma8_z = self.power_lin.GetCosmology().Sigma8_z(self.z)
            return self.sigma8_norm * self._sigma8_z

    @sigma8_z.deleter
    def sigma8_z(self):
        try:
            del self._sigma8_z
        except AttributeError:
            pass
            
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
    @property
    def zeldovich_power(self):
        """
        Return the power from the Zel'dovich term
        """
        return self.k_eval*0.
    
    #---------------------------------------------------------------------------
    @property
    def broadband_power(self):
        """
        Return the power from the broadband correction term
        """
        return self.k_eval*0
        
    #---------------------------------------------------------------------------
    @property
    def power(self):
        """
        Return the total power in units of (Mpc/h)^3
        """
        return self.zeldovich_power + self.broadband_power
        
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
    @property
    def zeldovich_power(self):
        """
        Return the power from the Zel'dovich term
        """
        try:
            return self._zeldovich_power
        except AttributeError:
            P00 = pygcl.ZeldovichP00(self.zeldovich_base)
            self._zeldovich_power = P00(self.k_eval)
            return self._zeldovich_power
    
    @zeldovich_power.deleter
    def zeldovich_power(self):
        try:
            del self._zeldovich_power
        except AttributeError:
            pass
            
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
            R1, R2, R3 = Ri_z(self.z)
            return A0, R1, R2, R3
        else:
            
            # simple power laws in sigma8 for model B
            A0 = 1529.87 * sigma8**3.9 
            A2 = 1299.75 * sigma8**3.0
            A4 = 758.31 * sigma8**2.2 
            return A0, A2, A4
            
    #---------------------------------------------------------------------------
    @property
    def broadband_power(self):
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
        
        return Ck(self.k_eval, 31.)*Fk(self.k_eval, *self.model_params)
        
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
    def zeldovich_power(self):
        """
        Return the power from the Zel'dovich term
        """
        try:
            return 2*self.f*self._zeldovich_power
        except AttributeError:
            P01 = pygcl.ZeldovichP01(self.zeldovich_base)
            self._zeldovich_power = P01(self.k_eval)
            return 2*self.f*self._zeldovich_power
    
    @zeldovich_power.deleter
    def zeldovich_power(self):
        try:
            del self._zeldovich_power
        except AttributeError:
            pass
                    
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
            R1, R2, R3 = Ri_z(self.z)
            dR1_dlna, dR2_dlna, dR3_dlna = dRi_dlna(self.z)
            
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
    @property
    def broadband_power(self):
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
        
        return Ck(self.k_eval, 31.)*Fk(self.k_eval, *self.model_params)
        
    #---------------------------------------------------------------------------
#endclass DarkMatterP01


    
