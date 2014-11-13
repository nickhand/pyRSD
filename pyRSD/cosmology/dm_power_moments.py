"""
 dm_power_moments.py
 pyRSD: functions for computing dark matter power moments: P00, P01, P11
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/12/2014
"""
from .cosmo import Cosmology
from . import parameters, power, growth

import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

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
    
    dR1_dlna = InterpolatedUnivariateSpline(z_center, np.diff(R1) / np.diff(np.log(1./(1+z_spline))))
    dR2_dlna = InterpolatedUnivariateSpline(z_center, np.diff(R2) / np.diff(np.log(1./(1+z_spline))))
    dR3_dlna = InterpolatedUnivariateSpline(z_center, np.diff(R3) / np.diff(np.log(1./(1+z_spline))))
    
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
    R1_spline = InterpolatedUnivariateSpline(Ri_zs, Ri[:,0])
    R2_spline = InterpolatedUnivariateSpline(Ri_zs, Ri[:,1])
    R3_spline = InterpolatedUnivariateSpline(Ri_zs, Ri[:,2])
    
    # return R1, R2, R3 at this z
    return R1_spline(z), R2_spline(z), R3_spline(z)
#end Ri_z

#-------------------------------------------------------------------------------
class DarkMatterPowerMoment(object):
    """
    A class to compute a generic dark matter power moment
    """
    def __init__(self, z,
                       cosmo={'default' : parameters.default_params, 'flat': True}, 
                       transfer_fit="CAMB",
                       camb_kwargs={},
                       zeldovich_file=None, 
                       model_type='A'):
        """
        Parameters
        ----------
        z : float
            The redshift of the power spectra
        cosmo : {dict, str, Cosmology}
            The cosmology to use in computing the spectra
        transfer_fit : str
            The type of transfer function to use
        camb_kwargs : dict
            Dictionary of keyword arguments to pass to the `power.Power` class
        zeldovich_file : str
            The name of the file holding the Zel'dovich power spectrum
        model_type : str, {'A', 'B'}
            The model type, either 'A' or 'B'
        """
        # the model type
        self.model_type = model_type
        
        # save the redshift
        self._z = z
        
        # save the cosmology
        self._cosmo = cosmo if isinstance(cosmo, Cosmology) else Cosmology(cosmo)

        # save the linear power
        self._power_lin = power.Power(z=self._z, transfer_fit=transfer_fit, 
                                        cosmo=self._cosmo, **camb_kwargs)
                                        
        # save the zeldovich power
        self._zeldovich_data = zeldovich_file
        if zeldovich_file is not None:
            try:
                self._zeldovich_data = np.loadtxt(zeldovich_file)
            except:
                raise ValueError("Error reading Zel'dovich power from %s" %zeldovich_file)
        
    #end __init__
    
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        """
        The read-only cosmology object
        """
        return self._cosmo
        
    @property
    def power_lin(self):
        """
        The read-only power.Power object holding the class
        """
        return self._power_lin
        
    @property
    def z(self):
        """
        The read-only redshift
        """
        return self._z
    
    @property
    def k(self):
        """
        The wavenumbers in units of h/Mpc
        """ 
        if self._zeldovich_data is None:
            raise ValueError("No data file specified for Zel'dovich power spectrum")   
        return self._zeldovich_data[:,0]
        
    @property
    def zeldovich_power(self):
        """
        The zeldovich power in units of (Mpc/h)^3
        """ 
        if self._zeldovich_data is None:
            raise ValueError("No data file specified for Zel'dovich power spectrum")   
        return self._zeldovich_data[:,1]
    
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
    
    def __init__(self, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterP00, self).__init__(*args, **kwargs)
        
    #---------------------------------------------------------------------------
    @property
    def model_params(self):
        """
        Return the model parameters for P00, needed to evaulate the broadband 
        power correction for the model type specified by `self.model_type`
        """
        # sigma 8 at this redshift, needed by both models
        sigma8 = self.power_lin.sigma_r(8., self.z)
        
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
        
        return Ck(self.k, 31.)*Fk(self.k, *self.model_params)
        
    #---------------------------------------------------------------------------
#endclass DarkMatterP00

#-------------------------------------------------------------------------------

class DarkMatterP01(DarkMatterPowerMoment):
    
    def __init__(self, *args, **kwargs):
        
        # initalize the base class
        super(DarkMatterP01, self).__init__(*args, **kwargs)
        
    #---------------------------------------------------------------------------
    @property 
    def f(self):
        """
        The growth rate f = dlnD / dlna
        """
        return growth.growth_rate(self.z, params=self.cosmo)
        
    #---------------------------------------------------------------------------
    @property
    def zeldovich_power(self):
        """
        The zeldovich power in units of (Mpc/h)^3
        """ 
        if self._zeldovich_data is None:
            raise ValueError("No data file specified for Zel'dovich power spectrum")   
        
        return 2*self.f*self._zeldovich_data[:,1]
        
    #---------------------------------------------------------------------------
    @property
    def model_params(self):
        """
        Return the model parameters for P01, needed to evaulate the broadband 
        power correction for the model type specified by `self.model_type`
        """
        # sigma 8 at this redshift, needed by both models
        sigma8 = self.power_lin.sigma_r(8., self.z)
        
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
        
        return Ck(self.k, 31.)*Fk(self.k, *self.model_params)
        
    #---------------------------------------------------------------------------

#endclass DarkMatterP01
#-------------------------------------------------------------------------------

def P00_runPB(**kwargs):
    """
    Return the dark matter P00 for the runPB simulations, at z = 0.55
    """
    z = 0.55
    omegab = 0.022/0.69**2
    omegac = 0.292 - omegab
    cosmo = {'flat' : True, 'default' : 'WMAP9_eCMB', 'h' : 0.69, 'n' : 0.965, 'sigma_8' : 0.82, 'omegac' : omegac, 'omegab': omegab, 'omegar': 0., 'Tcmb' : 2.725, 'omegar':0.}
    zeldovich_file = os.environ['PROJECTS_DIR'] + "/RSD-Modeling/GalaxyModelDev/data/P00_zeldovich_z0.55.dat"
    
    return DarkMatterP00(z, cosmo=cosmo, zeldovich_file=zeldovich_file, **kwargs)
    
#end P00_runPB

#-------------------------------------------------------------------------------
def P01_runPB(**kwargs):
    """
    Return the dark matter P01 for the runPB simulations, at z = 0.55
    """
    z = 0.55
    omegab = 0.022/0.69**2
    omegac = 0.292 - omegab
    cosmo = {'flat' : True, 'default' : 'WMAP9_eCMB', 'h' : 0.69, 'n' : 0.965, 'sigma_8' : 0.82, 'omegac' : omegac, 'omegab': omegab, 'omegar': 0., 'Tcmb' : 2.725, 'omegar':0.}
    zeldovich_file = os.environ['PROJECTS_DIR'] + "/RSD-Modeling/GalaxyModelDev/data/P01_zeldovich_z0.55.dat"
    
    return DarkMatterP01(z, cosmo=cosmo, zeldovich_file=zeldovich_file, **kwargs)
    
#end P01_runPB

#-------------------------------------------------------------------------------
    
    
