"""
 cosmology.pyx
 pyPT: cosmology module
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/21/2014
"""
import cosmo_dict
import numpy as np
cimport numpy as np

cdef class cosmology:
    
    def __init__(self, *args, **kwargs):
        """
        Inputs should specify the necessay cosmological parameters. 
        
        Notes
        -----
        At most 1 positional argument can be supplied. If a string, the 
        parameters will be set from file if the path exists, or set to the 
        builtin parameter set defined by the input name. If a dictionary, 
        the parameters will be updated from it. Parameters will be updated 
        from keyword arguments as well.
        """
        # set up the cosmo dict
        self.pdict = cosmo_dict.params(*args, **kwargs)
        
        # initialize the TF parameters
        TFset_parameters(self.pdict['omega_m_0']*self.pdict['h']**2,
                         self.pdict['omega_b_0']/self.pdict['omega_m_0'], 
                         self.pdict['Tcmb_0'])

    #---------------------------------------------------------------------------
    def _vectorize(self, x):
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        return x
    #----------------------------------------------------------------------------
    cpdef E(self, z):
         """
         The unitless Hubble expansion rate at redshift z, 
         modified to include non-constant w parameterized linearly 
         with z ( w = w0 + w1*z )
         """
         z = self._vectorize(z)
         return np.sqrt(self.pdict['omega_m_0']*(1.+z)**3 \
                            + self.pdict['omega_r_0']*(1.+z)**4 \
                            + self.pdict['omega_k_0']*(1.+z)**2 \
                            + self.pdict['omega_l_0']*np.exp(3.*self.pdict['w1']*z) \
                            *(1.+ z)**(3.*(1+self.pdict['w0']-self.pdict['w1'])))
    #end E
    #---------------------------------------------------------------------------
    cpdef omega_m_z(self, z):
        """
        The matter density omega_m as a function of redshift

        From Lahav et al. 1991 equations 11b-c. This is equivalent to 
        equation 10 of Eisenstein & Hu 1999.
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return self.pdict['omega_m_0']*(1.+z)**3./self.E(z)**2.
    #end omega_m_z
    
    #---------------------------------------------------------------------------
    cpdef omega_l_z(self, z):
        """
        The dark energy density omega_l as a function of redshift
        
        Parameters
        ----------
        z : float or numpy.ndarray
            the redshift to compute the function at
        """
        return self.pdict['omega_l_0']/self.E(z)**2
    #end omega_l_z
    
    #---------------------------------------------------------------------------
    cpdef growth_rate(self, z):
        """
        The growth rate, which is the logarithmic derivative of the growth 
        factor with respect to scale factor, denoted by f usually. Fitting formula 
        from eq 5 of Hamilton 2001; originally from Lahav et al. 1991. 
        """
        om_m = self.omega_m_z(z)
        om_l = self.omega_l_z(z)

        return om_m**(4./7) + (1 + 0.5*om_m)*om_l/70.
    #end growth_rate

    #---------------------------------------------------------------------------
    cpdef growth_factor(self, z):
        """
        The linear growth factor, using approximation
        from Carol, Press, & Turner (1992), else integrate the ODE.
        Normalized to 1 at z = 0.
        """
        om_m = self.omega_m_z(z)
        om_l = self.omega_l_z(z)

        om_m_0 = self.pdict['omega_m_0']
        om_l_0 = self.pdict['omega_l_0']
        norm = 2.5*om_m_0/(om_m_0**(4./7.) - om_l_0 + (1.+0.5*om_m_0)*(1.+om_l_0/70.))
        return 2.5*om_m/(om_m**(4./7.)-om_l+(1.+0.5*om_m)*(1.+om_l/70.))/(norm*(1+z))

    #end growth_factor
