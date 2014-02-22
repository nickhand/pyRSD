#!python
#cython: boundscheck=False
#cython: wraparound=False
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
from scipy.integrate import quad

cdef class power_eh:
    
    def __cinit__(self, *args, **kwargs):
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
        self.P0_full = 0.
        self.P0_nw = 0.
        
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
    #---------------------------------------------------------------------------
    cpdef H(self, z):
        """
        The value of the Hubble constant at redshift z in km/s/Mpc
        
        Parameters
        ----------
        z : {float, np.ndarray}
            the redshift to compute the function at
        """
        return 100.*self.pdict['h']*self.E(z)
    #end H
    
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
        z : {float, np.ndarray}
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
        z : {float, np.ndarray}
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
    
    #---------------------------------------------------------------------------
    def w_tophat(self, k, r):
        """
        The k-space Fourier transform of a spherical tophat.
        """
        return 3.*(np.sin(k*r)-k*r*np.cos(k*r))/(k*r)**3
    #end w_tophat

    #---------------------------------------------------------------------------
    def compute_P0_nw(self):
        """
        Compute the normalization "no-wiggles" power spectrum based on the
        value of sigma_8, which is sigma_r at r = 8/h Mpc
        """
        self.P0_nw = 1.
        I_dk = lambda k: k**2*self.Pk_nowiggles(k, 0.)*self.w_tophat(k, 8.)**2
        I = quad(I_dk, 0, np.inf)
        self.P0_nw = (self.pdict['sigma_8']**2)*(2*np.pi**2)/I[0]
    #end compute_P0_nw
    
    #---------------------------------------------------------------------------
    def compute_P0_full(self):
        """
        Compute the normalization of the full power spectrum based on the
        value of sigma_8, which is sigma_r at r = 8/h Mpc
        """
        self.P0_full = 1.
        I_dk = lambda k: k**2*self.Pk_full(k, 0.)*self.w_tophat(k, 8.)**2
        I = quad(I_dk, 0, np.inf)
        self.P0_full = (self.pdict['sigma_8']**2)*(2*np.pi**2)/I[0]
    #end compute_P0_full
    
    #-------------------------------------------------------------------------------
    cpdef Pk_full(self, k_hMpc, z):
        """
        Compute the CDM + baryon linear power spectrum using the full 
        Eisenstein and Hu transfer function fit, appropriately normalized 
        via sigma_8, at redshift z. The primordial spectrum is assumed 
        to be proportional to k^n.
        
        Calls the function TFfit_onek() from the tf_fit.c code.

        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber in units of h / Mpc
        z : float
            the redshift to compute the spectrum at
        
        Returns
        -------
        P_k : numpy.ndarray
            full, linear power spectrum in units of Mpc^3/h^3
        """
        k = self._vectorize(k_hMpc)
        
        # compute P0 if it is not yet computed
        if self.P0_full == 0.:
            self.compute_P0_full()
        
        Tfull = self.Tk_full(k)
        fg = self.growth_factor(z)
        return self.P0_full * k**self.pdict['n_s'] * (Tfull*fg)**2 
    #end Pk_full
    
    #---------------------------------------------------------------------------
    cpdef Pk_nowiggles(self, k_hMpc, z):
        """
        Compute the CDM + baryon linear power spectrum with no oscillatory 
        features using the "no-wiggles" Eisenstein and Hu transfer function fit, 
        appropriately normalized via sigma_8, at redshift z. The primordial 
        spectrum is assumed to be proportional to k^n.
        
        Calls the function TFnowiggles() from the tf_fit.c code.

        Parameters
        ----------
        k_hMpc : {float, np.ndarray}
            the wavenumber in units of h / Mpc
        z : float
            the redshift to compute the spectrum at
        
        Returns
        -------
        P_k : numpy.ndarray
            no-wiggle, linear power spectrum in units of Mpc^3/h^3
        """
        k = self._vectorize(k_hMpc)
        
        # compute P0 if it is not yet computed
        if self.P0_nw == 0.:
            self.compute_P0_nw()
        
        Tfull = self.Tk_nowiggles(k)
        fg = self.growth_factor(z)
        return self.P0_nw * k**self.pdict['n_s'] * (Tfull*fg)**2 
    #end Pk_nowiggles
    
    #---------------------------------------------------------------------------
    cpdef np.ndarray Tk_full(self, np.ndarray k):
        """
        Wrapper function to call TFfit_onek() from tf_fit.c and compute the 
        full EH transfer function. 
        """
        cdef float baryon_piece, cdm_piece, this_Tk
        cdef int N = k.shape[0]
        cdef np.ndarray[double, ndim=1] output = np.empty(N)
        cdef int i
        
        for i in xrange(N):
            this_Tk = TFfit_onek(k[i]*self.pdict['h'], &baryon_piece, &cdm_piece)
            output[i] = <double>this_Tk   
        return output
    #end Tk_full
    
    #---------------------------------------------------------------------------
    cpdef np.ndarray Tk_nowiggles(self, np.ndarray k):
        """
        Wrapper function to call TFnowiggles() from tf_fit.c and compute the 
        no-wiggle EH transfer function. 
        """
        cdef float this_Tk
        cdef int N = k.shape[0]
        cdef np.ndarray[double, ndim=1] output = np.empty(N)
        cdef int i

        for i in xrange(N):
            this_Tk = TFnowiggles(self.pdict['omega_m_0'], 
                                  self.pdict['omega_b_0']/self.pdict['omega_m_0'],  
                                  self.pdict['h'], self.pdict['Tcmb_0'], k[i])
            output[i] = <double>this_Tk
        return output
    #end Tk_nowiggles
    #---------------------------------------------------------------------------
#endclass power_eh

        
    
    
