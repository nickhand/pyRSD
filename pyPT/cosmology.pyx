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
from cython_gsl cimport *
from libc.math cimport pow, cos, sin, M_PI

#-------------------------------------------------------------------------------
def _vectorize(x):
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.array(x)
    return x
#end _vectorize

#-------------------------------------------------------------------------------
cpdef H(object z, object cosmo='Planck13'):
    """
    The value of the Hubble constant at redshift z in km/s/Mpc
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
    return 100.*cosmo['h']*E(z, cosmo)
#end H

#-------------------------------------------------------------------------------
cpdef E(object z, object cosmo='Planck13'):
    """
    The unitless Hubble expansion rate at redshift z, 
    modified to include non-constant w parameterized linearly 
    with z ( w = w0 + w1*z )

    Parameters
    ----------
    z : {float, np.ndarray}
    the redshift to compute the function at
    cosmo : {str, dict, cosmo_dict.params}
    the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    z = _vectorize(z)
    if not isinstance(cosmo, cosmo_dict.params):
     cosmo = cosmo_dict.params(cosmo)

    return np.sqrt(cosmo['omega_m_0']*(1.+z)**3 + cosmo['omega_r_0']*(1.+z)**4 \
                    + cosmo['omega_k_0']*(1.+z)**2 \
                    + cosmo['omega_l_0']*np.exp(3.*cosmo['w1']*z) \
                    *(1.+ z)**(3.*(1+cosmo['w0']-cosmo['w1'])))
#end E

#-------------------------------------------------------------------------------
cpdef omega_m_z(object z, object cosmo='Planck13'):
    """
    The matter density omega_m as a function of redshift

    From Lahav et al. 1991 equations 11b-c. This is equivalent to 
    equation 10 of Eisenstein & Hu 1999.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.   
    """
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
    return cosmo['omega_m_0']*(1.+z)**3/E(z, cosmo)**2.
#end omega_m_z

#-------------------------------------------------------------------------------
cpdef omega_l_z(object z, object cosmo='Planck13'):
    """
    The dark energy density omega_l as a function of redshift
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)    
    return cosmo['omega_l_0']/E(z, cosmo)**2
#end omega_l_z
 
#-------------------------------------------------------------------------------
cpdef growth_rate(object z, object cosmo='Planck13'):
    """
    The growth rate, which is the logarithmic derivative of the growth 
    factor with respect to scale factor, denoted by f usually. Fitting formula 
    from eq 5 of Hamilton 2001; originally from Lahav et al. 1991.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
    
    om_m = omega_m_z(z, cosmo)
    om_l = omega_l_z(z, cosmo)

    return om_m**(4./7) + (1 + 0.5*om_m)*om_l/70.
#end growth_rate

#---------------------------------------------------------------------------
cpdef growth_factor(object z, object cosmo="Planck13"):
    """
    The linear growth factor, using approximation
    from Carol, Press, & Turner (1992), else integrate the ODE.
    Normalized to 1 at z = 0.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
        
    om_m = omega_m_z(z, cosmo)
    om_l = omega_l_z(z, cosmo)

    om_m_0 = cosmo['omega_m_0']
    om_l_0 = cosmo['omega_l_0']
    norm = 2.5*om_m_0/(om_m_0**(4./7.) - om_l_0 + (1.+0.5*om_m_0)*(1.+om_l_0/70.))
    return 2.5*om_m/(om_m**(4./7.)-om_l+(1.+0.5*om_m)*(1.+om_l/70.))/(norm*(1+z))
#end growth_factor
 
#-------------------------------------------------------------------------------
cpdef Pk_full(object k_hMpc, object z, object cosmo='Planck13'):
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
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.     
    
    Returns
    -------
    P_k : numpy.ndarray
        full, linear power spectrum in units of Mpc^3/h^3
    """
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
    k = _vectorize(k_hMpc)
    
    # the power spectrum normalization
    norm  = (sigma_r(8., 0., normed=False, tf="full", cosmo=cosmo)/cosmo['sigma_8'])**2
    
    Tk = _Tk_full(k, cosmo['h'])
    D = growth_factor(z, cosmo)
    return k**cosmo['n_s'] * (Tk*D)**2 / norm
#end Pk_full

#---------------------------------------------------------------------------
cpdef Pk_nowiggles(object k_hMpc, object z, object cosmo='Planck13'):
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
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
        
    k = _vectorize(k_hMpc)

    # the power spectrum normalization
    norm  = (sigma_r(8., 0., normed=False, tf="no-wiggle", cosmo=cosmo)/cosmo['sigma_8'])**2
    
    Tk = _Tk_nowiggles(k, cosmo)
    D = growth_factor(z)
    return k**cosmo['n_s'] * (Tk*D)**2 / norm
#end Pk_nowiggles

#---------------------------------------------------------------------------
cpdef np.ndarray _Tk_full(np.ndarray k, double h):
    """
    Wrapper function to call TFfit_onek() from tf_fit.c and compute the 
    full EH transfer function. 
    """
    cdef float baryon_piece, cdm_piece, this_Tk
    cdef int N = k.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef int i
    
    for i in xrange(N):
        this_Tk = TFfit_onek(k[i]*h, &baryon_piece, &cdm_piece)
        output[i] = <double>this_Tk   
    return output
#end _Tk_full

#---------------------------------------------------------------------------
cpdef np.ndarray _Tk_nowiggles(np.ndarray k, object cosmo="Planck13"):
    """
    Wrapper function to call TFnowiggles() from tf_fit.c and compute the 
    no-wiggle EH transfer function. 
    """
    cdef float this_Tk
    cdef int N = k.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef int i

    for i in xrange(N):
        this_Tk = TFnowiggles(cosmo['omega_m_0'], 
                              cosmo['omega_b_0']/cosmo['omega_m_0'],  
                              cosmo['h'], cosmo['Tcmb_0'], k[i])
        output[i] = <double>this_Tk
    return output
#end _Tk_nowiggles
#-------------------------------------------------------------------------------

cpdef sigma_r(object r_Mpch, object z, bint normed=True, 
            object tf="full", object cosmo="Planck13"):
    """
    The average mass fluctuation within a sphere of radius r, using the 
    CDM + baryon linear power spectrum and the Eisenstein + Hu transfer function,
    which is smoothed by a top hat function of the input radis
    
    Parameters
    ----------
    r_Mpch : {float, np.ndarray}
        the radius to compute the variance within in units of Mpc/h
    z : {float, np.ndarray}
        the redshift to compute the variance at
    normed : bool, optional
        whether to normalize to the present value of sigma_8, the mass fluctuation
        within a sphere of radius 8 Mpc/h. Default is True
    tf : str, optional
        the transfer to use, either "full" or "no-wiggle". Default is "full"
    cosmo : {str, dict, cosmo_dict.params}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    # check that both r_Mpc or z are not arrays
    if not np.isscalar(r_Mpch) and not np.isscalar(z):
        raise ValueError("Radius and redshift inputs cannot both be arrays")
    
    # make cosmo parameter dictionary
    if not isinstance(cosmo, cosmo_dict.params):
        cosmo = cosmo_dict.params(cosmo)
    
    # vectorize the radius input
    r = _vectorize(r_Mpch)
    
    cdef int N = r.shape[0], i
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef object D
    cdef gsl_function F
    cdef double params[6]
    
    # set up the integration
    cdef gsl_integration_cquad_workspace * w
    cdef double result, error
    w = gsl_integration_cquad_workspace_alloc(1000)
    
    if tf == "full":
        F.function = &sigma2_integrand_full
        F.params = params
    elif tf == "no-wiggle":
        params[3] = cosmo['omega_m_0']
        params[4] = cosmo['omega_b_0']/cosmo['omega_m_0']
        params[5] = cosmo['Tcmb_0']
        
        F.function = &sigma2_integrand_nw
        F.params = params
    else:
        raise ValueError("Do not understand input for 'tf' keyword: %s" %tf)
        
    # these are the function parameters in common
    params[1] = cosmo['h']   
    params[2] = cosmo['n_s'] 

    # initialize the transfer function parameters
    TFset_parameters(cosmo['omega_m_0']*cosmo['h']*cosmo['h'], 
                     cosmo['omega_b_0']/cosmo['omega_m_0'], cosmo['Tcmb_0'])
    
    # do the integration
    for i in xrange(N):
        params[0] = r[i]
        gsl_integration_cquad(&F, 1e-5, 100., 0., 1e-5, w, &result, &error, NULL)
        output[i] = result
    
    # free the integration workspace
    gsl_integration_cquad_workspace_free(w)
    
    # return sigma with the proper normalization
    D = growth_factor(z, cosmo=cosmo)
    norm = 1.
    if normed: 
        norm = sigma_r(8., 0., normed=False, cosmo=cosmo)/cosmo['sigma_8']
    return np.sqrt(output*D*D)/norm
#end sigma_r
    
#-------------------------------------------------------------------------------
cdef double sigma2_integrand_full(double k, void *params) nogil:
    """
    The integrand of the sigma squared integral, which is equal to 
    k^2 W(kr)^2 T(k)^2 k^n_s, where the transfer function used is the full
    CDM + baryon Eisenstein + Hu fit (from tf_fit.c).
    """
    cdef float baryon_piece, cdm_piece
    cdef double r  = (<double*> params)[0]
    cdef double h  = (<double*> params)[1]
    cdef double ns = (<double*> params)[2]

    cdef double Tk = <double>TFfit_onek(k*h, &baryon_piece, &cdm_piece)
    cdef double W  = 3.*(sin(k*r)-k*r*cos(k*r))/(k*k*k*r*r*r)
    
    return (k*k*W*W)*(Tk*Tk)*pow(k, ns) / 2. / (M_PI*M_PI)
#end sigma2_integrand_full

#-------------------------------------------------------------------------------
cdef double sigma2_integrand_nw(double k, void *params) nogil:
    """
    The integrand of the sigma squared integral, which is equal to 
    k^2 W(kr)^2 T(k)^2 k^n_s, where the transfer function used is the 
    "no-wiggle" Eisenstein + Hu fit (from tf_fit.c).
    """
    cdef float baryon_piece, cdm_piece
    cdef double r    = (<double*> params)[0]
    cdef double h    = (<double*> params)[1]
    cdef double ns   = (<double*> params)[2]
    cdef double om_m = (<double*> params)[3]
    cdef double fb   = (<double*> params)[4]
    cdef double Tcmb = (<double*> params)[5]

    cdef double Tk = <double>TFnowiggles(om_m, fb, h, Tcmb, k)
    cdef double W  = 3.*(sin(k*r)-k*r*cos(k*r))/(k*k*k*r*r*r)
    
    return (k*k*W*W)*(Tk*Tk)*pow(k, ns) / 2. / (M_PI*M_PI)
#end sigma2_integrand_nw 
    
