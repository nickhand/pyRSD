#!python
#cython: boundscheck=False
#cython: wraparound=False
"""
 growth.pyx
 pyPT: module to compute quantities related to the growth of perturbations
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/21/2014
"""
from .cosmo import Cosmology
from pyPT.cosmology cimport cosmo_tools
import numpy as np
cimport numpy as np
    
cpdef growth_function(object z, bint normed=True, object params="Planck1_lens_WP_highL"):
    """
    The linear growth function, defined as:

    .. math::  D(a) = 5/2 \Omega_{m, 0} H(a)/H0 \int_0^a 1/(a*H/H0)^3.

    If ``normed`` = ``True``, the returned value is normalized to unity at z = 0.

    Parameters
    ----------
    z : {float, np.ndarray}
        Redshift(s) to compute the growth function at.
    normed : bool, optional
        If ``True``, normalize to unity at z = 0. Default is ``True``.
    params : {str, dict, cosmo.Cosmology}
        The cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    z = cosmo_tools.vectorize(z)
    
    cdef np.ndarray zarr
    cdef int N = z.shape[0]
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    
    zarr = np.ascontiguousarray(z, dtype=np.double)
    output = np.ascontiguousarray(output, dtype=np.double)
    
    # set the cosmological parameters
    set_parameters(params.omegam, params.omegab, params.omegal, params.omegar, 
                    params.sigma_8, params.h, params.n, params.Tcmb, params.w, -1);
    
    # compute the growth function
    D_plus(<double *>zarr.data, N, int(normed), <double *>output.data)
    return output
#end growth_function

#-------------------------------------------------------------------------------
cpdef growth_rate(object z, object params='Planck1_lens_WP_highL'):
    """
    The growth rate, which is the logarithmic derivative of the growth 
    function with respect to scale factor, denoted by f usually. Fitting formula 
    from eq 5 of Hamilton 2001; originally from Lahav et al. 1991.
    
    Parameters
    ----------
    z : {float, np.ndarray}
        the redshift to compute the function at
    params : {str, dict, cosmo.Cosmology}
        the cosmological parameters to use. Default is Planck 2013 parameters.
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    
    Ez = cosmo_tools.E(z, params=params)
    Dplus = growth_function(z, normed=False, params=params)

    return 0.5*((1+z)/Ez)**2*(5*params.omegam/Dplus - 3*params.omegam*(1+z) - 2*params.omegak)
#end growth_rate

#-------------------------------------------------------------------------------