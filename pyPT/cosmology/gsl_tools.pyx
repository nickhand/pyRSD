#!python
#cython: boundscheck=False
#cython: wraparound=False
"""
 gsl_tools.pyx
 cosmology: internal module for functons that need to call GSL functions, 
            usually for numerical integration
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/23/2014
"""
from libc.math cimport pow, cos, sin, M_PI, sqrt
cimport numpy as np
import numpy as np
from cython_gsl cimport *

#-------------------------------------------------------------------------------
cpdef np.ndarray compute_dlnsdlnm_integral(np.ndarray r, object params):
    """
    Internal method for use by hmf module that computes
    the integral for dlnsigma/dlnmass. 

    Notes
    -----
    Uses the GSL CQUAD integrator and calls 'dlnsdlnm_integrand' to 
    evaluate the integral. Should NOT be called by the user.  
    """

    cdef int N = r.shape[0], i
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef gsl_function F
    cdef double fparams[6]
    
    cdef double rmin = np.amin(r)
    cdef double rmax = np.amax(r)

    # set up the integration
    cdef gsl_integration_cquad_workspace * w
    cdef double result, error
    w = gsl_integration_cquad_workspace_alloc(1000)

    F.function = &dlnsdlnm_integrand
    F.params = fparams

    # these are the function parameters in common
    fparams[1] = params.h
    fparams[2] = params.n

    # initialize the transfer function parameters
    TFset_parameters(params.omegam*params.h*params.h, params.omegab/params.omegam, params.Tcmb)

    # do the integration
    for i in xrange(N):
       fparams[0] = r[i]
       gsl_integration_cquad(&F, 1e-3/rmax, 100./rmin, 0., 1e-5, w, &result, &error, NULL)
       output[i] = result

    # free the integration workspace
    gsl_integration_cquad_workspace_free(w)

    return output

#-------------------------------------------------------------------------------
cpdef np.ndarray compute_sigma_integral(np.ndarray r, object params, object tf):
    """
    Internal method for use by linear_growth.mass_variance that computes
    the mass variance at given radii specified by the input array r. 

    Notes
    -----
    Uses the GSL CQUAD integrator and calls 'sigma2_integrand*' to 
    evaluate the integral. Should NOT be called by the user.  
    """
    cdef int N = r.shape[0], i
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    cdef gsl_function F
    cdef double fparams[6]
    
    cdef double rmin = np.amin(r)
    cdef double rmax = np.amax(r)

    # set up the integration
    cdef gsl_integration_cquad_workspace * w
    cdef double result, error
    w = gsl_integration_cquad_workspace_alloc(1000)

    if tf == "full":
       F.function = &sigma2_integrand_full
       F.params = fparams
    elif 'no' in tf and 'wiggle' in tf:
       fparams[3] = params.omegam
       fparams[4] = params.omegab/params.omegam
       fparams[5] = params.Tcmb
   
       F.function = &sigma2_integrand_nw
       F.params = fparams
    else:
       raise ValueError("Do not understand input for 'tf' keyword: %s" %tf)
   
    # these are the function parameters in common
    fparams[1] = params.h
    fparams[2] = params.n

    # initialize the transfer function parameters
    TFset_parameters(params.omegam*params.h*params.h, params.omegab/params.omegam, params.Tcmb)

    # do the integration
    for i in xrange(N):
       fparams[0] = r[i]
       gsl_integration_cquad(&F, 1e-3/rmax, 100./rmin, 0., 1e-5, w, &result, &error, NULL)
       output[i] = result

    # free the integration workspace
    gsl_integration_cquad_workspace_free(w)

    return output

#-------------------------------------------------------------------------------
cpdef np.ndarray compute_growth_integral(np.ndarray z, object params):
    """
    Internal method for use by linear_growth.growth_function that computes
    the integral \int_0^a 1/(a*E(a)^3). 
    
    Notes
    -----
    Uses the GSL CQUAD integrator and calls 'growth_function_integrand' to 
    evaluate the integral. Should NOT be called by the user.  
    """
    # initialize the output array
    cdef int N = z.shape[0], i
    cdef np.ndarray[double, ndim=1] output = np.empty(N)
    
    # set up the gsl function and integration workspace
    cdef gsl_function F
    cdef gsl_integration_cquad_workspace * work
    cdef double result, error
    work = gsl_integration_cquad_workspace_alloc(1000)
    
    # now set up the function params
    cdef double fparams[5]
    fparams[0] = params.omegam
    fparams[1] = params.omegar
    fparams[2] = params.omegak
    fparams[3] = params.omegal
    fparams[4] = params.w
    
    F.function = &growth_function_integrand
    F.params = fparams

    # integrate for every redshift provided
    for i in xrange(N):
        gsl_integration_cquad(&F, 0., 1./(1.+z[i]), 0., 1e-5, work, &result, &error, NULL)
        output[i] = result
    
    # free the integration workspace
    gsl_integration_cquad_workspace_free(work)
    return output
#end compute_growth_integral

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

#-------------------------------------------------------------------------------
cdef double growth_function_integrand(double a, void *params) nogil:
    
    cdef double omegam = (<double*> params)[0]
    cdef double omegar = (<double*> params)[1]
    cdef double omegak = (<double*> params)[2]
    cdef double omegal = (<double*> params)[3]
    cdef double w      = (<double*> params)[4]
    cdef double E
        
    if a == 0.:
        E = 1.
    else:
        E = sqrt(omegam/(a*a*a) + omegar/(a*a*a*a) + omegak/(a*a) + omegal/pow(a, 3.*(1+w)))
        E = 1./(E*E*E*a*a*a)
    return E
#end growth_function_integrand

#-------------------------------------------------------------------------------
cdef double dlnsdlnm_integrand(double k, void *params) nogil:
    
    cdef float baryon_piece, cdm_piece
    cdef double r  = (<double*> params)[0]
    cdef double h  = (<double*> params)[1]
    cdef double ns = (<double*> params)[2]

    cdef double Tk = <double>TFfit_onek(k*h, &baryon_piece, &cdm_piece)
    cdef double dW2dm = (sin(k*r)-k*r*cos(k*r))*(sin(k*r)*(1.-3./(k*k*r*r))+3*cos(k*r)/(k*r))
    
    return dW2dm*(Tk*Tk)*pow(k, ns)/(k*k)
#end dlnsdlm_integrand               
