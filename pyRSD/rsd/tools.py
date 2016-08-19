from .. import pygcl, numpy as np
from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator, InterpolationDomainError

from scipy.integrate import simps
import scipy.interpolate as interp
from scipy.optimize import brentq

import functools
import itertools

#-------------------------------------------------------------------------------
# AP effect
#-------------------------------------------------------------------------------  
def k_AP(k_obs, mu_obs, alpha_perp, alpha_par):
    """
    Return the `true` k values, given an observed (k, mu).
    
    This corresponds to the `k` to evaluate P(k,mu) at when including
    the AP effect
    """
    F = alpha_par / alpha_perp
    if (F != 1.):
        return (k_obs/alpha_perp)*(1 + mu_obs**2*(1./F**2 - 1))**(0.5)
    else:
        return k_obs/alpha_perp
            
def mu_AP(mu_obs, alpha_perp, alpha_par):
    """
    Return the `true` mu values, given an observed mu
    
    This corresponds to the `mu` to evaluate P(k,mu) at when including
    the AP effect
    """
    F = alpha_par / alpha_perp
    return (mu_obs/F) * (1 + mu_obs**2*(1./F**2 - 1))**(-0.5)

#-------------------------------------------------------------------------------
# decorators
#-------------------------------------------------------------------------------      
def doublewrap(f):
    """
    A decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """
    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec
    
def unpacked(method):
    """
    Decorator to avoid return lists/tuples of length 1
    """
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return result if len(result) != 1 else result[0]
        except:
            return result
    return _decorator


def align_input(f):
    """
    Decorator to align input by repeating any scalar entries
    """     
    @functools.wraps(f)
    def wrapper(self, **kw):
        ii = [k for k in kw if not np.isscalar(kw[k])]
        if len(ii):
            if len(ii) > 1:
                if not all(len(kw[k]) == len(kw[ii[0]]) for k in ii):
                    raise ValueError("size mismatch when aligning multiple arrays")
            N = len(kw[ii[0]])
            for k in kw:
                if k not in ii:
                    kw[k] = np.repeat(kw[k], N)
        return f(self, **kw)
        
    return wrapper
    
@doublewrap
def alcock_paczynski(f, alpha_par=None, alpha_perp=None):
    """
    Decorator to introduce the AP effect
    """
    @functools.wraps(f)
    def wrap(self, *args, **kwargs):
        args = list(args)
        
        # if model is AP locked, do the distortion and lock
        if not hasattr(self, '_AP_lock'):
            # determine alpha_par, alpha_perp            
            alpha_par_  = alpha_par if alpha_par is not None else self.alpha_par
            alpha_perp_  = alpha_perp if alpha_perp is not None else self.alpha_perp
        
            # the k,mu to evaluate P(k, mu)
            k = k_AP(args[0], args[1], alpha_perp_, alpha_par_)
            mu = mu_AP(args[1], alpha_perp_, alpha_par_)
        
            # evaluate at the AP (k,mu)
            args[:2] = k, mu
            
            self._AP_lock = True
            pkmu = f(self, *args, **kwargs)
            del self._AP_lock
        
            # do the volume rescaling
            pkmu /= (alpha_perp_**2 * alpha_par_)
                
        # if locked, the distortion was already added
        else:
            pkmu = f(self, *args, **kwargs)
            
        return pkmu

    return wrap
             

def broadcast_kmu(f):
    """
    Decorator to properly handle broadcasting of k, mu. 
    
    This will call the decorator function with the fully
    broadcasted values, such that the input arguments
    have shape (Nk, Nmu)
    
    Notes
    -----
    This assumes the first two arguments of ``f()`` 
    are `k` and `mu`
    """ 
    @functools.wraps(f)    
    def wrapper(self, *args, **kwargs):
        args = list(args)
        k = args[0]
        mu = args[1]
        if isinstance(k, list): k = np.array(k)
        if isinstance(mu, list): mu = np.array(mu)
        if np.isscalar(k): k = np.array([k])
        if np.isscalar(mu): mu = np.array([mu])
        args[0] = k; args[1] = mu
        
        mu_dim = np.ndim(mu); k_dim = np.ndim(k)
        if mu_dim == 1 and k_dim == 1:
            if len(mu) != len(k):
                k = k[:, np.newaxis]
                mu = mu[np.newaxis, :]
        else:
            if k_dim > 1:
                mu =  mu[np.newaxis, :]
            else:
                k = k[:, np.newaxis]
        
        args[:2] = np.broadcast_arrays(k, mu)
        return np.squeeze(f(self, *args, **kwargs))
        
    return wrapper
                
def monopole(f):
    """
    Decorator to compute the monopole from a `self.power` function
    """     
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, k, mus, **kwargs)
        return np.array([simps(pk, x=mus) for pk in Pkmus])
    return wrapper

def quadrupole(f):
    """
    Decorator to compute the quadrupole from a `self.power` function
    """ 
    @functools.wraps(f)    
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, k, mus, **kwargs)
        kern = 2.5*(3*mus**2 - 1.)
        return np.array([simps(kern*pk, x=mus) for pk in Pkmus])
    return wrapper

def hexadecapole(f):
    """
    Decorator to compute the hexadecapole from a `self.power` function
    """  
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, k, mus, **kwargs)
        kern = 9./8.*(35*mus**4 - 30.*mus**2 + 3.)
        return np.array([simps(kern*pk, x=mus) for pk in Pkmus])
    return wrapper

def tetrahexadecapole(f):
    """
    Decorator to compute the tetrahexadecapole from a `self.power` function
    """  
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        k = args[0]
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, k, mus, **kwargs)
        kern = 13./16.*(231*mus**6 - 315*mus**4 + 105*mus**2 - 5)
        return np.array([simps(kern*pk, x=mus) for pk in Pkmus])
    return wrapper
    

#-------------------------------------------------------------------------------
# InterpolatedUnivariateSpline with extrapolation
#-------------------------------------------------------------------------------
class RSDSpline(interp.InterpolatedUnivariateSpline):
    """
    Class to implement an `InterpolatedUnivariateSpline` that remembers 
    the x-domain
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        x : (N,) array_like
            Input dimension of domain points -- must be increasing
        y : (N,) array_like
            input dimension of data points
        bounds_error : bool, optional
            If `True`, raise an exception if the desired input domain value
            is out of the input range. Default is `False`
        fill_value : float, optional
            The fill value to use for any values that are out of bounds 
            on the input domain. Default is `0`
        extrap : bool, optional
            If desired domain value is out of bounds, do a linear extrapolation.
            Default is `False`
        """
        
        # default kwargs
        self.bounds_error = kwargs.pop('bounds_error', False)
        self.fill_value   = kwargs.pop('fill_value', 0.)
        self.extrap       = kwargs.pop('extrap', False)
        
        self.x = args[0]
        self.y = args[1]
        super(RSDSpline, self).__init__(*args, **kwargs)

    def _check_bounds(self, x_new):
        """
        Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """
        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                                "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                                "range.")

        out_of_bounds = np.logical_or(below_bounds, above_bounds)
        return out_of_bounds
            
    def __call__(self, x_new):
        """
        Return the interpolated value
        """
        if self.extrap: 
            if np.isscalar(x_new):
                return self.linear_extrap(x_new)*1.
            else:
                return np.array([self.linear_extrap(x) for x in x_new])
        else:
            return self._evaluate_spline(x_new)*1.
       
    def _evaluate_spline(self, x_new):
        """
        Evaluate the spline
        """
        out_of_bounds = self._check_bounds(x_new)
        y_new = interp.InterpolatedUnivariateSpline.__call__(self, x_new)
        if np.isscalar(y_new) or y_new.ndim == 0:
            return self.fill_value if out_of_bounds else y_new
        else:
            y_new[out_of_bounds] = self.fill_value
            return y_new
    
    def linear_extrap(self, x):
        """
        Do a linear extrapolation
        """
        if x < self.x[0]:
            return self.y[0] + (x-self.x[0])*(self.y[1]-self.y[0])/(self.x[1]-self.x[0])
        elif x > self.x[-1]:
            return self.y[-1] + (x-self.x[-1])*(self.y[-1]-self.y[-2])/(self.x[-1]-self.x[-2])
        else:
            return self._evaluate_spline(x)
    
    
#-------------------------------------------------------------------------------
# bias to mass relation
#-------------------------------------------------------------------------------
class BiasToMassRelation(Cache):
    """
    Class to handle conversions between mass (in M_sun/h) and bias quickly 
    using an interpolation table
    """
    # define the interpolation grid
    interpolation_grid = {}
    interpolation_grid['sigma8_z'] = np.linspace(0.3, 1.0, 100)
    interpolation_grid['b1'] = np.linspace(0.9, 8., 70)
    
    def __init__(self, z, cosmo, interpolate=False):
        """
        Parameters
        ----------
        z : float
            The redshift to compute the relation at
        cosmo : pygcl.Cosmology
            The cosmology object
        interpolate : bool, optional
            Whether to return results from an interpolation table
        """        
        # save the parameters
        self.z = z
        self.cosmo = cosmo
        self.interpolate = interpolate
        self.delta_halo = 200
        
    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    @parameter
    def interpolate(self, val):
        """
        If `True`, return the Zel'dovich power term from an interpolation table
        """
        return val
        
    @parameter
    def z(self, val):
        """
        The redshift
        """
        return val
        
    @parameter
    def cosmo(self, val):
        """
        The cosmology of the input linear power spectrum
        """
        return val
        
    @parameter
    def delta_halo(self, val):
        return val
            
            
    @cached_property("cosmo")
    def power_lin(self):
        """
        The `pygcl.LinearPS` object defining the linear power spectrum at `z=0`
        """
        return pygcl.LinearPS(self.cosmo, 0.)
        
    #---------------------------------------------------------------------------
    @cached_property()
    def interpolation_table(self):
        """
        Evaluate the bias to mass relation at the interpolation grid points
        """
        # setup a few functions we need
        mean_dens = self.cosmo.rho_bar_z(0.)
        mass_norm = 1e13
        mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
        s8_0 = self.cosmo.sigma8()
        
        # setup the sigma spline
        R_interp = np.logspace(-3, 4, 1000)
        sigmas = self.power_lin.Sigma(R_interp)
        sigma_spline = RSDSpline(R_interp, sigmas, bounds_error=True)
        
        # the objective function to minimize
        def objective(mass, bias, rescaling):
            sigma = rescaling*sigma_spline(mass_to_radius(mass))
            return bias_Tinker(sigma) - bias
        
        # setup the grid points
        sigma8s = self.interpolation_grid['sigma8_z']
        b1s = self.interpolation_grid['b1']
        pts = np.asarray(list(itertools.product(sigma8s, b1s)))

        grid_vals = []
        for (s8_z, b1) in pts:
            
            # get appropriate rescalings
            rescaling = (s8_z / s8_0)
            
            # find the zero
            try:
                M = brentq(objective, 1e-10, 1e4, args=(b1, rescaling))*mass_norm
            except Exception as e:
                print(s8, b1)
                print(objective(1e-10, b1, rescaling), objective(1e4, b1, rescaling))
                raise Exception(e)
                
            grid_vals.append(M)
        grid_vals = np.array(grid_vals).reshape((len(sigma8s), len(b1s)))
        
        # return the interpolator
        return RegularGridInterpolator((sigma8s, b1s), grid_vals)
        
    @unpacked
    def __call__(self, sigma8_z, b1):
        """
        Return the mass [units: `M_sun/h`] associated with the desired 
        `b1` and `sigma8` values
        
        Parameters
        ----------
        sigma8_z : float
            The sigma8 value
        b1 : float
            The linear bias
        """     
        def compute_fresh(sigma8_z, b1):
            # setup a few functions we need
            mean_dens = self.cosmo.rho_bar_z(0.)
            mass_norm = 1e13
            mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
            rescaling = sigma8_z / self.cosmo.sigma8()
            
            # the objective function to minimize
            def objective(mass):
                sigma = rescaling*self.power_lin.Sigma(mass_to_radius(mass))
                return bias_Tinker(sigma, delta_halo=self.delta_halo) - b1

            return brentq(objective, 1e-8, 1e3)*mass_norm
               
        if self.interpolate:
            try:
                return self.interpolation_table([sigma8_z, b1])
            except InterpolationDomainError:
                return compute_fresh(sigma8_z, b1)
        else:
            return compute_fresh(sigma8_z, b1)

            
#-------------------------------------------------------------------------------
# bias to sigma relation
#-------------------------------------------------------------------------------
class BiasToSigmaRelation(BiasToMassRelation):
    """
    Class to represent the relation between velocity dispersion and halo bias
    """
    def __init__(self, z, cosmo, interpolate=False, sigmav_0=None, M_0=None):
        """
        Parameters
        ----------
        z : float
            The redshift to compute the relation at
        cosmo : pygcl.Cosmology
            The cosmology object
        interpolate : bool, optional
            Whether to return results from an interpolation table
        """
        # initialize the base class
        super(BiasToSigmaRelation, self).__init__(z, cosmo, interpolate=interpolate)
        
        # store the normalizations
        self.sigmav_0 = sigmav_0
        self.M_0 = M_0
        
    @parameter
    def sigmav_0(self, val):
        """
        The velocity normalization of the sigma to mass relation
        """
        return val
        
    @parameter
    def M_0(self, val):
        """
        The mass normalization of the sigma to mass relation
        """
        return val
    
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("z", "cosmo")
    def Hz(self):
        """
        The Hubble parameter at `z`
        """
        return self.cosmo.H_z(self.z)
        
    @cached_property("z", "cosmo")
    def Ez(self):
        """
        The normalized Hubble parameter at `z`
        """
        return self.cosmo.H_z(self.z) / self.cosmo.H0()
        
    def evrard_sigmav(self, mass):
        """
        Return the line-of-sight velocity dispersion for dark matter using
        the relationship found between velocity dispersion and halo mass found
        by Evrard et al. 2008:
        
        \sigma_DM(M, z) = (1082.9 km/s) * (E(z) * M / 1e15 M_sun/h)**0.33
        
        Parameters
        ----------
        mass : float
            The halo mass in units of `M_sun/h`
        """
        # model params
        sigmav_0 = 1082.9 # in km/s
        M_0      = 1e15 # in M_sun/h
        alpha    = 0.336
        
        # the Evrard value
        sigmav = sigmav_0*(self.Ez * mass/M_0)**alpha
        
        # LOS is sigmav / sqrt(3) due to spherical symmetry
        sigmav /= 3**0.5
        
        # put it into units of Mpc/h
        return sigmav * (1 + self.z)/self.Hz
    
    @unpacked
    def mass(self, sigma8_z, b1):
        """
        Return the mass at this bias and sigma8 value
        """
        return BiasToMassRelation.__call__(self, sigma8_z, b1)
        
    @unpacked
    def __call__(self, sigma8_z, b1):
        """
        Return the velocity dispersion [units: `Mpc/h`] associated with the 
        desired `b1` and `sigma8` values
        
        Parameters
        ----------
        sigma8 : float
            The sigma8 value
        b1 : float
            The linear bias
        """
        # first get the halo mass
        M = self.mass(sigma8_z, b1)
    
        # evrard relation
        if self.sigmav_0 is None or self.M_0 is None:
            return self.evrard_sigmav(M)
        else:
            return self.sigmav_0 * (M / self.M_0)**(1./3.)
        
   
#-------------------------------------------------------------------------------
# bias(M) tools
#-------------------------------------------------------------------------------         
def mass_from_bias(bias, z, linearPS):
    """
    Given an input bias, return the corresponding mass, using Tinker et al.
    bias fits
    
    Notes
    -----
    This returns the halo mass in units of M_sun / h
    """
    # convenience normalization for finding the zero crossing
    mass_norm = 1e13
    
    # critical density in units of h^2 M_sun / Mpc^3
    kms_Mpc = pygcl.Constants.km/pygcl.Constants.second/pygcl.Constants.Mpc
    crit_dens = 3*(pygcl.Constants.H_0*kms_Mpc)**2 / (8*np.pi*pygcl.Constants.G) 
    
    unit_conversion = (pygcl.Constants.M_sun/pygcl.Constants.Mpc**3)
    crit_dens /= unit_conversion
    
    # mean density at z = 0
    cosmo = linearPS.GetCosmology()
    mean_dens = crit_dens * cosmo.Omega0_m()
    
    # convert mass to radius
    mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
    
    # growth factor at this z
    z_Plin = linearPS.GetRedshift()
    Dz = 1.
    if z_Plin != z: 
        Dz = (cosmo.D_z(z) / cosmo.D_z(z_Plin))

    def objective(mass):
        sigma = Dz*linearPS.Sigma(mass_to_radius(mass))
        return bias_Tinker(sigma) - bias
        
    return brentq(objective, 1e-5, 1e5)*mass_norm

def bias_Tinker(sigmas, delta_c=1.686, delta_halo=200):
    """
    Return the halo bias for the Tinker form.
    
    Tinker, J., et al., 2010. ApJ 724, 878-886.
    http://iopscience.iop.org/0004-637X/724/2/878
    """
    y = np.log10(delta_halo)
    
    # get the parameters as a function of halo overdensity
    A = 1. + 0.24*y*np.exp(-(4./y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4)
    c = 2.4
    
    nu = delta_c / sigmas
    return 1. - A * (nu**a)/(nu**a + delta_c**a) + B*nu**b + C*nu**c

def sigma_from_bias(bias, z, linearPS):
    """
    Return sigma from bias
    """
    # normalized Teppei's sims at z = 0.509 and sigma_sAsA = 3.6
    sigma0 = 3.6 # in Mpc/h
    M0 = 5.4903e13 # in M_sun / h
    return sigma0 * (mass_from_bias(bias, z, linearPS) / M0)**(1./3)
    

#-------------------------------------------------------------------------------
# window convolution tools
#-------------------------------------------------------------------------------     
def window_convolved_xi(xi_data, W):
    """
    Compute the window-convolved configuration space multipoles, from
    the ell = 0, 2, 4 (,6) unconvolved multipoles and the window
    
    Parameters
    ----------
    xi_data : array_like, (Ns, 3 or 4)
        the unconvolved configuration space multipoles, with the three columns
        being the ell = 0, 2, 4 (,6) multipoles, respectively
    W : array_like, (Ns, Nl)
        the even-ell configuration space window function multipoles, 
        where Nl must be >= 5; the first column is the ell=0, second 
        is ell=2, etc
    """
    shape = xi_data.shape
    Nell = shape[-1]
    if Nell == 3:
        ells = [0, 2, 4]
    elif Nell == 4:
        ells = [0, 2, 4, 6]
        have_window_ell10 = W.shape[1] == 6
    else:
        raise ValueError("`xi_data` should have 3 or 4 columns in `window_convolved_xi`")
    
    if shape[0] != W.shape[0]:
        raise ValueError("shape mismatch in first dimension between `xi_data` and window")
    
    # loop over each ell 
    # each column in xi_data is a value in ells
    toret = np.empty(shape)
    kern = np.empty(shape)
    
    for i, ell in enumerate(ells):
        
        # get the kernel for this ell
        if ell == 0:    
            kern[:,:3] = W[:,:3] * np.array([1., 1./5, 1./9])
            if Nell == 4: # tetraxhex
                kern[:,3] = W[:,3] * 1./13. 
            
        elif ell == 2:
            kern[:,0] = W[:,1]
            kern[:,1] = np.einsum('...i,i...', W[:,:3], np.array([1., 2./7, 2./7]))
            kern[:,2] = np.einsum('...i,i...', W[:,1:4], np.array([2./7, 100./693, 25./143]))
            if Nell == 4: # tetraxhex
                kern[:,3] = np.einsum('...i,i...', W[:,2:5], np.array([25./143, 14./143., 28./221.]))
        
        elif ell == 4:
            kern[:,0] = W[:,2]
            kern[:,1] = np.einsum('...i,i...', W[:,1:4], np.array([18./35, 20./77, 45./143]))
            kern[:,2] = np.einsum('...i,i...', W[:,:5], np.array([1., 20./77, 162./1001, 20./143, 490./2431]))
            
            # tetraxhex
            if Nell == 4: 
                # include ell = 10 window
                if not have_window_ell10:
                    k =  np.array([45./143., 20./143., 252./2431., 4536./46189.])
                else:
                    k =  np.array([45./143., 20./143., 252./2431., 4536./46189., 630./4199.])
                kern[:,3] = np.einsum('...i,i...', W[:,1:], k)
                    
        toret[:,i] = np.einsum('ij,ij->i', xi_data, kern)
        
    return toret
    
    
def convolve_multipoles(k, Pell, window, k_out=None, interpolate=True, 
                        pk_smooth=0., xi_smooth=0., method=pygcl.IntegrationMethods.TRAPZ):
    """
    Convolve the input ell = 0, 2, 4 power multipoles, specified by `Pell`,
    with the specified window function.
    
    Parameters
    ----------
    k : array_like, (Nk,)
        the array of wavenumbers where `Pell` is defined -- to avoid convolution
        errors, `k` should probably extend to higher values than the desired `k_out`
    Pell : array_like, (Nk, 3)
        the ell = 0, 2, 4 power multipoles, defined at `k`
    window : array_like, (Ns, ...)
        the window function to convolve with, where the first column is `s` and 
        the other columns are the even `ell` configuration space window 
        multipoles
    k_out : array_like
        the array of desired output k values
    """
    shape = Pell.shape
    Nell = shape[-1]
    if Nell == 3:
        ells = [0, 2, 4]
    elif Nell == 4:
        ells = [0, 2, 4, 6]
    else:
        raise ValueError("please provide 3 or 4 input power multipoles as columns")
    
    # separation is the first window column
    s = window[:,0]
    
    # format the k_out
    if k_out is None: k_out = k
    if np.ndim(k_out) == 1:
        k_out = np.repeat(k_out[:,None], Nell, axis=1)
    if k_out.shape[-1] != len(ells):
        raise ValueError("input `k_out` must have %d columns for ell=%s multipoles" %(Nell, str(ells)))
    
    # make the hires version to avoid wiggles when convolving
    if interpolate and len(k) < 500:
        k_hires = np.logspace(np.log10(k.min()), np.log10(k.max()), 500)
        poles_hires = []
        for i in range(Nell):
            tck = interp.splrep(k, Pell[:,i], k=3, s=0)
            poles_hires.append(interp.splev(k_hires, tck))
        Pell = np.vstack(poles_hires).T
        k = k_hires.copy()
        
    # FT the power multipoles    
    xi = np.empty((len(s), Nell))
    for i, ell in enumerate(ells): 
        xi[:,i] = pygcl.pk_to_xi(ell, k, Pell[:,i], s, smoothing=pk_smooth, method=method)
    
    # convolve the config space multipole
    xi_conv = window_convolved_xi(xi, window[:,1:])
    
    # FT back to get convolved power pole
    toret = np.empty((len(k_out), Nell))
    for i, ell in enumerate(ells):
        toret[:,i] = pygcl.xi_to_pk(ell, s, xi_conv[:,i], k_out[:,i], smoothing=xi_smooth, method=method)
    
    return toret
 

    
