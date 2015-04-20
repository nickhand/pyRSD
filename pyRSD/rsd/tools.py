from .. import pygcl, numpy as np

from scipy.lib._version import NumpyVersion
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.integrate import simps
from scipy.optimize import brentq

import warnings
import functools
import itertools

NUMPY_LT_160 = NumpyVersion(np.__version__) < '1.6.0'
warnings.filterwarnings("ignore", category=DeprecationWarning,module="scipy")

#-------------------------------------------------------------------------------
# DECORATORS
#-------------------------------------------------------------------------------  
def unpacked(method):
    """
    Decorator to avoid return lists/tuples of length 1
    """
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        return result if len(result) != 1 else result[0]
    return _decorator

#-------------------------------------------------------------------------------
def monopole(f):
    """
    Decorator to compute the monopole from a `self.power` function
    """     
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, mus, **kwargs)
        return np.array([simps(Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper

#-------------------------------------------------------------------------------
def quadrupole(f):
    """
    Decorator to compute the quadrupole from a `self.power` function
    """     
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, mus, **kwargs)
        kern = 2.5*(3*mus**2 - 1.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper

#-------------------------------------------------------------------------------
def hexadecapole(f):
    """
    Decorator to compute the hexadecapole from a `self.power` function
    """  
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, mus, **kwargs)
        kern = 9./8.*(35*mus**4 - 30.*mus**2 + 3.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper
    
#-------------------------------------------------------------------------------
class InterpolationTable(object):
    """
    A helper class for doing linear interpolation on a regular grid using
    scipy.interpolate.RegularGridInterpolator
    """
    def __init__(self, names, *args, **kwargs):
        """
        Parameters
        ----------
        names : list
            A list of strings giving the names of the dimensions
        args: 
            A series of arrays holding the interpolation domain in each
            dimension matching `names`
            
        kwargs:
            interpolated : bool, optional
                If `True`, make the interpolation value and return the value
                when __call__() is called
                
            bounds_error : bool, optional
                If `True`, when interpolated values are requested outside of the
                domain of the input data, a ValueError is raised.
                If `False`, then `fill_value` is used.
                
            fill_value : number, optional
                If provided, the value to use for points outside of the
                interpolation domain. If None, values outside
                the domain are extrapolated.
        """
        if len(names) != len(args):
            raise ValueError("Mismatch between names of interpolation dimensions "
                                "and provided values")
        # interpolation domain
        self._names = names
        self._pts = {}
        for index, name in zip(args, names):
            self._pts[name] = index
        
        # kwargs defaults
        self._bounds_error = kwargs.get('bounds_error', True)
        self._fill_value   = kwargs.get('fill_value', np.nan)
        self.interpolated  = kwargs.get('interpolated', True)
            
    #---------------------------------------------------------------------------
    def make_interpolation_table(self):
        """
        Make the interpolation table
        """
        # make the pts array
        pts_tuple = tuple(self._pts[k] for k in self._names)
        pts = np.asarray(list(itertools.product(*pts_tuple)))
        
        # compute the grid values
        shapes = tuple(len(self._pts[k]) for k in self._names)
        grid_vals = self.evaluate_table(pts).reshape(shapes)
        
        # now save the table
        kwargs = {k:getattr(self, '_'+k) for k in ['bounds_error', 'fill_value']}
        self.table = RegularGridInterpolator(pts_tuple, grid_vals, **kwargs)

    #---------------------------------------------------------------------------
    @property
    def interpolated(self):
        """
        If `True`, return results from the interpolation table
        """
        return self._interpolated

    @interpolated.setter
    def interpolated(self, val):
        # don't do anything if we are setting the same value
        if hasattr(self, '_interpolated') and self._interpolated == val:
            return

        self._interpolated = val
        if self._interpolated:
            self.make_interpolation_table()
    
    #---------------------------------------------------------------------------
    def _stack_pts_arrays(self, **kwargs):
        """
        Stack the points arrays
        """
        max_shape = max(len(np.array(kwargs[k], ndmin=1)) for k in self._names)
        pts = []        
        for k in self._names:
            x = np.array(kwargs[k], ndmin=1)
            if len(x) == max_shape:
                pts.append(x)
            else:
                pts.append(np.repeat(x, max_shape))
        return np.vstack(pts).T
        
    #---------------------------------------------------------------------------
    def __call__(self, **kwargs):
        """
        Interpolation at coordinates

        Parameters
        ----------
        kwargs : 
            Keyword arguments passed should be the names of each interpolation
            dimension and associated value
        """
        return self.table(self._stack_pts_arrays(**kwargs))
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class RSDSpline(InterpolatedUnivariateSpline):
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

    #----------------------------------------------------------------------------
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
            
    #---------------------------------------------------------------------------
    def __call__(self, x_new):
        """
        Return the interpolated value
        """
        if self.extrap: 
            return self.linear_extrap(x_new)*1.
        else:
            return self._evaluate_spline(x_new)*1.
       
    #--------------------------------------------------------------------------- 
    def _evaluate_spline(self, x_new):
        """
        Evaluate the spline
        """
        out_of_bounds = self._check_bounds(x_new)
        y_new = InterpolatedUnivariateSpline.__call__(self, x_new)
        if np.isscalar(y_new) or y_new.ndim == 0:
            return self.fill_value if out_of_bounds else y_new
        else:
            y_new[out_of_bounds] = self.fill_value
            return y_new
    
    #---------------------------------------------------------------------------
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
    
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
# mass to bias relation
#-------------------------------------------------------------------------------
class BiasToMassRelation(InterpolationTable):
    """
    Class to handle conversions between mass and bias quickly using an 
    interpolation table
    """
    z_interp  = np.linspace(0, 1., 30) 
    s8_interp = np.linspace(0.5, 1.5, 30)
    b1_interp = np.linspace(1.1, 6., 50)
    
    def __init__(self, cosmo, interpolated=False):
        
        # save the cosmo
        self.cosmo = cosmo
        self.Plin = pygcl.LinearPS(cosmo, 0.)
        
        # setup the interpolation table
        names = ['z', 'sigma8', 'b1']
        args = [self.z_interp, self.s8_interp, self.b1_interp]
        super(BiasToMassRelation, self).__init__(names, *args, interpolated=interpolated)

    #---------------------------------------------------------------------------
    def evaluate_table(self, pts):
        """
        Evaluate the desired function at the input grid points
        """
        # setup a few functions we need
        mean_dens = self.cosmo.rho_bar_z(0.)
        mass_norm = 1e13
        mass_to_radius = lambda M: (3.*M*mass_norm/(4.*np.pi*mean_dens))**(1./3.)
        s8_0 = self.cosmo.sigma8()
        
        # setup the sigma spline
        R_interp = np.logspace(-3, 4, 1000)
        sigmas = self.Plin.Sigma(R_interp)
        sigma_spline = RSDSpline(R_interp, sigmas, bounds_error=True)
        
        # the objective function to minimize
        def objective(mass, bias, rescaling):
            sigma = rescaling*sigma_spline(mass_to_radius(mass))
            return bias_Tinker(sigma) - bias

        ans = []
        for (z, s8, b1) in pts:
            # get appropriate rescalings
            Dz_ratio = self.cosmo.D_z(z) / 1.0 # since Plin is at z = 0 already
            s8_ratio = s8 / s8_0
            rescaling = Dz_ratio*s8_ratio
            
            # find the zero
            M = brentq(objective, 1e-8, 1e3, args=(b1, rescaling))*mass_norm
            ans.append(M)
        
        return np.array(ans)
    #---------------------------------------------------------------------------
    @unpacked
    def __call__(self, **kwargs):
        """
        Return the mass associated with the input sigma8, b1, and z
        """        
        if self.interpolated:
            return InterpolationTable.__call__(self, **kwargs)
        else:
            pts = self._stack_pts_arrays(**kwargs)
            return self.evaluate_table(pts)
    #---------------------------------------------------------------------------
        
            
#-------------------------------------------------------------------------------
# Sigma-Bias relation
#-------------------------------------------------------------------------------
class SigmaBiasRelation(object):
    """
    Class to represent the relation between velocity dispersion and halo bias
    """
    def __init__(self, z, linearPS):
        """
        Initialize and setup the splines
        """
        self.z = z
        self.power_lin = linearPS
        self._initialize_splines()
        
    #---------------------------------------------------------------------------
    def _initialize_splines(self):
        """
        Initialize the splines we need
        """
        biases = np.linspace(1.0, 7.0, 100)
        sigmas = np.array([sigma_from_bias(bias, self.z, self.power_lin) for bias in biases])
        
        self.sigma_to_bias_spline = RSDSpline(sigmas, biases, extrap=True)
        self.bias_to_sigma_spline = RSDSpline(biases, sigmas, extrap=True)
        
    #---------------------------------------------------------------------------
    def bias(self, sigma):
        """
        Return the linear bias for the input sigma in Mpc/h
        """
        return self.sigma_to_bias_spline(sigma)
    
    #-------------------------------------------------------------------------------
    def sigma(self, bias):
        """
        Return the sigma in Mpc/h for the input linear bias value
        """
        return self.bias_to_sigma_spline(bias)
    
    #-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------        
def mass_from_bias(bias, z, linearPS):
    """
    Given an input bias, return the corresponding mass, using Tinker et al.
    bias fits
    """
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

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
def sigma_from_bias(bias, z, linearPS):
    """
    Return sigma from bias
    """
    # normalized Teppei's sims at z = 0.509 and sigma_sAsA = 3.6
    sigma0 = 3.6 # in Mpc/h
    M0 = 5.4903e13 # in M_sun / h
    return sigma0 * (mass_from_bias(bias, z, linearPS) / M0)**(1./3)
 
#-------------------------------------------------------------------------------
# Regular grid interpolator from scipy
#-------------------------------------------------------------------------------

class RegularGridInterpolator(object):
    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    .. versionadded:: 0.14

    See also
    --------
    NearestNDInterpolator : Nearest neighbour interpolation on unstructured
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Trilinear interpolation. (2013, January 17). In Wikipedia, The Free
           Encyclopedia. Retrieved 27 Feb 2013 01:28.
           http://en.wikipedia.org/w/index.php?title=Trilinear_interpolation&oldid=533448871
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           http://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if not NUMPY_LT_160:
                if (hasattr(values,
                            'dtype') and not np.can_cast(fill_value_dtype,
                                                         values.dtype,
                                                         casting='same_kind')):
                    raise ValueError("fill_value must be either 'None' or "
                                     "of a type compatible with values")

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances, out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances, out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= .5, i, i + 1))
        return self.values[idx_res]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds

#-------------------------------------------------------------------------------
    
