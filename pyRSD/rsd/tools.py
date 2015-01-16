from .. import pygcl

import numpy as np
import bisect
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.optimize import brentq
import pandas as pd
from sklearn.gaussian_process import GaussianProcess
import warnings

#-------------------------------------------------------------------------------
class RSDSpline(InterpolatedUnivariateSpline):
    """
    Class to implement a spline that remembers the x-domain
    """
    def __init__(self, *args, **kwargs):
        
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
#endclass RSDSpline

#------------------------------------------------------------------------------0
class SimInterpolator(object):
    """
    Class to interpolate simulation data as a function of bias and redshift
    """
    
    def __init__(self, return_nan=False, corr_model="linear", columns=(), use_ratio=False):
        
        # store the sim specific variables
        self.columns = columns
        self.use_ratio = use_ratio
        
        # whether to return NaNs outside bounds, rather than raising exception
        self.return_nan = return_nan
        
        # GP correlation model
        self.corr_model = corr_model
        
        # setup sim data
        self._setup_sim_results()
        
        # setup the gaussian processes
        self._setup_gps()
         
    #__init__
    
    #--------------------------------------------------------------------------- 
    def _check_z_bounds(self, z):   
        """
        Check the redshift bounds
        """
        assert np.isscalar(z), 'Redshift input must be scalar'
        
        # check redshift value
        if z > self.zmax: 
            if self.return_nan: 
                return (np.nan,)*len(self.columns)
            else:
                raise ValueError("Cannot perform interpolation for z > %s" %self.zmax)

        if z < self.zmin: 
            if self.return_nan: 
                return (np.nan,)*len(self.columns)
            else:
                raise ValueError("Cannot perform interpolation for z < %s" %self.zmin)
                
    #end _check_z_bounds
    
    #---------------------------------------------------------------------------
    def _setup_sim_results(self):
        """
        Construct the pandas DataFrame holding the data from sims
        """
        keys = []
        data = []
        for z, params in self.sim_results:
            for bias in sorted(params.keys()):
                keys.append((z, float(bias)))
                data.append(params[bias])

        index = pd.MultiIndex.from_tuples(keys, names=['z', 'b1'])
        self.data = pd.DataFrame(data, index=index, columns=self.columns)
        
        # save the redshift info too
        self.redshifts = np.array(self.data.index.levels[0])
        self.zmax = np.amax(self.redshifts)
        self.zmin = np.amin(self.redshifts)
        
    #end _setup_sim_results
    
    #---------------------------------------------------------------------------
    def _setup_gps(self):
        """
        Setup the Gaussian Processes as a function of bias at each redshift
        """
        self.gps = {}
        gp_kwargs = {'corr':self.corr_model, 'theta0':1e-2, 'thetaL':1e-4, 
                     'thetaU':0.1, 'random_start':100}

        # loop over each redshift
        for z in self.redshifts:

            self.gps[z] = {}
            frame = self.data.xs(z)
            biases = np.array(frame.index)
            x = np.atleast_2d(biases).T
            
            # initialize a Gaussian Process for each fitting column
            for col in self.columns:
                gp = GaussianProcess(**gp_kwargs)
                sim_data = frame[col]
                if self.use_ratio:
                    sim_data /= biases
                y = np.atleast_2d(sim_data).T
                gp.fit(x, y)
                self.gps[z][col] = gp
            
    #end _setup_gps
    
    #---------------------------------------------------------------------------
    def _check_scalar(self, val):
        """
        Check if the value is a scalar and return it if True
        """
        try:
            if len(val) == 1:
                return val[0]
            else:
                return val
        except:
            return val
    #end _check_scalar
    
    #---------------------------------------------------------------------------
    def _check_tuple_length(self, val):
        """
        Check for a return tuple length of one
        """
        if len(self.columns) == 1:
            return val[0]
        else:
            return val
    #end _check_tuple_length
    
    #---------------------------------------------------------------------------
    def __call__(self, bias, z):
        """
        Evaluate at specified bias and redshift
        """
        self._check_z_bounds(z)
        
        # determine the z indices
        redshifts = []
        if z in self.redshifts:
            redshifts.append(z)
        else:

            index_zhi = bisect.bisect(self.redshifts, z)
            index_zlo = index_zhi - 1
            zhi = self.redshifts[index_zhi]
            zlo = self.redshifts[index_zlo]
            redshifts.append(zlo)
            redshifts.append(zhi)
        
        params = []
        for zi in redshifts:
            values = ()
            for col in self.columns:
                f = getattr(self.gps[zi][col], 'predict')
                value = f(np.atleast_2d(bias).T)
                if self.use_ratio: value *= bias
                values += (value,)
            params.append(values)
            
        # now return
        if len(params) == 1:
            return self._check_tuple_length(tuple(map(self._check_scalar, params[0])))
        else:
            
            zlo, zhi = redshifts[0], redshifts[1]
            w = (z - zlo) / (zhi - zlo) 
            toret = ()
            for i, col in enumerate(self.columns):
                value = (1 - w)*params[0][i] + w*params[1][i]
                toret += (self._check_scalar(value),)

            return self._check_tuple_length(toret)
    #end __call__
    
    #---------------------------------------------------------------------------
#endclass SimInterpolator
    
#-------------------------------------------------------------------------------
class LambdaStochasticityFits(SimInterpolator):
    """
    Class implementing the fits to the scale-dependent stochasticity lambda
    """
    # the stochasticity fit parameters at z = 0
    params_z0 = {'3.05': (-19202.9, 111.395), 
                 '2.04': (-3890.46, -211.452), 
                 '1.47': (-645.891, -365.149),
                 '1.18': (83.2376, -246.78)}

    # the stochasticity fit parameters at z = 0.509
    params_z1 = {'4.82': (-35181.3, -5473.79), 
                 '3.13': (-6813.82, -241.446), 
                 '2.18': (-1335.54, -104.929),
                 '1.64': (-168.216, -137.268)}

    # the stochasticity fit parameters at z = 1.0
    params_z2 = {'4.64': (-16915.5, -3215.98), 
                 '3.17': (-2661.11, -229.627), 
                 '2.32': (-427.779, -41.3676)}                
    sim_results = [(0., params_z0), (0.509, params_z1), (0.989, params_z2)]
    
    #---------------------------------------------------------------------------   
    def __init__(self, *args, **kwargs):

        kwargs['columns'] = ['constant', 'slope']
        kwargs['use_ratio'] = False
        super(LambdaStochasticityFits, self).__init__(*args, **kwargs)

    #end __init__
    #---------------------------------------------------------------------------
#endclass LambdaStochasticityFits

#-------------------------------------------------------------------------------
class NonlinearBiasFits(SimInterpolator):
    """
    Class implementing nonlinear bias fits from Vlah et al. 2013
    """
    # the nonlinear bias values at z = 0
    params_z0 = {'1.18' : (-0.39, -0.45), 
                 '1.47' : (-0.08, -0.35), 
                 '2.04' : (0.91, 0.14), 
                 '3.05' : (3.88, 2.00)}
    
    # the nonlinear bias values at z = 0.509
    params_z1 = {'1.64' : (0.18, -0.20), 
                 '2.18' : (1.29, 0.48), 
                 '3.13' : (4.48, 2.60), 
                 '4.82' : (12.70, 9.50)}
                 
    # the nonlinear bias values at z = 0.509
    params_z2 = {'2.32' : (1.75, 0.80), 
                 '3.17' : (4.77, 3.15), 
                 '4.64' : (12.80, 10.80)}
    sim_results = [(0., params_z0), (0.509, params_z1), (0.989, params_z2)]
    
    #---------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):

        kwargs['columns'] = ['b2_00', 'b2_01']
        kwargs['use_ratio'] = True
        super(NonlinearBiasFits, self).__init__(*args, **kwargs)

    #end __init__
    
    #---------------------------------------------------------------------------
#endclass NonlinearBiasFits    

#-------------------------------------------------------------------------------
class SigmavFits(SimInterpolator):
    """
    The halo velocity dispersion as measured from simulations, as computed
    from Figure 7 of Vlah et al. 2013. These are computed in km/s as
    :math: \sigma_v(z=0) * D(z) * f(z) * H(z) / h where 
    :math: \sigma_v(z=0) ~ 6 Mpc/h.
    """
    # the values at z = 0
    params_z0 = {'1.18' : (306.), 
                 '1.47' : (302.), 
                 '2.04' : (296.), 
                 '3.05' : (288.)}
    
    # the values at z = 0.509
    params_z1 = {'1.64' : (357.), 
                 '2.18' : (352.), 
                 '3.13' : (346.), 
                 '4.82' : (339.)}
                 
    # the values at z = 0.509
    params_z2 = {'2.32' : (340.), 
                 '3.17' : (337.), 
                 '4.64' : (330.)}
    sim_results = [(0., params_z0), (0.509, params_z1), (0.989, params_z2)]
    
    #---------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):

        kwargs['columns'] = ['sigmav']
        kwargs['use_ratio'] = False
        super(SigmavFits, self).__init__(*args, **kwargs)

    #end __init__
    
    #---------------------------------------------------------------------------
#endclass SigmavFits

#-------------------------------------------------------------------------------
def monopole(f):
    """
    Decorator to compute the monopole from a `self.power` function
    """ 
    warnings.filterwarnings("ignore", category=DeprecationWarning,module="scipy")    
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
    warnings.filterwarnings("ignore", category=DeprecationWarning,module="scipy")    
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
    warnings.filterwarnings("ignore", category=DeprecationWarning,module="scipy") 
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, mus, **kwargs)
        kern = 9./8.*(35*mus**4 - 30.*mu**2 + 3.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k_obs))])
    return wrapper
    
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
#endclass SigmaBiasRelation

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
#end mass_from_bias

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
#end bias_Tinker

#-------------------------------------------------------------------------------
def sigma_from_bias(bias, z, linearPS):
    """
    Return sigma from bias
    """
    # normalized Teppei's sims at z = 0.509 and sigma_sAsA = 3.6
    sigma0 = 3.6 # in Mpc/h
    M0 = 5.4903e13 # in M_sun / h
    return sigma0 * (mass_from_bias(bias, z, linearPS) / M0)**(1./3)
 
#end sigma_from_bias

#-------------------------------------------------------------------------------  
    
    
