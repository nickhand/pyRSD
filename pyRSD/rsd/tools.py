from .. import pygcl

import numpy as np
import bisect
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import brentq
import pandas as pd
from sklearn.gaussian_process import GaussianProcess
    
def extrap1d(interpolator):
    """
    A 1d extrapolator function, using linear extrapolation
    
    Parameters
    ----------
    interpolator : scipy.interpolate.interp1d 
        the interpolator function
    
    Returns
    -------
    ufunclike : function
        the extrapolator function
    """
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike
#end extrap1d 

#-------------------------------------------------------------------------------

class LambdaStochasticity(object):
    """
    Class implementing the fits to the scale-dependent stochasticity lambda
    """
    # the stochasticity fit parameters at z = 0
    lambda_z0 = {'3.05': (-19202.9, 111.395), 
                 '2.04': (-3890.46, -211.452), 
                 '1.47': (-645.891, -365.149),
                 '1.18': (83.2376, -246.78)}

    # the stochasticity fit parameters at z = 0.509
    lambda_z1 = {'4.82': (-35181.3, -5473.79), 
                 '3.13': (-6813.82, -241.446), 
                 '2.18': (-1335.54, -104.929),
                 '1.64': (-168.216, -137.268)}

    # the stochasticity fit parameters at z = 1.0
    lambda_z2 = {'4.64': (-16915.5, -3215.98), 
                 '3.17': (-2661.11, -229.627), 
                 '2.32': (-427.779, -41.3676)} 
                              
    lambdas = [(0., lambda_z0), (0.509, lambda_z1), (0.989, lambda_z2)]

    def __init__(self, return_nan=False, corr_model="linear"):
        
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
    def _setup_sim_results(self):
        """
        Construct the pandas DataFrame holding the lambda data from sims
        """
        keys = []
        data = []
        for red, params in LambdaStochasticity.lambdas:
            for b in sorted(params.keys()):
                keys.append((red, float(b)))
                data.append(params[b])

        index = pd.MultiIndex.from_tuples(keys, names=['z', 'b1'])
        self.data = pd.DataFrame(data, index=index, columns=['constant', 'slope'])
        
        # save the redshift info too
        self.redshifts = np.array(self.data.index.levels[0])
        self.zmax = np.amax(self.redshifts)
        self.zmin = np.amin(self.redshifts)
        
    #end _setup_sim_results
    
    #--------------------------------------------------------------------------- 
    def _check_z_bounds(self, z):   
        """
        Check the redshift bounds
        """
        assert np.isscalar(z), 'Redshift input must be scalar'
        
        # check redshift value
        if z > self.zmax: 
            if self.return_nan: 
                return (np.nan, np.nan)
            else:
                raise ValueError("Cannot determine stochasticity for z > %s" %self.zmax)

        if z < self.zmin: 
            if self.return_nan: 
                return (np.nan, np.nan)
            else:
                raise ValueError("Cannot determine stochasticity for z < %s" %self.zmin)
                
    #end _check_z_bounds
    
    #---------------------------------------------------------------------------
    def _setup_gps(self):
        """
        Setup the Gaussian Processes as a function of bias at each redshift
        """
        self.gps = {}
        # loop over each redshift
        for z in self.redshifts:

            self.gps[z] = {}
            
            frame = self.data.xs(z)
            biases = np.array(frame.index)
            
            # initialize the Gaussian Process
            gp_const = GaussianProcess(corr=self.corr_model, theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
            gp_slope = GaussianProcess(corr=self.corr_model, theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
            x = np.atleast_2d(biases).T
            
            # the constant
            y_const = np.atleast_2d(frame.constant).T
            gp_const.fit(x, y_const)
            self.gps[z]['constant'] = gp_const
            
            # the slope
            y_slope = np.atleast_2d(frame.slope).T
            gp_slope.fit(x, y_slope)
            self.gps[z]['slope'] = gp_slope
    #end _setup_gps
    
    #---------------------------------------------------------------------------
    def __call__(self, bias, z):
        """
        Evaluate lambda at specified bias and redshift
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

            # predict the constant
            f = getattr(self.gps[zi]['constant'], 'predict')
            constant = f(np.atleast_2d(bias).T)

            f = getattr(self.gps[zi]['slope'], 'predict')
            slope = f(np.atleast_2d(bias).T)

            params.append((constant, slope))
            
        if len(params) == 1:
            return  params[0][0], params[0][1]
        else:

            
            zlo, zhi = redshifts[0], redshifts[1]
            w = (z - zlo) / (zhi - zlo) 
            constant = (1 - w)*params[0][0] + w*params[1][0]
            slope = (1 - w)*params[0][1] + w*params[1][1]

            return constant, slope
    #end __call__
    
#-------------------------------------------------------------------------------
def b2_00(bias, z):
    """
    Given a linear bias and redshift, return the nonlinear bias for the P00_hh 
    term based on the mean of simulation results.
    """
    bias = np.array(bias, copy=False, ndmin=1)
    
    bias_z0 = {'1.18' : -0.39, '1.47' : -0.08, '2.04' : 0.91, '3.05' : 3.88}
    bias_z1 = {'1.64' : 0.18, '2.18' : 1.29, '3.13' : 4.48, '4.82' : 12.70}
    bias_z2 = {'2.32' : 1.75, '3.17' : 4.77, '4.64' : 12.80}
    
    biases = {'0.000': bias_z0, '0.509': bias_z1, '0.989': bias_z2}
    
    # check redshift value
    z_keys_str = sorted(biases.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    if z > np.amax(z_keys): raise ValueError("Cannot determine b2_00 for z > %s" %np.amax(z_keys))
    if z < np.amin(z_keys): raise ValueError("Cannot determine b2_00 for z < %s" %np.amin(z_keys))

    # determine the z indices
    redshift_biases = {}
    if z in z_keys:
        inds = np.where(z_keys == z)[0][0]
        redshift_biases[str(z)] = biases[z_keys_str[inds]]
    else:
        
        index_zhi = bisect.bisect(z_keys, z)
        index_zlo = index_zhi - 1
        zhi = z_keys_str[index_zhi]
        zlo = z_keys_str[index_zlo]
        
        redshift_biases[zlo] = biases[zlo]
        redshift_biases[zhi] = biases[zhi]

    # now get the mean values for this bias, at this redshift
    z_keys_str = sorted(redshift_biases.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    params = []
    for z_key in z_keys_str:

        z_bias = redshift_biases[z_key]
        
        b1s = np.array(z_bias.keys(), dtype=float)
        inds = np.argsort(b1s)
        b1s = b1s[inds]
        b2s = np.array(z_bias.values())[inds]
        
        interp = interp1d(b1s, b2s/b1s)
        extrap = extrap1d(interp)
        params.append(extrap(bias)*bias)
        
    if len(params) == 1:
        return  params[0]
    else:

        w = (z - z_keys[0]) / (z_keys[1] - z_keys[0]) 
        return (1 - w)*params[0] + w*params[1]
        
#end b2_00

#-------------------------------------------------------------------------------
def b2_01(bias, z):
    """
    Given a linear bias and redshift, return the nonlinear bias for the P01_hh 
    term based on the mean of simulation results.
    """
    bias = np.array(bias, copy=False, ndmin=1)
    
    bias_z0 = {'1.18' : -0.45, '1.47' : -0.35, '2.04' : 0.14, '3.05' : 2.00}
    bias_z1 = {'1.64' : -0.20, '2.18' : 0.48, '3.13' : 2.60, '4.82' : 9.50}
    bias_z2 = {'2.32' : 0.80, '3.17' : 3.15, '4.64' : 10.80}
    
    biases = {'0.000': bias_z0, '0.509': bias_z1, '0.989': bias_z2}
    
    # check redshift value
    z_keys_str = sorted(biases.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    if z > np.amax(z_keys): raise ValueError("Cannot determine b2_00 for z > %s" %np.amax(z_keys))
    if z < np.amin(z_keys): raise ValueError("Cannot determine b2_00 for z < %s" %np.amin(z_keys))

    # determine the z indices
    redshift_biases = {}
    if z in z_keys:
        inds = np.where(z_keys == z)[0][0]
        redshift_biases[str(z)] = biases[z_keys_str[inds]]
    else:
        
        index_zhi = bisect.bisect(z_keys, z)
        index_zlo = index_zhi - 1
        zhi = z_keys_str[index_zhi]
        zlo = z_keys_str[index_zlo]
        
        redshift_biases[zlo] = biases[zlo]
        redshift_biases[zhi] = biases[zhi]

    # now get the mean values for this bias, at this redshift
    z_keys_str = sorted(redshift_biases.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    params = []
    for z_key in z_keys_str:

        z_bias = redshift_biases[z_key]
        
        b1s = np.array(z_bias.keys(), dtype=float)
        inds = np.argsort(b1s)
        b1s = b1s[inds]
        b2s = np.array(z_bias.values())[inds]
        
        interp = interp1d(b1s, b2s/b1s)
        extrap = extrap1d(interp)
        params.append(extrap(bias)*bias)
        
    if len(params) == 1:
        return  params[0]
    else:

        w = (z - z_keys[0]) / (z_keys[1] - z_keys[0]) 
        return (1 - w)*params[0] + w*params[1]

#end b2_01
#-------------------------------------------------------------------------------
def sigma_from_sims(bias, z):
    """
    The halo velocity dispersion as measured from simulations, as computed
    from Figure 7 of Vlah et al. 2013. These are computed in km/s as
    :math: \sigma_v(z=0) * D(z) * f(z) * H(z) / h where 
    :math: \sigma_v(z=0) ~ 6 Mpc/h.
    """
    bias = np.array(bias, copy=False, ndmin=1)
    
    sigma_z0 = {'1.18' : 3.06, '1.47' : 3.02, '2.04' : 2.96, '3.05' : 2.88}
    sigma_z1 = {'1.64' : 3.57, '2.18' : 3.52, '3.13' : 3.46, '4.82' : 3.39}
    sigma_z2 = {'2.32' : 3.4, '3.17' : 3.37, '4.64' : 3.3}
    
    sigmas = {'0.000': sigma_z0, '0.509': sigma_z1, '0.989': sigma_z2}
    
    # check redshift value
    z_keys_str = sorted(sigmas.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    if z > np.amax(z_keys): raise ValueError("Cannot determine sigma for z > %s" %np.amax(z_keys))
    if z < np.amin(z_keys): raise ValueError("Cannot determine sigma for z < %s" %np.amin(z_keys))

    # determine the z indices
    redshift_sigmas = {}
    if z in z_keys:
        inds = np.where(z_keys == z)[0][0]
        redshift_sigmas[str(z)] = sigmas[z_keys_str[inds]]
    else:
        
        index_zhi = bisect.bisect(z_keys, z)
        index_zlo = index_zhi - 1
        zhi = z_keys_str[index_zhi]
        zlo = z_keys_str[index_zlo]
        
        redshift_sigmas[zlo] = sigmas[zlo]
        redshift_sigmas[zhi] = sigmas[zhi]

    # now get the mean values for this bias, at this redshift
    z_keys_str = sorted(redshift_sigmas.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    values = []
    for z_key in z_keys_str:

        z_bias = redshift_sigmas[z_key]
        
        b1s = np.array(z_bias.keys(), dtype=float)
        inds = np.argsort(b1s)
        b1s = b1s[inds]
        sigs = np.array(z_bias.values())[inds]
        
        interp = interp1d(b1s, sigs)
        extrap = extrap1d(interp)
        values.append(extrap(bias))
        
    if len(values) == 1:
        return  values[0]*100.
    else:

        w = (z - z_keys[0]) / (z_keys[1] - z_keys[0]) 
        return 100.*((1 - w)*values[0] + w*values[1])
#end sigma_from_sims

#-------------------------------------------------------------------------------
def monopole(f):
    """
    Decorator to compute the monopole from a `self.power` function
    """     
    def wrapper(self, *args):
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, mus)
        return np.array([simps(Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k))])
    return wrapper
#-------------------------------------------------------------------------------
def quadrupole(f):
    """
    Decorator to compute the quadrupole from a `self.power` function
    """     
    def wrapper(self, *args):
        mus = np.linspace(0., 1., 101)
        Pkmus = f(self, mus)
        kern = 2.5*(3*mus**2 - 1.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k))])
    return wrapper
#-------------------------------------------------------------------------------
def mu_vectorize(f):
    """
    Vectorize the function to handle scalar or array_like `mu` input
    """ 
    def wrapper(self, *args):
        mu = args[0]
        if np.isscalar(mu):
            return f(self, mu)
        else:
            return np.vstack([f(self, imu) for imu in mu]).T
        
    return wrapper
#-------------------------------------------------------------------------------
def hexadecapole(f):
    """
    Decorator to compute the hexadecapole from a `self.power` function
    """ 
    def wrapper(self, *args):
        mus = np.linspace(0., 1., 1001)
        Pkmus = f(self, mus)
        kern = 9./8.*(35*mus**4 - 30.*mu**2 + 3.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in xrange(len(self.k))])
    return wrapper
    
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
        return bias - bias_Tinker(sigma)
        
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
    
    