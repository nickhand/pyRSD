import numpy as np
import bisect
from scipy.interpolate import interp1d
from scipy.integrate import simps

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

# the stochasticity fit parameters at z = 0
lambda_z0 = {'3.05': (-19202.9, 111.395), 
             '2.04': (-3890.46, -211.452), 
             '1.47': (-645.891, -365.149),
             '1.18': (83.2376, -246.78)}
 
# the stochasticity fit parameters at z = 0.5
lambda_z1 = {'4.82': (-35181.3, -5473.79), 
             '3.13': (-6813.82, -241.446), 
             '2.18': (-1335.54, -104.929),
             '1.64': (-168.216, -137.268)}
             
# the stochasticity fit parameters at z = 1.0
lambda_z2 = {'4.64': (-16915.5, -3215.98), 
             '3.17': (-2661.11, -229.627), 
             '2.32': (-427.779, -41.3676)}              
lambdas = {'0.000': lambda_z0, '0.509': lambda_z1, '0.989': lambda_z2}

def stochasticity(bias, z, return_nan=False):
    """
    Given a linear bias and redshift, return the stochasticity based on 
    the mean of simulation results.
    """
    # check redshift value
    z_keys_str = sorted(lambdas.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    if z > np.amax(z_keys): 
        if return_nan: 
            return (np.nan, np.nan)
        else:
            raise ValueError("Cannot determine stochasticity for z > %s" %np.amax(z_keys))
        
    if z < np.amin(z_keys): 
        if return_nan: 
            return (np.nan, np.nan)
        else:
            raise ValueError("Cannot determine stochasticity for z < %s" %np.amin(z_keys))

    # determine the z indices
    redshift_lambdas = {}
    if z in z_keys:
        inds = np.where(z_keys == z)[0][0]
        redshift_lambdas[str(z)] = lambdas[z_keys_str[inds]]
    else:
        
        index_zhi = bisect.bisect(z_keys, z)
        index_zlo = index_zhi - 1
        zhi = z_keys_str[index_zhi]
        zlo = z_keys_str[index_zlo]
        
        redshift_lambdas[zlo] = lambdas[zlo]
        redshift_lambdas[zhi] = lambdas[zhi]

    # now get the mean values for this bias, at this redshift
    z_keys_str = sorted(redshift_lambdas.keys())
    z_keys = np.array(z_keys_str, dtype=float)
    params = []
    for z_key in z_keys_str:
    
        z_lambda = redshift_lambdas[z_key]
        
        bias_keys_str = sorted(z_lambda.keys())
        bias_keys = np.array(bias_keys_str, dtype=float)
        if bias > np.amax(bias_keys): 
            if return_nan: 
                return (np.nan, np.nan)
            else:
                raise ValueError("Cannot determine stochasticity for b > %s" %np.amax(bias_keys))
        if bias < np.amin(bias_keys): 
            if return_nan: 
                return (np.nan, np.nan)
            else:
                raise ValueError("Cannot determine stochasticity for b < %s" %np.amin(bias_keys))
        
        
        if bias in bias_keys:
            inds = np.where(bias_keys == bias)[0][0]
            lam  = z_lambda[bias_keys_str[inds]][0]
            A    = z_lambda[bias_keys_str[inds]][1]
        else:

            index_bhi = bisect.bisect(bias_keys, bias)
            index_blo = index_bhi - 1
            bhi = bias_keys_str[index_bhi]
            blo = bias_keys_str[index_blo]
    
            # get the mean values
            w = (bias - float(blo)) / (float(bhi) - float(blo))
            lam = (1 - w)*z_lambda[blo][0] + w*z_lambda[bhi][0]
            A = (1 - w)*z_lambda[blo][1] + w*z_lambda[bhi][1]
    
        params.append((lam, A))
        
    if len(params) == 1:
        return  params[0][0], params[0][1]
    else:
        
        w = (z - z_keys[0]) / (z_keys[1] - z_keys[0]) 
        lam = (1 - w)*params[0][0] + w*params[1][0]
        A = (1 - w)*params[0][1] + w*params[1][1]
    
        return lam, A
#end stochasticity

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
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 1000)
        Pkmus = f(self, mus)
        return np.array([simps(Pkmus[k_index,:], x=mus) for k_index in range(len(self.k))])
    return wrapper
#-------------------------------------------------------------------------------
def quadrupole(f):
    """
    Decorator to compute the quadrupole from a `self.power` function
    """ 
    def wrapper(self, *args, **kwargs):
        mus = np.linspace(0., 1., 1000)
        Pkmus = f(self, mus)
        kern = 2.5*(3*mus**2 - 1.)
        return np.array([simps(kern*Pkmus[k_index,:], x=mus) for k_index in range(len(self.k))])
    return wrapper
#-------------------------------------------------------------------------------
