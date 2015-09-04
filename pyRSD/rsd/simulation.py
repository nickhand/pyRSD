from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from .. import pygcl, numpy as np, data as sim_data
from . import tools, INTERP_KMIN, INTERP_KMAX
from .mu0_modeling import GPModelParams 

import itertools
import pandas as pd
from sklearn.gaussian_process import GaussianProcess

#-------------------------------------------------------------------------------
# SIMULATION DATA INTERPOLATED WITH A GAUSSIAN PROCESS
#-------------------------------------------------------------------------------
class GaussianProcessSimulationData(Cache):
    """
    Class to interpolate simulation data as a function of bias and redshift, 
    using `GaussianProcess` class from `sklearn.gaussian_process`
    """
    _gp_parameters = ['corr', 'theta0', 'thetaL', 'thetaU', 'random_start', 'regr']
    
    def __init__(self, param_names, use_bias_ratio=False, use_errors=False, **gp_kwargs):
        """
        Parameters
        ----------
        param_names : list
            A list of the names of each parameter to interpolate
        use_bias_ratio : bool, optional
            If `True`, interpolate the ratio of the parameters to the linear
            bias values. Default is `False`
        use_errors : bool, optional
            If `True`, read the errors from param_name + '_err' column and input
            those to the GP. Default is `False`
        """            
        # initalize the base class
        super(GaussianProcessSimulationData, self).__init__()    
        
        # store the parameters
        self.param_names = param_names
        self.use_bias_ratio = use_bias_ratio
        self.use_errors = use_errors
        
        # set defaults of gp kwargs
        self.corr         = gp_kwargs.get('corr', 'squared_exponential')
        self.theta0       = gp_kwargs.get('theta0', [0.1, 0.1])
        self.thetaL       = gp_kwargs.get('thetaL', [1e-4, 1e-4])
        self.thetaU       = gp_kwargs.get('thetaU', [1., 1.])
        self.random_start = gp_kwargs.get('random_start', 100)
        self.regr         = gp_kwargs.get('regr', 'linear')
         
    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    @parameter
    def corr(self, val):
        """
        The GP correlation kernel to use
        """
        return val
        
    @parameter
    def theta0(self, val):
        """
        The initial values of the GP hyperparameters
        """
        return val
            
    @parameter
    def thetaL(self, val):
        """
        The lower bounds on the values of the GP hyperparameters
        """
        return val
        
    @parameter
    def thetaU(self, val):
        """
        The upper bounds on the values of the GP hyperparameters
        """
        return val 
        
    @parameter
    def random_start(self, val):
        """
        The number of restarts from random hyperparameters to do
        """
        return val 
        
    @parameter
    def regr(self, val):
        """
        The type of GP regression model
        """
        return val          
        
    @parameter
    def use_bias_ratio(self, val):
        """
        Interpolate the ratio of the parameters to the linear bias values
        """
        return val
        
    @parameter
    def use_errors(self, val):
        """
        Interpolate using the associated errors on the simulation values
        """
        return val
        
    #---------------------------------------------------------------------------
    # Cached properties
    #---------------------------------------------------------------------------
    @cached_property()
    def data(self):
        """
        Construct the `pandas` `DataFrame` holding the data, as measured
        from sims
        """
        keys = []
        data = []
        for z, params in self.sim_results.iteritems():
            for bias in sorted(params.keys()):
                keys.append((float(z), float(bias)))
                data.append(params[bias])

        # make the data data frame
        index = pd.MultiIndex.from_tuples(keys, names=['z', 'b1'])
        return pd.DataFrame(data, index=index, columns=self.param_names).sort_index()
        
    
    #---------------------------------------------------------------------------
    @cached_property("use_errors", "use_bias_ratio", 'corr', 'theta0', 'thetaU',
                     'thetaL', 'random_start', 'regr', 'data')
    def interpolation_table(self):
        """
        Setup the backend Gaussian processes needed to do the interpolation
        """
        table = {}
        
        # initialize GP for each column in `self.data`
        for col in self.param_names:
            kwargs = {name : getattr(self, name) for name in self._gp_parameters}
            
            # get the data to be interpolated, making sure to remove nulls
            y = self.data[col]
            inds = y.notnull() 
            y = y[inds]            
            
            # check for error columns
            if self.use_errors and col+'_err' in self.data:
                dy = self.data[col+'_err'][inds]
                kwargs['nugget'] = (dy/y)**2
                
            # initialize
            table[col] = GaussianProcess(**kwargs)
            
            # use the results divided by bias
            if self.use_bias_ratio:
                y /= y.index.get_level_values('b1')

            # do the fit
            X = np.asarray(list(y.index.get_values()))
            table[col].fit(X, y)            
            
        return table
        
    #---------------------------------------------------------------------------
    @tools.unpacked
    def __call__(self, b1, z, col=None):
        """
        Evaluate the Gaussian processes at specified bias and redshift
        
        Parameters
        ----------
        b1 : float
            The linear bias parameter
        z : float
            The redshift to evaluate at
        """
        factor = 1. if not self.use_bias_ratio else b1
        if col is None:
            return [(self.interpolation_table[col].predict([z, b1])*factor)[0] for col in self.param_names]
        else:
            return [(self.interpolation_table[col].predict([z, b1])*factor)[0]]
        
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class NonlinearBiasFits(GaussianProcessSimulationData):
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
                 
    # the nonlinear bias values at z = 0.989
    params_z2 = {'2.32' : (1.75, 0.80), 
                 '3.17' : (4.77, 3.15), 
                 '4.64' : (12.80, 10.80)}
    sim_results = {'0':params_z0, '0.509':params_z1, '0.989':params_z2}
    
    #---------------------------------------------------------------------------
    def __init__(self):

        cols = ['b2_00', 'b2_01']
        super(NonlinearBiasFits, self).__init__(cols, use_bias_ratio=True)
    
    #---------------------------------------------------------------------------
  

#-------------------------------------------------------------------------------
class SigmavFits(GaussianProcessSimulationData):
    """
    The halo velocity dispersion in Mpc/h at z = 0, as measured from 
    simulations. The values are taken from Figure 8 of Vlah et al. 2013. 
    
    These are computed in km/s as:
    
    :math: \sigma_v(z=0) * D(z) * f(z) * (100h km/s/Mpc)
    :math: \sigma_v(z=0) ~ 6 Mpc/h.
    
    The results will be returned in units of Mpc/h, in order to remove
    the f dependence by:
    
    sigma_v(z) [Mpc/h] = sigma_v(z) [km/s] / f(z) / D(z) / (100h km/s/Mpc)
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
                 
    sim_results = {'0':params_z0, '0.509':params_z1, '0.989':params_z2}
    
    #---------------------------------------------------------------------------
    def __init__(self):

        cols = ['sigmav']
        
        # convert the values to Mpc/h
        cosmo = pygcl.Cosmology("teppei_sims.ini")
        new_sim_results = {}
        for z_str, values in self.sim_results.iteritems():
            z = float(z_str)
            params = {}
            for b1, sigma in values.iteritems():
               factor = cosmo.f_z(z)*cosmo.D_z(z)*100.
               params[b1] = (sigma/factor)
               
            new_sim_results[z_str] = params
        self.sim_results = new_sim_results
        super(SigmavFits, self).__init__(cols, use_bias_ratio=False)
        
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------  
# SIMULATION DATA INTERPOLATION ON A GRID
#-------------------------------------------------------------------------------
class InterpolatedSimulationData(Cache):
    """
    A base class for computing power moments from interpolated simulation data
    """
    def __init__(self, power_lin, z, sigma8_z, f):
        
        # initialize the Cache base class
        Cache.__init__(self)
        
        # set the parameters
        self.z         = z
        self.sigma8_z  = sigma8_z
        self.f         = f
        self.power_lin = power_lin
        self.cosmo     = self.power_lin.GetCosmology()

        # make sure power spectrum redshift is 0
        msg = "input linear power spectrum must be defined at z = 0"
        assert self.power_lin.GetRedshift() == 0., msg

    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    @parameter
    def f(self, val):
        """
        The growth rate, defined as the `dlnD/dlna`.
        """
        return val

    #---------------------------------------------------------------------------
    @parameter
    def power_lin(self, val):
        """
        The `pygcl.LinearPS` object defining the linear power spectrum at `z=0`
        """
        return val

    #---------------------------------------------------------------------------
    @parameter
    def cosmo(self, val):
        """
        The cosmology of the input linear power spectrum
        """
        return val

    #---------------------------------------------------------------------------
    @parameter
    def z(self, val):
        """
        Desired redshift for the output power spectrum
        """
        return val

    #---------------------------------------------------------------------------        
    @parameter
    def sigma8_z(self, val):
        """
        Sigma_8 at `z=0` to compute the spectrum at, which gives the 
        normalization of the linear power spectrum
        """
        return val
            
#-------------------------------------------------------------------------------
class SimulationP11(InterpolatedSimulationData):
    """
    Dark matter model for the mu^4 term of P11, computed by interpolating 
    simulation data as a function of (f*sigma8)^2
    """
    
    def __init__(self, power_lin, z, sigma8_z, f):
        """
        Parameters
        ----------
        power_lin : pygcl.LinearPS
            Linear power spectrum with Eisenstein-Hu no-wiggle transfer function
        z : float
            Redshift to compute the power spectrum at
        sigma8 : float
            Desired sigma8 value
        f : float
            Desired logarithmic growth rate value
        """
        # initialize the base class holding parameters
        super(SimulationP11, self).__init__(power_lin, z, sigma8_z, f)
        
        # make sure power spectrum is no-wiggle
        if self.cosmo.GetTransferFit() != pygcl.Cosmology.EH_NoWiggle:
            raise ValueError("Interpolated sim results require the no-wiggle power spectrum")
        
        # load the data
        self._load_data()
          
    #---------------------------------------------------------------------------
    def _load_data(self):
        """
        Load the P11 simulation data
        """
        # cosmology and linear power spectrum for teppei's sims
        cosmo = pygcl.Cosmology("teppei_sims.ini", pygcl.Cosmology.EH_NoWiggle)
        Plin = pygcl.LinearPS(cosmo, 0.)

        # the interpolation data
        redshifts = [0., 0.509, 0.989]
        data = [sim_data.P11_mu4_z_0_000(), sim_data.P11_mu4_z_0_509(), sim_data.P11_mu4_z_0_989()]
        interp_vars = redshifts
      
        # make the data frame
        k = data[0][:,0]
        index_tups = list(itertools.product(interp_vars, k))
        index = pd.MultiIndex.from_tuples(index_tups, names=['z', 'k'])
        d = []
        for i, x in enumerate(data):
            d += list(x[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(x[:,0]) * cosmo.f_z(redshifts[i])**2))
        
        # now store the results
        self.data = pd.DataFrame(data=d, index=index, columns=['P11'])
        self.interpolation_grid = {}
        self.interpolation_grid['z'] = self.data.index.get_level_values('z').unique()
        self.interpolation_grid['k'] = self.data.index.get_level_values('k').unique()
      
    #---------------------------------------------------------------------------
    @cached_property()
    def interpolation_table(self):
        """
        Return an interpolation table for P11, normalized by the no-wiggle
        power spectrum
        """
        # the interpolation grid points
        zs = self.interpolation_grid['z']
        ks = self.interpolation_grid['k']
        
        # get the grid values
        grid_vals = []
        for i, z in enumerate(zs):
            grid_vals += list(self.data.xs(z).P11)
        grid_vals = np.array(grid_vals).reshape((len(zs), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((zs, ks), grid_vals)

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of (f*sigma8)^2 by assuming the shape of
        the normalized power spectrum is the same, i.e., just rescaling
        by the low-k amplitude
        """
        zs = self.interpolation_grid['z']
        if x < np.amin(zs):
            val = np.amin(zs)
        else:
            val = np.amax(zs)
        
        # get the renormalization factor
        normed_power = self.power_lin(k) / self.cosmo.sigma8()**2
        factor = x*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [val, k]
        else:
            pts = np.asarray(list(itertools.product([val], k)))
        return self.interpolation_table(pts)*factor
            
    #---------------------------------------------------------------------------
    @tools.unpacked
    def __call__(self, k):
        """
        Evaluate P11 at the redshift `z` and the specified `k`
        """
        fs8_sq = (self.f*self.sigma8_z)**2
        
        # extrapolate?
        grid_pts = self.interpolation_grid['z']
        if self.z < np.amin(grid_pts) or self.z > np.amax(grid_pts):
            return self._extrapolate(self.z, k)
        
        # get the renormalization factor
        normed_power = self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [self.z, k]
        else:
            pts = np.asarray(list(itertools.product([self.z], k)))
        return self.interpolation_table(pts)*factor
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class SimulationPdv(InterpolatedSimulationData):
    """
    Dark matter model for density -- velocity divergence cross power spectrum
    Pdv, computed by interpolating simulation data as a function of f*sigma8^2
    """
    
    def __init__(self, power_lin, z, sigma8_z, f):
        """
        Parameters
        ----------
        power_lin : pygcl.LinearPS
            Linear power spectrum with Eisenstein-Hu no-wiggle transfer function
        z : float
            Redshift to compute the power spectrum at
        sigma8_z : float
            Desired sigma8 value at z
        f : float
            Desired logarithmic growth rate value
        """
        # initialize the base class holding parameters
        super(SimulationPdv, self).__init__(power_lin, z, sigma8_z, f)
        
        # make sure power spectrum is no-wiggle
        if self.cosmo.GetTransferFit() != pygcl.Cosmology.EH_NoWiggle:
            raise ValueError("Interpolated sim results require the no-wiggle power spectrum")
        
        # load the data
        self._load_data()

    #---------------------------------------------------------------------------
    def _load_data(self):
        """
        Load the simulation data
        """
        # cosmology and linear power spectrum for teppei's sims
        cosmo = pygcl.Cosmology("teppei_sims.ini", pygcl.Cosmology.EH_NoWiggle)
        Plin = pygcl.LinearPS(cosmo, 0.)

        # the interpolation data
        redshifts = [0., 0.509, 0.989]
        data = [sim_data.Pdv_mu0_z_0_000(), sim_data.Pdv_mu0_z_0_509(), sim_data.Pdv_mu0_z_0_989()]
        interp_vars = redshifts

        # make the data frame
        k = data[0][:,0]
        index_tups = list(itertools.product(interp_vars, k))
        index = pd.MultiIndex.from_tuples(index_tups, names=['z', 'k'])
        d = []
        for i, x in enumerate(data):
            d += list(x[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(x[:,0]) * cosmo.f_z(redshifts[i])))
        
        # now store the results
        self.data = pd.DataFrame(data=d, index=index, columns=['Pdv'])
        self.interpolation_grid = {}
        self.interpolation_grid['z'] = self.data.index.get_level_values('z').unique()
        self.interpolation_grid['k'] = self.data.index.get_level_values('k').unique()

    #---------------------------------------------------------------------------
    @cached_property()
    def interpolation_table(self):
        """
        Return an interpolation table for Pdv, normalized by the no-wiggle
        power spectrum
        """
        # the interpolation grid points
        zs = self.interpolation_grid['z']
        ks = self.interpolation_grid['k']
        
        # get the grid values
        grid_vals = []
        for i, z in enumerate(zs):
            grid_vals += list(self.data.xs(z).Pdv)
        grid_vals = np.array(grid_vals).reshape((len(zs), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((zs, ks), grid_vals)

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of f*sigma8^2 by assuming the shape of
        the normalized power spectrum is the same, i.e., just rescaling
        by the low-k amplitude
        """
        zs = self.interpolation_grid['z']
        if x < np.amin(zs):
            val = np.amin(zs)
        else:
            val = np.amax(zs)
        
        # get the renormalization factor
        normed_power = self.power_lin(k) / self.cosmo.sigma8()**2
        factor = x*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [val, k]
        else:
            pts = np.asarray(list(itertools.product([val], k)))
        return self.interpolation_table(pts)*factor
            
    #---------------------------------------------------------------------------
    @tools.unpacked
    def __call__(self, k):
        """
        Evaluate Pdv at the redshift `z` and the specified `k`
        """
        fs8_sq = self.f*self.sigma8_z**2
        
        # extrapolate?
        grid_pts = self.interpolation_grid['z']
        if self.z < np.amin(grid_pts) or self.z > np.amax(grid_pts):
            return self._extrapolate(self.z, k)
        
        # get the renormalization factor
        normed_power = self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [self.z, k]
        else:
            pts = np.asarray(list(itertools.product([self.z], k)))
        return self.interpolation_table(pts)*factor
    #---------------------------------------------------------------------------

 #------------------------------------------------------------------------------
class Mu6CorrectionParams(GPModelParams):
    """
    The model parameters for the mu^6 correction 
    """
    def __init__(self):
        path = sim_data.mu6_correction_params()
        super(Mu6CorrectionParams, self).__init__(path)
    
    
