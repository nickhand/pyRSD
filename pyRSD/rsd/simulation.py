from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from .. import pygcl, numpy as np, data as sim_data
from . import tools

import itertools
import pandas as pd
from sklearn.gaussian_process import GaussianProcess


#-------------------------------------------------------------------------------
# simulation measurements, interpolated with a gaussian process
#-------------------------------------------------------------------------------
class GaussianProcessSimulationData(Cache):
    """
    Class to interpolate simulation data as a function of bias and redshift, 
    using `GaussianProcess` class from `sklearn.gaussian_process`
    """
    _gp_parameters = ['corr', 'theta0', 'thetaL', 'thetaU', 'random_start', 'regr']
    
    def __init__(self, param_names, data, use_bias_ratio=False, use_errors=False, **gp_kwargs):
        """
        Parameters
        ----------
        param_names : list of str
            the names of the parameters we are interpolated -- each must
            be a column in `data`
        data : pd.DataFrame
            the data frame holding the data to be interpolated using the GP, 
            where the index is interpreted as the independent variables
        use_errors : bool, optional
            If `True`, read the errors from param_name + '_err' column and input
            those to the GP. Default is `False`
        """            
        # initalize the base Cache class
        super(GaussianProcessSimulationData, self).__init__()    
        
        self.param_names = param_names
        self.data = data
        
        # whether we are using errors
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
    def data(self, val):
        """
        The data frame holding the data to interpolate
        """
        return val
        
    @parameter
    def use_errors(self, val):
        """
        Interpolate using the associated errors on the simulation values
        """
        return val
        
    @parameter
    def param_names(self, val):
        """
        The names of the parameters to interpolate
        """
        return val
        
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("data")
    def independent_vars(self):
        """
        The names of the independent variables
        """
        return list(self.data.index.names)
        
    @cached_property('data', 'param_names', 'use_errors', 'corr', 'theta0', 
                        'thetaU', 'thetaL', 'random_start', 'regr')
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
            
            # do the fit
            X = np.asarray(list(y.index.get_values()))
            table[col].fit(X, y)            
            
        return table
        
    @tools.unpacked
    def __call__(self, select=None, **indep_vars):
        """
        Evaluate the Gaussian processes at the specified independent variables
        
        Parameters
        ----------
        select : str or list of str, optional
            If not `None`, only return the parameters with names specified here
        indep_vars : keywords
            the independent variables to evaluate at
        """
        for p in indep_vars:
            if p not in self.independent_vars:
                raise ValueError("please specify the `%s` independent variable" %p)
        
        # the domain point to evaluate at        
        pt = np.array([indep_vars[x] for x in self.independent_vars]).reshape((1,-1))
        
        # determine which parameters we are returning
        if select is None:
            select = self.param_names
        elif isinstance(select, str):
            select = [select]
        
        for col in select:
            if col not in self.param_names:
                raise ValueError("the parameter '%s' is not valid" %col)
        
        return [(self.interpolation_table[sel].predict(pt))[0] for sel in select]


class NonlinearBiasFits(GaussianProcessSimulationData):
    """
    Class implementing nonlinear bias fits from Vlah et al. 2013
    """
    def __init__(self, fit='runPB', **kwargs):
        """
        Parameters
        ----------
        fit : str, {`runPB`, `zvonimir`}
            which fit to use, either from RunPB simulations or Zvonimir's
            simulations
        """
        # load the data from the json file
        data = sim_data.nonlinear_bias_params(fit)

        cols = ['b2_00', 'b2_01']
        use_errors = all(col+'_err' in data for col in cols)

        # interpolate b2 / b1
        for col in cols:
            data[col] /= data['b1']
            if use_errors:
                data[col+'_err'] /= data['b1']

        data = data.set_index(['z', 'b1'])
        super(NonlinearBiasFits, self).__init__(cols, data, use_errors=use_errors, **kwargs)
        
    def __call__(self, **kwargs):
        
        b1 = kwargs['b1']
        toret = super(NonlinearBiasFits, self).__call__(**kwargs)
        if isinstance(toret, list):
            return [b1*x for x in toret]
        else:
            return b1*toret
    
class VelocityDispersionFits(GaussianProcessSimulationData):
    """
    The halo velocity dispersion in Mpc/h, normalized by f(z)*sigma8(z),
    as measured from the runPB simulations
    """
    def __init__(self):
        
        # load the data from the json file
        data = sim_data.velocity_dispersion_params()
        
        # divide measured sigma_v by f
        data['sigma_v'] /= data['f']
        
        # set the index
        data = data.set_index(['sigma8_z', 'b1'])
        kws = {'use_errors':False, 'regr':'quadratic'}
        super(VelocityDispersionFits, self).__init__(['sigma_v'], data, **kws)
        
class P11PlusP02Correction(GaussianProcessSimulationData):
    """
    Return the correction parameters for the halo P11 + P02
    correction model.
    
    The parameters are: [`b2_00`, `A1`, `A2`, `A3`, `A4`]
    
    where `b2_00` is used to compute `P02` and we add
    `(1 + A1*k + A2*k**2 + A3*k**3 + A4*k**4) * P11`
    """
    def __init__(self):
        
        # load the data from the json file
        data = sim_data.P11_plus_P02_correction_params()
        
        # initialize
        data = data.set_index(['sigma8_z', 'b1'])
        param_names = ['b2_00', 'A1', 'A2', 'A3', 'A4']
        kws = {'use_errors':True, 'regr':'quadratic'}
        super(P11PlusP02Correction, self).__init__(param_names, data, **kws)
        
 
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
        self._sigma8_0     = self.power_lin.GetCosmology().sigma8()

        # make sure power spectrum redshift is 0
        if self.power_lin.GetRedshift() != 0.:
            raise ValueError("linear power spectrum should be at z=0")

    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    @parameter
    def f(self, val):
        """
        The growth rate, defined as the `dlnD/dlna`.
        """
        return val

    @parameter
    def power_lin(self, val):
        """
        The `pygcl.LinearPS` object defining the linear power spectrum at `z=0`
        """
        return val

    @parameter
    def z(self, val):
        """
        Desired redshift for the output power spectrum
        """
        return val
       
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
            Linear power spectrum
        z : float
            Redshift to compute the power spectrum at
        sigma8 : float
            Desired sigma8 value
        f : float
            Desired logarithmic growth rate value
        """
        # initialize the base class holding parameters
        super(SimulationP11, self).__init__(power_lin, z, sigma8_z, f)
                
        # load the data
        self._load_data()
         
    def _load_data(self):
        """
        Load the P11 simulation data
        """
        # cosmology and linear power spectrum for teppei's sims
        cosmo = pygcl.Cosmology("teppei_sims.ini", pygcl.Cosmology.CLASS)
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
        normed_power = self.power_lin(k) / self._sigma8_0**2
        factor = x*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [val, k]
        else:
            pts = np.asarray(list(itertools.product([val], k)))
        return self.interpolation_table(pts)*factor
            
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
        normed_power = self.power_lin(k) / self._sigma8_0**2
        factor = fs8_sq*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [self.z, k]
        else:
            pts = np.asarray(list(itertools.product([self.z], k)))
        return self.interpolation_table(pts)*factor

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
        
        # load the data
        self._load_data()

    def _load_data(self):
        """
        Load the simulation data
        """
        # cosmology and linear power spectrum for teppei's sims
        cosmo = pygcl.Cosmology("teppei_sims.ini", pygcl.Cosmology.CLASS)
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
        normed_power = self.power_lin(k) / self._sigma8_0**2
        factor = x*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [val, k]
        else:
            pts = np.asarray(list(itertools.product([val], k)))
        return self.interpolation_table(pts)*factor
            
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
        normed_power = self.power_lin(k) / self._sigma8_0**2
        factor = fs8_sq*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [self.z, k]
        else:
            pts = np.asarray(list(itertools.product([self.z], k)))
        return self.interpolation_table(pts)*factor

    
    
