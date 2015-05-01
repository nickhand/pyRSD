from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from .. import pygcl, numpy as np, data as sim_data
from . import tools, INTERP_KMIN, INTERP_KMAX

import itertools
import pandas as pd
from sklearn.gaussian_process import GaussianProcess

#-------------------------------------------------------------------------------
# SIMULATION DATA INTERPOLATED WITH A GAUSSIAN PROCESS
#-------------------------------------------------------------------------------
class GaussianProcessSimulationData(object):
    """
    Class to interpolate simulation data as a function of bias and redshift, 
    using `GaussianProcess` class from `sklearn.gaussian_process`
    """
    def __init__(self, columns, use_bias_ratio=False):
        """
        Parameters
        ----------
        columns : list
            A list of the names of each parameter to interpolate
        use_bias_ratio : bool, optional
            If `True`, interpolate the ratio of the parameters to the linear
            bias values
        interpolated : bool, optional
            If `True`, store an interpolation table as a function of linear
            bias in order to speed up calculations
        """                
        # store the sim specific variables
        self.columns = columns
        self.use_bias_ratio = use_bias_ratio
        
        # load the simulation data
        self._load_sim_data()
        
        # setup the gaussian processes
        self._setup_GPS()
         
    #---------------------------------------------------------------------------
    # SETUP FUNCTIONS
    #---------------------------------------------------------------------------
    def _load_sim_data(self):
        """
        Construct the `pandas` `DataFrame` holding the data from sims, as 
        measured from sims
        """
        keys = []
        data = []
        for z, params in self.sim_results.iteritems():
            for bias in sorted(params.keys()):
                keys.append((float(z), float(bias)))
                data.append(params[bias])

        # make the data data frame
        index = pd.MultiIndex.from_tuples(keys, names=['z', 'b1'])
        self.data = pd.DataFrame(data, index=index, columns=self.columns).sort_index()
        
        # save the index levels
        self.redshifts = self.data.index.get_level_values('z').unique()
        self.biases = self.data.index.get_level_values('b1').unique()
        
    
    #---------------------------------------------------------------------------
    def _setup_GPS(self):
        """
        Setup the backend Gaussian processes needed to do the interpolation
        """
        self.gps = {}
        kwargs = {'corr' : 'squared_exponential', 'theta0' : [0.1, 0.1],     
                  'thetaL' : [1e-4, 1e-4], 'thetaU' : [1., 1.], 
                  'random_start' : 100, 'regr' : 'linear'}
        
        X = np.asarray(list(self.data.index.get_values()))
        for col in self.columns:
            self.gps[col] = GaussianProcess(**kwargs)
            
            if self.use_bias_ratio:
                y = self.data[col] / self.data.index.get_level_values('b1')
            else:
                 y = self.data[col]
            self.gps[col].fit(X, y)            
        
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
            return [(self.gps[col].predict([z, b1])*factor)[0] for col in self.columns]
        else:
            return [(self.gps[col].predict([z, b1])*factor)[0]]
        
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
    The halo velocity dispersion as measured from simulations, as computed
    from Figure 7 of Vlah et al. 2013. 
    
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
class StochasticityLogModel(GaussianProcessSimulationData):
    """
    Class implementing the fits to the scale-dependent stochasticity, Lambda,
    using a constant + log model: 
    
    :math: \Lambda = A0 + A1*ln(k)
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
    sim_results = {'0':params_z0, '0.509':params_z1, '0.989':params_z2}
    
    #---------------------------------------------------------------------------   
    def __init__(self):

        cols = ['constant', 'slope']
        super(StochasticityLogModel, self).__init__(cols, use_bias_ratio=False)

    #---------------------------------------------------------------------------
    def __call__(self, k, b1, z):
        """
        Return the stochasticity
        """
        A0, A1 = GaussianProcessSimulationData.__call__(self, b1, z)
        return A0 + A1*np.log(k)
        
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
# SIMULATION DATA INTERPOLATION ON A GRID
#-------------------------------------------------------------------------------
class InterpolatedSimulationData(Cache):
    """
    A base class for computing power moments from interpolated simulation data
    """
    def __init__(self, power_lin, z, sigma8, f):
        
        # initialize the Cache base class
        Cache.__init__(self)
        
        # set the parameters
        self.z         = z
        self.sigma8    = sigma8
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
    def sigma8(self, val):
        """
        Sigma_8 at `z=0` to compute the spectrum at, which gives the 
        normalization of the linear power spectrum
        """
        return val
            
    #---------------------------------------------------------------------------
    @cached_property("z")
    def D(self):
        """
        The growth function at `z`, normalized to unity at z = 0
        """
        return self.cosmo.D_z(self.z)
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class SimulationP11(InterpolatedSimulationData):
    """
    Dark matter model for the mu^4 term of P11, computed by interpolating 
    simulation data as a function of (f*sigma8)^2
    """
    
    def __init__(self, power_lin, z, sigma8, f):
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
        super(SimulationP11, self).__init__(power_lin, z, sigma8, f)
        
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
        interp_vars = [(cosmo.f_z(z)*cosmo.sigma8())**2 for z in redshifts]
      
        # make the data frame
        k = data[0][:,0]
        index_tups = list(itertools.product(interp_vars, k))
        index = pd.MultiIndex.from_tuples(index_tups, names=['fs8_sq', 'k'])
        d = []
        for i, x in enumerate(data):
            d += list(x[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(x[:,0]) * cosmo.f_z(redshifts[i])**2))
        
        # now store the results
        self.data = pd.DataFrame(data=d, index=index, columns=['P11'])
        self.interpolation_grid = {}
        self.interpolation_grid['fs8_sq'] = self.data.index.get_level_values('fs8_sq').unique()
        self.interpolation_grid['k'] = self.data.index.get_level_values('k').unique()
      
    #---------------------------------------------------------------------------
    @cached_property()
    def interpolation_table(self):
        """
        Return an interpolation table for P11, normalized by the no-wiggle
        power spectrum
        """
        # the interpolation grid points
        fs8_sqs = self.interpolation_grid['fs8_sq']
        ks = self.interpolation_grid['k']
        
        # get the grid values
        grid_vals = []
        for i, fs8_sq in enumerate(fs8_sqs):
            grid_vals += list(self.data.xs(fs8_sq).P11)
        grid_vals = np.array(grid_vals).reshape((len(fs8_sqs), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((fs8_sqs, ks), grid_vals)

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of (f*sigma8)^2 by assuming the shape of
        the normalized power spectrum is the same, i.e., just rescaling
        by the low-k amplitude
        """
        fs8_sqs = self.interpolation_grid['fs8_sq']
        if x < np.amin(fs8_sqs):
            val = np.amin(fs8_sqs)
        else:
            val = np.amax(fs8_sqs)
        
        # get the renormalization factor
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
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
        fs8_sq = (self.f*self.sigma8)**2
        
        # extrapolate?
        grid_pts = self.interpolation_grid['fs8_sq']
        if fs8_sq < np.amin(grid_pts) or fs8_sq > np.amax(grid_pts):
            return self._extrapolate(fs8_sq, k)
        
        # get the renormalization factor
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [val, k]
        else:
            pts = np.asarray(list(itertools.product([fs8_sq], k)))
        return self.interpolation_table(pts)*factor
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class SimulationPdv(InterpolatedSimulationData):
    """
    Dark matter model for density -- velocity divergence cross power spectrum
    Pdv, computed by interpolating simulation data as a function of f*sigma8^2
    """
    
    def __init__(self, power_lin, z, sigma8, f):
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
        super(SimulationPdv, self).__init__(power_lin, z, sigma8, f)
        
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
        interp_vars = [cosmo.f_z(z)*cosmo.sigma8()**2 for z in redshifts]

        # make the data frame
        k = data[0][:,0]
        index_tups = list(itertools.product(interp_vars, k))
        index = pd.MultiIndex.from_tuples(index_tups, names=['fs8_sq', 'k'])
        d = []
        for i, x in enumerate(data):
            d += list(x[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(x[:,0]) * cosmo.f_z(redshifts[i])))
        
        # now store the results
        self.data = pd.DataFrame(data=d, index=index, columns=['Pdv'])
        self.interpolation_grid = {}
        self.interpolation_grid['fs8_sq'] = self.data.index.get_level_values('fs8_sq').unique()
        self.interpolation_grid['k'] = self.data.index.get_level_values('k').unique()

    #---------------------------------------------------------------------------
    @cached_property()
    def interpolation_table(self):
        """
        Return an interpolation table for Pdv, normalized by the no-wiggle
        power spectrum
        """
        # the interpolation grid points
        fs8_sqs = self.interpolation_grid['fs8_sq']
        ks = self.interpolation_grid['k']
        
        # get the grid values
        grid_vals = []
        for i, fs8_sq in enumerate(fs8_sqs):
            grid_vals += list(self.data.xs(fs8_sq).Pdv)
        grid_vals = np.array(grid_vals).reshape((len(fs8_sqs), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((fs8_sqs, ks), grid_vals)

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of f*sigma8^2 by assuming the shape of
        the normalized power spectrum is the same, i.e., just rescaling
        by the low-k amplitude
        """
        fs8_sqs = self.interpolation_grid['fs8_sq']
        if x < np.amin(fs8_sqs):
            val = np.amin(fs8_sqs)
        else:
            val = np.amax(fs8_sqs)
        
        # get the renormalization factor
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
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
        fs8_sq = self.f*self.sigma8**2
        
        # extrapolate?
        grid_pts = self.interpolation_grid['fs8_sq']
        if fs8_sq < np.amin(grid_pts) or fs8_sq > np.amax(grid_pts):
            return self._extrapolate(fs8_sq, k)
        
        # get the renormalization factor
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power
        
        # get the pts
        if np.isscalar(k):
            pts = [val, k]
        else:
            pts = np.asarray(list(itertools.product([fs8_sq], k)))
        return self.interpolation_table(pts)*factor
    #---------------------------------------------------------------------------
    
    
#-------------------------------------------------------------------------------
# Gaussian Process Fits
#-------------------------------------------------------------------------------
class StochasticityGPModel(Cache):
    """
    Class implementing the fits to the scale-dependent stochasticity, Lambda,
    using a Gaussian Process model based on simulation data
    
    Notes
    -----
    This will be treated as a function of redshift and bias, independent 
    of cosmology
    """    
    # define the interpolation grid
    interpolation_grid = {}
    interpolation_grid['b1'] = np.linspace(1., 6., 200)
    interpolation_grid['k'] = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 200)
    
    #---------------------------------------------------------------------------
    def __init__(self, z, sigma8, cosmo, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The redshift
        sigma8: float
            The value of sigma8
        interpolated : bool, optional
            If `True`, return results from an interpolation table, otherwise,
            evaluate the Gaussian Process for each value
        """             
        # initialize the Cache base class
        Cache.__init__(self)
        
        # set the parameters
        self.z            = z
        self.sigma8       = sigma8
        self.interpolated = interpolated
        self.cosmo        = cosmo
        
        # load the sim GP
        self.gp = sim_data.stochasticity_gp_model()

    #---------------------------------------------------------------------------
    @parameter
    def interpolated(self, val):
        """
        If `True`, return the stochasticity from the interpolation table
        """
        return val
        
    @parameter
    def z(self, val):
        """
        Redshift to compute the power at
        """
        return val
        
    @parameter
    def sigma8(self, val):
        """
        The sigma8 value to compute the power at
        """
        return val
        
    @parameter
    def cosmo(self, val):
        """
        The cosmology of the input linear power spectrum
        """
        return val
        
    #---------------------------------------------------------------------------
    @cached_property("sigma8", "_normalized_sigma8_z")
    def sigma8_z(self):
        """
        Return sigma8(z), normalized to the desired sigma8 at z = 0
        """
        return self.sigma8 * self._normalized_sigma8_z

    @cached_property('z', 'cosmo')
    def _normalized_sigma8_z(self):
        """
        Return the normalized sigma8(z) from the input cosmology
        """
        return self.cosmo.Sigma8_z(self.z) / self.cosmo.sigma8()
            
    #---------------------------------------------------------------------------
    @cached_property("sigma8_z")
    def interpolation_table(self):
        """
        Evaluate the Zeldovich power for storing in the interpolation table.
        
        Notes
        -----
        This dependes on the redshift stored in the `sigma8_z` attribute and must be 
        recomputed whenever that quantity changes.
        """ 
        # the interpolation grid points
        b1s = self.interpolation_grid['b1']
        ks = self.interpolation_grid['k']
        pts = np.asarray(list(itertools.product([self.sigma8_z], b1s, ks)))
        
        # get the grid values
        grid_vals = self.gp.predict(pts, batch_size=10000)
        grid_vals = grid_vals.reshape((len(b1s), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((b1s, ks), grid_vals)
        
    #---------------------------------------------------------------------------
    def __call__(self, b1, k, return_error=False):
        """
        Evaluate the stochasticity at the specified `b1`, and `k`

        Parameters
        ----------
        b1 : float
            The value of the halo bias
        k : float, array_like
            The wavenumbers in units of `h/Mpc`
        """
        if return_error or not self.interpolated:
            if np.isscalar(k):
                pts = [self.z, b1, k]
            else:
                pts = np.asarray(list(itertools.product([self.sigma8_z], [b1], k)))
            return self.gp.predict(pts, batch_size=10000, eval_MSE=return_error)
        else:
            if np.isscalar(k):
                pts = [b1, k]
            else:
                pts = np.asarray(list(itertools.product([b1], k)))
            toret = self.interpolation_table(pts)
            return toret if len(toret) != 1 else toret[0]
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class PhmResidualGPModel(Cache):
    """
    Class implementing the fits to the Phm residual, Phm - b1*Pzel
    """    
    # define the interpolation grid
    interpolation_grid = {}
    interpolation_grid['b1'] = np.linspace(1., 6., 200)
    interpolation_grid['k'] = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 200)
    
    #---------------------------------------------------------------------------
    def __init__(self, z, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The redshift
        interpolated : bool, optional
            If `True`, return results from an interpolation table, otherwise,
            evaluate the Gaussian Process for each value
        """             
        # initialize the Cache base class
        Cache.__init__(self)
        
        # set the parameters
        self.z            = z
        self.interpolated = interpolated
        
        # load the sim GP
        self.gp = sim_data.Phm_residual_gp_model()

    #---------------------------------------------------------------------------
    @parameter
    def interpolated(self, val):
        """
        If `True`, return the stochasticity from the interpolation table
        """
        return val
        
    @parameter
    def z(self, val):
        """
        Redshift to compute the power at
        """
        return val
            
    #---------------------------------------------------------------------------
    @cached_property("z")
    def interpolation_table(self):
        """
        Evaluate the Zeldovich power for storing in the interpolation table.
        
        Notes
        -----
        This dependes on the redshift stored in the `z` attribute and must be 
        recomputed whenever that quantity changes.
        """ 
        # the interpolation grid points
        b1s = self.interpolation_grid['b1']
        ks = self.interpolation_grid['k']
        pts = np.asarray(list(itertools.product([self.z], b1s, ks)))
        
        # get the grid values
        grid_vals = self.gp.predict(pts, batch_size=10000)
        grid_vals = grid_vals.reshape((len(b1s), len(ks)))
        
        # return the interpolator
        return RegularGridInterpolator((b1s, ks), grid_vals)
        
    #---------------------------------------------------------------------------
    def __call__(self, b1, k, return_error=False):
        """
        Evaluate the Phm residual at the specified `b1`, and `k`

        Parameters
        ----------
        b1 : float
            The value of the halo bias
        k : float, array_like
            The wavenumbers in units of `h/Mpc`
        """
        if return_error or not self.interpolated:
            if np.isscalar(k):
                pts = [self.z, b1, k]
            else:
                pts = np.asarray(list(itertools.product([self.z], [b1], k)))
            return self.gp.predict(pts, batch_size=10000, eval_MSE=return_error)
        else:
            if np.isscalar(k):
                pts = [b1, k]
            else:
                pts = np.asarray(list(itertools.product([b1], k)))
            toret = self.interpolation_table(pts)
            return toret if len(toret) != 1 else toret[0]
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------