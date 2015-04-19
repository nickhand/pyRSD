from .. import pygcl, numpy as np, data as sim_data
from .. import data as sim_data 
from . import tools, INTERP_KMIN, INTERP_KMAX

import bisect
import itertools
import pandas as pd
from sklearn.gaussian_process import GaussianProcess

class SimInterpolator(object):
    """
    Class to interpolate simulation data as a function of bias and redshift
    """
    
    def __init__(self, return_nan=False, corr_model="linear", spline_kwargs={}, 
                columns=(), use_ratio=False):
        
        # store the sim specific variables
        self.columns = columns
        self.use_ratio = use_ratio
        
        # whether to return NaNs outside bounds, rather than raising exception
        self.return_nan = return_nan
        
        # GP correlation model, if None, use RSDSplines
        self.corr_model = corr_model
        self.spline_kwargs = spline_kwargs
        
        # setup sim data
        self._setup_sim_results()
        
        # setup the interpolator
        self._setup_interpolator()
         
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
        
    #---------------------------------------------------------------------------
    def _setup_interpolator(self):
        """
        Setup the backend interpolator, either a Gaussian Process or RSDSpline
        """
        if self.corr_model is not None:
            self._setup_gps()
        else:
            self._setup_splines()
        
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

    #---------------------------------------------------------------------------
    def _setup_splines(self):
        """
        Setup the RSDSplines as a function of bias at each redshift
        """
        self.splines = {}
        
        # loop over each redshift
        for z in self.redshifts:

            self.splines[z] = {}
            frame = self.data.xs(z)
            biases = np.array(frame.index)
            
            # setup the spline
            for col in self.columns:
                sim_data = frame[col]
                if self.use_ratio:
                    sim_data /= biases
                self.splines[z][col] = tools.RSDSpline(biases, np.array(sim_data), **self.spline_kwargs)
        
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
    
    #---------------------------------------------------------------------------
    def _check_tuple_length(self, val):
        """
        Check for a return tuple length of one
        """
        try:
            if len(val) == 1:
                return val[0]
            else:
                return val
        except:
            return val
    #---------------------------------------------------------------------------
    def __call__(self, bias, z, col=None):
        """
        Evaluate at specified bias and redshift
        """
        if col is None:
            columns = self.columns
        else:
            columns = [col]
            
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
            for col in columns:
                
                if self.corr_model is not None:
                    f = getattr(self.gps[zi][col], 'predict')
                    value = f(np.atleast_2d(bias).T)
                else:
                    value = self.splines[zi][col](bias)
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
            for i, col in enumerate(columns):
                value = (1 - w)*params[0][i] + w*params[1][i]
                toret += (self._check_scalar(value),)

            return self._check_tuple_length(toret)
    #---------------------------------------------------------------------------

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
                 
    # the nonlinear bias values at z = 0.989
    params_z2 = {'2.32' : (1.75, 0.80), 
                 '3.17' : (4.77, 3.15), 
                 '4.64' : (12.80, 10.80)}
    sim_results = [(0., params_z0), (0.509, params_z1), (0.989, params_z2)]
    
    #---------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):

        kwargs['columns'] = ['b2_00', 'b2_01']
        kwargs['use_ratio'] = True
        kwargs['corr_model'] = None
        kwargs['spline_kwargs'] = {'extrap' : True, 'k' : 1}
        super(NonlinearBiasFits, self).__init__(*args, **kwargs)
    
    #---------------------------------------------------------------------------
   

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

    #---------------------------------------------------------------------------


#-------------------------------------------------------------------------------
class SimInterpolatorofBiasRedshift(object):
    """
    Class to interpolate simulation data as a function of bias and redshift, 
    using `GaussianProcess` class from `sklearn`
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
# class NonlinearBiasFits(SimInterpolatorofBiasRedshift):
#     """
#     Class implementing nonlinear bias fits from Vlah et al. 2013
#     """
#     # the nonlinear bias values at z = 0
#     params_z0 = {'1.18' : (-0.39, -0.45), 
#                  '1.47' : (-0.08, -0.35), 
#                  '2.04' : (0.91, 0.14), 
#                  '3.05' : (3.88, 2.00)}
#     
#     # the nonlinear bias values at z = 0.509
#     params_z1 = {'1.64' : (0.18, -0.20), 
#                  '2.18' : (1.29, 0.48), 
#                  '3.13' : (4.48, 2.60), 
#                  '4.82' : (12.70, 9.50)}
#                  
#     # the nonlinear bias values at z = 0.989
#     params_z2 = {'2.32' : (1.75, 0.80), 
#                  '3.17' : (4.77, 3.15), 
#                  '4.64' : (12.80, 10.80)}
#     sim_results = {'0':params_z0, '0.509':params_z1, '0.989':params_z2}
#     
#     #---------------------------------------------------------------------------
#     def __init__(self):
# 
#         cols = ['b2_00', 'b2_01']
#         super(NonlinearBiasFits, self).__init__(cols, use_bias_ratio=True)
#     
#     #---------------------------------------------------------------------------
#   
# 
# #-------------------------------------------------------------------------------
# class SigmavFits(SimInterpolatorofBiasRedshift):
#     """
#     The halo velocity dispersion as measured from simulations, as computed
#     from Figure 7 of Vlah et al. 2013. 
#     
#     These are computed in km/s as:
#     
#     :math: \sigma_v(z=0) * D(z) * f(z) * H(z) / h where 
#     :math: \sigma_v(z=0) ~ 6 Mpc/h.
#     """
#     # the values at z = 0
#     params_z0 = {'1.18' : (306.), 
#                  '1.47' : (302.), 
#                  '2.04' : (296.), 
#                  '3.05' : (288.)}
#     
#     # the values at z = 0.509
#     params_z1 = {'1.64' : (357.), 
#                  '2.18' : (352.), 
#                  '3.13' : (346.), 
#                  '4.82' : (339.)}
#                  
#     # the values at z = 0.509
#     params_z2 = {'2.32' : (340.), 
#                  '3.17' : (337.), 
#                  '4.64' : (330.)}
#     sim_results = {'0':params_z0, '0.509':params_z1, '0.989':params_z2}
#     
#     #---------------------------------------------------------------------------
#     def __init__(self):
# 
#         cols = ['sigmav']
#         super(SigmavFits, self).__init__(cols, use_bias_ratio=False)
#         
#     #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class StochasticityLogModel(SimInterpolatorofBiasRedshift):
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
        super(LambdaStochasticityLogFits, self).__init__(cols, use_bias_ratio=False)

    #---------------------------------------------------------------------------
    def __call__(self, k, b1, z):
        """
        Return the stochasticity
        """
        A0, A1 = SimInterpolatorofBiasRedshift.__call__(self, b1, z)
        return A0 + A1*np.log(k)
        
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Gaussian Process Fits
#-------------------------------------------------------------------------------
class StochasticityGPModel(tools.InterpolationTable):
    """
    Class implementing the fits to the scale-dependent stochasticity, Lambda,
    using a Gaussian Process model based on simulation data
    """
    
    k_interp = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 100)
    b1_interp = np.linspace(1.1, 5.5, 100)
    
    def __init__(self, z, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The redshift to measure the stochasticity at. The interpolation
            table must be recomputed whenever `z` changes
        interpolated : bool, optional
            If `True`, return results as a function of bias from an 
            interpolation table
        """        
        # store the redshift
        self._z = z
        
        # load the sim GP
        self.gp = sim_data.stochasticity_gp_model()
        
        # initialize the base class
        super(StochasticityGPModel, self).__init__(self.b1_interp, self.k_interp, interpolated)
                    
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        The redshift
        """
        return self._z
        
    @z.setter
    def z(self, val):
        self._z = val
        if self.interpolated:
            self.make_interpolation_table()
        
    #---------------------------------------------------------------------------
    def evaluate_table(self, k, b1, return_error=False):
        """
        The stochasticity as computed from simulations using a Gaussian Process
        fit
        """
        x = np.vstack((np.ones(len(k))*self.z, np.ones(len(k))*b1, k)).T
        if return_error:
            lam, sig_sq = self.gp.predict(x, eval_MSE=True)
        else:
            lam = self.gp.predict(x)

        if return_error:
            return lam, sig_sq**0.5
        else:
            return lam
    
    #---------------------------------------------------------------------------
    def __call__(self, k, b1, return_error=False):
        
        if return_error or not self.interpolated:
            return self.evaluate_table(k, b1, return_error)
        else:
            return tools.InterpolationTable.__call__(self, k, b1)
    #---------------------------------------------------------------------------
        
#-------------------------------------------------------------------------------
class PhmResidualGPModel(tools.InterpolationTable):
    """
    Class implementing the fits to the residual of Phm, modeled with a Pade
    expansion, using a Gaussian Process model based on simulation data
    """
    
    k_interp = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 100)
    b1_interp = np.linspace(1.1, 5.5, 100)
    
    def __init__(self, z, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The redshift to measure the stochasticity at. The interpolation
            table must be recomputed whenever `z` changes
        interpolated : bool, optional
            If `True`, return results as a function of bias from an 
            interpolation table
        """        
        # store the redshift
        self._z = z
        
        # load the sim GP
        self.gp = sim_data.Phm_residual_gp_model()
        
        # initialize the base class
        super(PhmResidualGPModel, self).__init__(self.b1_interp, self.k_interp, interpolated)
                    
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        The redshift
        """
        return self._z
        
    @z.setter
    def z(self, val):
        self._z = val
        if self.interpolated:
            self.make_interpolation_table()
        
    #---------------------------------------------------------------------------
    def evaluate_table(self, k, b1, return_error=False):
        """
        The stochasticity as computed from simulations using a Gaussian Process
        fit
        """
        x = np.vstack((np.ones(len(k))*self.z, np.ones(len(k))*b1, k)).T
        if return_error:
            Phm, sig_sq = self.gp.predict(x, eval_MSE=True)
        else:
            Phm = self.gp.predict(x)

        if return_error:
            return Phm, sig_sq**0.5
        else:
            return Phm
    
    #---------------------------------------------------------------------------
    def __call__(self, k, b1, return_error=False):
        
        if return_error or not self.interpolated:
            return self.evaluate_table(k, b1, return_error)
        else:
            return tools.InterpolationTable.__call__(self, k, b1)
    #---------------------------------------------------------------------------
        
#-------------------------------------------------------------------------------
class HaloP00(object):
    """
    Class to compute P00 for halos as a function of linear bias `b1` and 
    redshift `z`
    """
    def __init__(self, HaloZelP00, interpolated=False):
        """
        Initialize with a `pygcl.HaloZeldovichP00` object
        """
        # doesnt make a copy -- just a reference so that the redshift will 
        # be updated
        self.HaloZelP00 = HaloZelP00
        self.HaloZelP00.interpolated = interpolated
        
        # initialize the GP classes
        self.gp_Phm   = PhmResidualGPModel(self.z, interpolated)
        self.gp_stoch = StochasticityGPModel(self.z, interpolated)
        
        # store the interpolation variable
        self._interpolated = interpolated
        
    #---------------------------------------------------------------------------
    @property
    def interpolated(self):
        """
        If `True`, return results using the interpolation table
        """
        return self._interpolated

    @interpolated.setter
    def interpolated(self, val):
        if hasattr(self, '_interpolated') and self._interpolated == val:
            return
            
        self._interpolated           = val
        self.gp_Phm.interpolated     = val
        self.gp_stoch.interpolated   = val
        self.HaloZelP00.interpolated = val
            
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        The redshift, taken from the `P00_model`; 
        """
        return self.HaloZelP00.z
        
    @z.setter
    def z(self, val):
        if hasattr(self, '_z') and self._z == val:
            return
            
        self.HaloZelP00.z = val
        self.gp_Phm.z     = val
        self.gp_stoch.z   = val
        
    #---------------------------------------------------------------------------
    def Pmm(self, k):
        """
        The dark matter density auto correlation as computed from 
        the Halo Zeldovich power spectrum 
        """
        return self.HaloZelP00(k)
        
    #---------------------------------------------------------------------------
    def Phm(self, k, b1, return_error=False):
        """
        The halo-matter cross correlation at the bias specified by `b1`, 
        as computed from the Gaussian Process fit
        """
        toret = self.gp_Phm(k, b1, return_error)
        Pzel = b1*self.HaloZelP00.zeldovich_power(k)
        
        if return_error:
            return toret[0] + Pzel, toret[1]**0.5
        else:
            return toret + Pzel
        
    #---------------------------------------------------------------------------
    def stochasticity(self, k, b1, return_error=False):
        """
        The stochasticity as computed from simulations using a Gaussian Process
        fit
        """
        toret = self.gp_stoch(k, b1, return_error)
        if return_error:
            return toret[0], toret[1]**0.5
        else:
            return toret
    
    #---------------------------------------------------------------------------
    def __call__(self, k, b1, return_error=False):
        """
        Return the halo P00 power, optionally returning the error as computed
        from the Gaussian Process fit
        """        
        # the three parts we need
        Pmm = self.Pmm(k)
        Phm = self.Phm(k, b1, return_error)
        lam = self.stochasticity(k, b1, return_error)
        
        if return_error:
            toret =  2*b1*Phm[0] - b1**2*Pmm + lam[0]
            err = np.sqrt((2*b1*Phm[1])**2 + lam[1]**2)
            return toret, err
        else:
            return 2*b1*Phm - b1**2*Pmm + lam
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
class InterpDarkMatterPowerMoment(object):
    """
    A base class for dark matter power moments computed from interpolating
    simulation results
    """
    def __init__(self, power_lin, z, sigma8, f):

        # store the input arguments
        self._power_lin = power_lin
        self._cosmo     = self._power_lin.GetCosmology()

        # make sure power spectrum redshift is 0
        msg = "input linear power spectrum must be defined at z = 0"
        assert self._power_lin.GetRedshift() == 0., msg

        # set the initial redshift, sigma8 
        self.z = z
        self.sigma8 = sigma8
        self.f = f


    #---------------------------------------------------------------------------
    @property
    def f(self):
        """
        The growth rate, defined as the `dlnD/dlna`. 

        If the parameter has not been explicity set, it defaults to the value
        at `self.z`
        """
        try:
          return self._f
        except AttributeError:
            return self.cosmo.f_z(self.z)

    @f.setter
    def f(self, val):
        self._f = val
      
    #---------------------------------------------------------------------------
    @property
    def power_lin(self):
        """
        Linear power spectrum object
        """
        return self._power_lin

    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        """
        The cosmology of the input linear power spectrum
        """
        return self._cosmo

    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        Redshift to compute the integrals at
        """
        return self._z

    @z.setter
    def z(self, val):
        self._z = val
        del self.D

    #---------------------------------------------------------------------------
    @property
    def D(self):
        """
        The growth function, normalized to unity at z = 0
        """
        try:
            return self._D
        except AttributeError:
            self._D = self.cosmo.D_z(self.z)
            return self._D

    @D.deleter
    def D(self):
        try:
            del self._D
        except AttributeError:
            pass

    #---------------------------------------------------------------------------        
    @property
    def sigma8(self):
        """
        Sigma_8 at `z=0` to compute the spectrum at, which gives the 
        normalization of the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        self._sigma8 = val
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class DarkMatterP01(InterpDarkMatterPowerMoment, tools.InterpolationTable):
    """
    Dark matter model for mu^4 term of P11. Computed by interpolating 
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
        InterpDarkMatterPowerMoment.__init__(self, power_lin, z, sigma8, f)

        # make sure power spectrum is no-wiggle
        if self.cosmo.GetTransferFit() != pygcl.Cosmology.EH_NoWiggle:
            raise ValueError("Interpolated sim results require the no-wiggle power spectrum")

        # load the data
        self._load_data()

        # initialize the base class
        index_1 = self.data.index.get_level_values('fs8_sq').unique()
        index_2 = self.data.index.get_level_values('k').unique()
        tools.InterpolationTable.__init__(self, index_1, index_2, True)

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
        data = [sim_data.P01_mu2_z_0_000(), sim_data.P01_mu2_z_0_509(), sim_data.P01_mu2_z_0_989()]
        interp_vars = [cosmo.f_z(z)*cosmo.sigma8()**2 for z in redshifts]

        # make the data frame
        k = data[0][:,0]
        index_tups = list(itertools.product(interp_vars, k))
        index = pd.MultiIndex.from_tuples(index_tups, names=['fs8_sq', 'k'])
        d = []
        for i, x in enumerate(data):
            d += list(x[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(x[:,0]) * cosmo.f_z(redshifts[i])))
        self.data = pd.DataFrame(data=d, index=index, columns=['P01'])


    #---------------------------------------------------------------------------
    def evaluate_table(self, k, fs8_sq):
        """
        Return the normalized P11 at this value of (f*sigma8)^2
        """
        if fs8_sq not in self.data.index.get_level_values('fs8_sq').unique():
            raise ValueError("Cannot evaluate P11 at this value of (f*sigma8)^2")
        return self.data.xs(fs8_sq).P01

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of (f*sigma8)^2 by assuming the shape of
        the normalized power spectrum is the sames
        """
        if x < self.index_1[0]:
            index = -1
        else:
            index = 0

        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = x*normed_power
        return self.table[index](k)*factor

    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Evaluate P11 at the redshift `self.z` and the specified `k`
        """
        fs8_sq = self.f*self.sigma8**2

        # extrapolate?
        if fs8_sq < self.index_1[0] or fs8_sq > self.index_1[-1]:
            return self._extrapolate(fs8_sq, k)

        # evaluate the interpolation table at this (f*sigma8)^2
        toret = tools.InterpolationTable.__call__(self, k, fs8_sq)

        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power

        return toret*factor
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
class DarkMatterP11(InterpDarkMatterPowerMoment, tools.InterpolationTable):
    """
    Dark matter model for mu^4 term of P11. Computed by interpolating 
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
        InterpDarkMatterPowerMoment.__init__(self, power_lin, z, sigma8, f)
        
        # make sure power spectrum is no-wiggle
        if self.cosmo.GetTransferFit() != pygcl.Cosmology.EH_NoWiggle:
            raise ValueError("Interpolated sim results require the no-wiggle power spectrum")
        
        # load the data
        self._load_data()
      
        # initialize the base class
        index_1 = self.data.index.get_level_values('fs8_sq').unique()
        index_2 = self.data.index.get_level_values('k').unique()
        tools.InterpolationTable.__init__(self, index_1, index_2, True)
    
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
        self.data = pd.DataFrame(data=d, index=index, columns=['P11'])
      
      
    #---------------------------------------------------------------------------
    def evaluate_table(self, k, fs8_sq):
        """
        Return the normalized P11 at this value of (f*sigma8)^2
        """
        if fs8_sq not in self.data.index.get_level_values('fs8_sq').unique():
            raise ValueError("Cannot evaluate P11 at this value of (f*sigma8)^2")
        return self.data.xs(fs8_sq).P11

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of (f*sigma8)^2 by assuming the shape of
        the normalized power spectrum is the sames
        """
        if x < self.index_1[0]:
            index = -1
        else:
            index = 0
        
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = x*normed_power
        return self.table[index](k)*factor
            
    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Evaluate P11 at the redshift `self.z` and the specified `k`
        """
        fs8_sq = (self.f*self.sigma8)**2
        
        # extrapolate?
        if fs8_sq < self.index_1[0] or fs8_sq > self.index_1[-1]:
            return self._extrapolate(fs8_sq, k)
            
        # evaluate the interpolation table at this (f*sigma8)^2
        toret = tools.InterpolationTable.__call__(self, k, fs8_sq)
        
        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power

        return toret*factor
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class DarkMatterPdv(InterpDarkMatterPowerMoment, tools.InterpolationTable):
    """
    Dark matter model for density -- velocity divergence cross power spectrum
    Pdv, computed by interpolating simulation data as a function 
    of f*sigma8^2
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
        InterpDarkMatterPowerMoment.__init__(self, power_lin, z, sigma8, f)
        
        # make sure power spectrum is no-wiggle
        if self.cosmo.GetTransferFit() != pygcl.Cosmology.EH_NoWiggle:
            raise ValueError("Interpolated sim results require the no-wiggle power spectrum")
        
        # load the data
        self._load_data()
      
        # initialize the base class
        index_1 = self.data.index.get_level_values('fs8_sq').unique()
        index_2 = self.data.index.get_level_values('k').unique()
        tools.InterpolationTable.__init__(self, index_1, index_2, True)

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
        data = [sim_data.Pdv_mu0_z_0_000(), sim_data.Pdv_mu0_z_0_509(), sim_data.Pdv_mu0_z_0_989()]
        interp_vars = [cosmo.f_z(z)*cosmo.sigma8()**2 for z in redshifts]

        # make the data frame
        k = data[0][:,0]
        index_tups = list(itertools.product(interp_vars, k))
        index = pd.MultiIndex.from_tuples(index_tups, names=['fs8_sq', 'k'])
        d = []
        for i, x in enumerate(data):
            d += list(x[:,1] / (cosmo.D_z(redshifts[i])**2 * Plin(x[:,0]) * cosmo.f_z(redshifts[i])))
        self.data = pd.DataFrame(data=d, index=index, columns=['Pdv'])

    #---------------------------------------------------------------------------
    def evaluate_table(self, k, fs8_sq):
        """
        Return the normalized P11 at this value of (f*sigma8)^2
        """
        if fs8_sq not in self.data.index.get_level_values('fs8_sq').unique():
            raise ValueError("Cannot evaluate Pdv at this value of f*sigma8^2")
        return self.data.xs(fs8_sq).Pdv

    #---------------------------------------------------------------------------
    def _extrapolate(self, x, k):
        """
        Extrapolate out of the range of (f*sigma8)^2 by assuming the shape of
        the normalized power spectrum is the sames
        """
        if x < self.index_1[0]:
            index = -1
        else:
            index = 0

        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = x*normed_power
        return self.table[index](k)*factor

    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Evaluate P11 at the redshift `self.z` and the specified `k`
        """
        fs8_sq = self.f*self.sigma8**2

        # extrapolate?
        if fs8_sq < self.index_1[0] or fs8_sq > self.index_1[-1]:
            return self._extrapolate(fs8_sq, k)

        # evaluate the interpolation table at this (f*sigma8)^2
        toret = tools.InterpolationTable.__call__(self, k, fs8_sq)

        normed_power = self.D**2 * self.power_lin(k) / self.cosmo.sigma8()**2
        factor = fs8_sq*normed_power

        return toret*factor
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
# Halo Zeldovich Power Spectra 
#-------------------------------------------------------------------------------
class HaloZeldovichPS(tools.InterpolationTable):
    """
    Base class to represent a Halo Zeldovich power spectrum
    """
    k_interp = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 100)
    s8_interp = np.linspace(0.1, 2.5, 100)
    
    def __init__(self, z, sigma8, interpolated=False):
        """
        Parameters
        ----------
        z : float
            The desired redshift to compute the power at
        sigma8 : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """
        self._z = z
        self._sigma8 = sigma8
        
        # initialize the InterpolationTable
        super(HaloZeldovichPS, self).__init__(self.s8_interp, self.k_interp, interpolated)
        
    #---------------------------------------------------------------------------
    def evaluate_table(self, k, s8):
        """
        Evaluate the Zeldovich power for storing in the interpolation table
        """
        self.Pzel.SetSigma8(s8)
        return self.zeldovich_power(k, ignore_interpolated=True)
        
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        """
        The cosmology of the input linear power spectrum
        """
        return self._cosmo
        
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        Redshift to compute the integrals at
        """
        return self._z
    
    @z.setter
    def z(self, val):
        if hasattr(self, '_z') and self._z == val:
            return
             
        self._z = val
        self.Pzel.SetRedshift(val)
        del self.sigma8_z
        
        if self.interpolated:
            self.make_interpolation_table()
    
    #---------------------------------------------------------------------------        
    @property
    def sigma8(self):
        """
        Sigma_8 at `z=0` to compute the spectrum at, which gives the 
        normalization of the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        if hasattr(self, '_sigma8') and self._sigma8 == val:
            return
            
        self._sigma8 = val
        self.Pzel.SetSigma8(val)
        del self.sigma8_z
        
    #---------------------------------------------------------------------------
    @property
    def A0(self):
        """
        Returns the A0 radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are power [(h/Mpc)^3]
        """
        return 750*(self.sigma8_z/0.8)**3.75

    #---------------------------------------------------------------------------
    @property
    def R(self):
        """
        Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 26*(self.sigma8_z/0.8)**0.15

    #---------------------------------------------------------------------------
    @property
    def R1(self):
        """
        Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 3.33*(self.sigma8_z/0.8)**0.88

    #---------------------------------------------------------------------------
    @property
    def R1h(self):
        """
        Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 3.87*(self.sigma8_z/0.8)**0.29
    
    #---------------------------------------------------------------------------
    @property
    def R2h(self):
        """
        Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return 1.69*(self.sigma8_z/0.8)**0.43

    #---------------------------------------------------------------------------
    def compensation(self, k):
        """
        The compensation function F(k) that causes the broadband power to go
        to zero at low k, in order to conserver mass/momentum

        The functional form is given by 1 - 1 / (1 + k^2 R^2), where R(z) 
        is given by Eq. 4 in arXiv:1501.07512.
        """
        return 1. - 1./(1. + (k*self.R)**2)
    
    #---------------------------------------------------------------------------
    @property
    def sigma8_z(self):
        """
        Return sigma8(z), normalized to the desired sigma8 at z = 0
        """
        try:
            return self._sigma8_z
        except AttributeError:
            self._sigma8_z = self.sigma8 * (self.cosmo.Sigma8_z(self.z) / self.cosmo.sigma8())
            return self._sigma8_z
    
    @sigma8_z.deleter
    def sigma8_z(self):
        if hasattr(self, '_sigma8_z'): del self._sigma8_z

    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Return the total power, equal to the Zeldovich power + broadband 
        correction
        """
        return self.broadband_power(k) + self.zeldovich_power(k)

    #---------------------------------------------------------------------------
    def broadband_power(self, k):
        """
        The broadband power correction in units of (Mpc/h)^3

        The functional form is given by: 

        P_BB = A0 * F(k) * [ (1 + (k*R1)^2) / (1 + (k*R1h)^2 + (k*R2h)^4) ], 
        as given by Eq. 1 in arXiv:1501.07512.
        """
        F = self.compensation(k)
        return F*self.A0*(1 + (k*self.R1)**2) / (1 + (k*self.R1h)**2 + (k*self.R2h)**4)
    
    #---------------------------------------------------------------------------
    def zeldovich_power(self, k, ignore_interpolated=False):
        """
        Return the Zel'dovich power at the specified `k`
        """
        if self.interpolated and not ignore_interpolated:
            return tools.InterpolationTable.__call__(self, k, self.sigma8)
        else:
            return self.Pzel(k)
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class HaloZeldovichP00(HaloZeldovichPS):
    """
    Halo Zel'dovich P00
    """ 
    def __init__(self, cosmo, z, sigma8, interpolated=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8 : float
            The desired sigma8 to compute the power at
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP00(cosmo, z)
        
        # save the cosmology too
        self._cosmo = cosmo
        
        # initialize the base class
        super(HaloZeldovichP00, self).__init__(z, sigma8, interpolated)
            
    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Return the full Halo Zeldovich P00, optionally using the interpolation
        table to compute the Zeldovich part
        """
        # make sure sigma8 is set properly
        if self.Pzel.GetSigma8() != self.sigma8:
            self.Pzel.SetSigma8(self.sigma8)
            
        if not self.interpolated:
            return self.broadband_power(k) + self.zeldovich_power(k)
        else:
            
            Pzel = tools.InterpolationTable.__call__(self, k, self.sigma8)
            return self.broadband_power(k) + Pzel
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
class HaloZeldovichP01(HaloZeldovichPS):
    """
    Halo Zel'dovich P01
    """ 
    def __init__(self, cosmo, z, sigma8, f, interpolated=False):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            The cosmology object
        z : float
            The desired redshift to compute the power at
        sigma8 : float
            The desired sigma8 to compute the power at
        f : float
            The desired logarithmic growth rate
        interpolated : bool, optional
            Whether to return Zel'dovich power from the interpolation table
            using sigma8 as the index variable
        """   
        # initialize the Pzel object
        self.Pzel = pygcl.ZeldovichP01(cosmo, z)
        
        # save the cosmology too
        self._cosmo = cosmo
        
        # and the growth rate
        self._f = f
        
        # initialize the base class
        super(HaloZeldovichP01, self).__init__(z, sigma8, interpolated)
        
    #---------------------------------------------------------------------------       
    @property
    def f(self):
        """
        The logarithmic growth rate
        """
        return self._f

    @f.setter
    def f(self, val):
        if hasattr(self, '_f') and self._f == val:
            return
            
        self._f = val
        
    #--------------------------------------------------------------------------- 
    def broadband_power(self, k):
        """
        The broadband power correction for P01 in units of (Mpc/h)^3

        This is basically the derivative of the broadband band term for P00, taken
        with respect to lna
        """
        F = self.compensation(k)

        # store these for convenience
        norm = 1 + (k*self.R1h)**2 + (k*self.R2h)**4
        C = (1. + (k*self.R1)**2) / norm

        # 1st term of tot deriv
        term1 = self.dA0_dlna*F*C;

        # 2nd term
        term2 = self.A0*C * (2*k**2*self.R*self.dR_dlna) / (1 + (k*self.R)**2)**2

        # 3rd term
        term3_a = (2*k**2*self.R1*self.dR1_dlna) / norm
        term3_b = -(1 + (k*self.R1**2)) / norm**2 * (2*k**2*self.R1h*self.dR1h_dlna + 4*k**4*self.R2h**3*self.dR2h_dlna)
        term3 = self.A0*F * (term3_a + term3_b)
        return term1 + term2 + term3
    
    #---------------------------------------------------------------------------
    @property
    def dA0_dlna(self):
        return self.f * 3.75 * self.A0

    #---------------------------------------------------------------------------
    @property
    def dR_dlna(self):
        return self.f * 0.15 * self.R

    #---------------------------------------------------------------------------
    @property
    def dR1_dlna(self):
        return self.f * 0.88 * self.R1

    #---------------------------------------------------------------------------
    @property
    def dR1h_dlna(self):
        return self.f * 0.29 * self.R1h

    #---------------------------------------------------------------------------
    @property
    def dR2h_dlna(self):
        return self.f * 0.43 * self.R2h
            
    #---------------------------------------------------------------------------
    def __call__(self, k):
        """
        Return the full Halo Zeldovich P01, optionally using the interpolation
        table to compute the Zeldovich part
        
        Note
        ----
        The true Zel'dovich term is 2*f times the result returned by
        `self.zeldovich_power`
        """
        # make sure sigma8 is set properly
        if self.Pzel.GetSigma8() != self.sigma8:
            self.Pzel.SetSigma8(self.sigma8)
            
        if not self.interpolated:
            return self.broadband_power(k) + 2*self.f*self.zeldovich_power(k)
        else:
            
            Pzel = tools.InterpolationTable.__call__(self, k, self.sigma8)
            return self.broadband_power(k) + 2*self.f*Pzel
    #---------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
        
    
