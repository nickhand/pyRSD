from .. import numpy as np, data as sim_data
from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from .simulation import GaussianProcessSimulationData
from . import tools, INTERP_KMIN, INTERP_KMAX

import itertools
from scipy.interpolate import UnivariateSpline as spline
import bisect 

#-------------------------------------------------------------------------------
# P_hm Models
#-------------------------------------------------------------------------------
class PhmBiasingCorrection(GaussianProcessSimulationData):
    """
    Class implementing the correction to the nonlinear biasing term K00
    in the Phm model. The correction is assumed to be a linear function
    of `k` past `k_transition = 0.15 h/Mpc`:
    
    :math: Phm_corr = A1*k + A0,
    
    where A1*(sigma8(z)/0.8)**(-3/2) depends only on b2
    """    
    #---------------------------------------------------------------------------   
    def __init__(self):
        
        # initialize the base class
        param_names = ['A0', 'A1_scaled']
        super(PhmBiasingCorrection, self).__init__(param_names, use_errors=False, regr='quadratic')

    #---------------------------------------------------------------------------
    @cached_property()
    def data(self):
        """
        Construct the `pandas` `DataFrame` holding the data, as measured
        from sims
        """
        return sim_data.Phm_biasing_correction()
        
    #---------------------------------------------------------------------------
    def transition(self, k, k_transition=0.15, b=0.05):
        """
        The transition function between the low-k and high-k values
        """
        return 0.5 + 0.5*np.tanh((k-k_transition)/b)
        
    #---------------------------------------------------------------------------
    def __call__(self, b1, z):
        """
        Return the parameters of the biasing correction, `A0` and `A1_scaled`. 
        
        Notes
        -----
        We are evaluating the correction as a function of linear bias, b1,
        and redshift, z. The sigma8_z dependence of `A1` has been taken out, 
        so we must multiply by (s8_z/0.8)^{3/2} to recover the true `A1` value
        
        Parameters
        ----------
        k : array_like
            The wavenumbers to evaluate the model at
        b1 : float
            The nonlinear bias parameter entering into P00
        z : float
            The redshift to evaluate at
            
        Returns
        -------
        A0 : float
            the constant offset
        A1_scaled : float
            The linear slope times (s8_z/0.8)^{-1/2}
            
        """
        return GaussianProcessSimulationData.__call__(self, b1, z)
       
    #---------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Stochasticity modeling
#-------------------------------------------------------------------------------
class StochasticityPadeModel(Cache):
    """
    Class implementing the stochasticity (either type A or type B), using a 
    Pade fit of the form
    
    Lambda = (A0 + A1*(k*R)**2) / (1 + (k*R)**2)
    
    Notes
    -----
    The simulation data is indexed by `sigma8` at `z` and `b1`
    """    
    #---------------------------------------------------------------------------   
    def __init__(self, use_errors=True, stoch_type="B"):
        """
        Parameters
        ----------
        stoch_type : {`A`, `B`}
            either type A or type B stochasticity. Default is `B`
        """        
        # initialize the base class
        super(StochasticityPadeModel, self).__init__()
        
        # either type A or B stochasticity
        self.stoch_type = stoch_type
        self.use_errors = use_errors
        self.param_names = ['A0', 'A1', 'R']

    #---------------------------------------------------------------------------
    @parameter
    def stoch_type(self, val):
        """
        Either type `A` or `B`
        """
        if val.lower() not in ["a", "b"]:
            raise ValueError("Stochasticity type must be either `A` or `B`")
        return val
        
    @parameter
    def use_errors(self, val):
        """
        Whether to use the errors when interpolating
        """
        return val
        
    @cached_property('stoch_type')
    def data(self):
        """
        Construct the `pandas` `DataFrame` holding the data, as measured
        from sims
        """
        if self.stoch_type.lower() == 'a':
            return sim_data.stochA_pade_model_params()
        else:
            return sim_data.stochB_pade_model_params()
                
    #---------------------------------------------------------------------------
    @cached_property('sigma8s')
    def sigma8_min(self):
        return np.amin(self.sigma8s)
        
    @cached_property('sigma8s')
    def sigma8_max(self):
        return np.amax(self.sigma8s)
        
    @cached_property('data')
    def sigma8s(self):
        return self.data.index.get_level_values('s8_z').unique()
        
    #---------------------------------------------------------------------------
    @cached_property("use_errors", "data")
    def interpolation_table(self):
        """
        Setup the backend splines for the parameters
        """
        table = {}
        
        # loop over each sigma8 value
        for s8_z in self.sigma8s:
            table[s8_z] = {}
            
            # loop over each parameter
            for name in self.param_names:
            
                d = self.data.xs(s8_z)
                
                # get the data to be interpolated, making sure to remove nulls
                y = d[name]
                null_inds = y.notnull() 
                y = np.array(y[null_inds])
        
                # X values are only b1 not sigma8 too
                X = np.array(d.index)           
  
                # check for error columns
                kwargs = {'k' : 2}
                if self.use_errors and name+'_err' in self.data:
                    dy = d[name+'_err'][null_inds]
                    kwargs['w'] = 1.0/dy

                # set the spline
                table[s8_z][name] = spline(X, y, **kwargs)

        return table
        
    #---------------------------------------------------------------------------
    def _get_parameters(self, b1, s8_z, col=None):
        """
        Return the parameter values
        """
        # return NaNs if we are out of bounds
        if s8_z < self.sigma8_min or s8_z > self.sigma8_max:
            args = (s8_z, self.sigma8_min, self.sigma8_max)
            return ValueError("Sigma8(z) vale {} is out of interpolation range [{}, {}]".format(*args))

        if s8_z in self.sigma8s:
            i = abs(self.sigma8s - s8_z).argmin()
            s8_lo = s8_hi = self.sigma8s[i]
            w = 0.
        else:
            ihi = bisect.bisect(self.sigma8s, s8_z)
            ilo = ihi - 1

            s8_lo = self.sigma8s[ilo]
            s8_hi = self.sigma8s[ihi]
            w = (s8_z - s8_lo) / (s8_hi - s8_lo) 
        
        if col is None:
            toret = []
            for col in self.param_names:
                val_lo = (1 - w)*self.interpolation_table[s8_lo][col](b1) 
                val_hi = w*self.interpolation_table[s8_hi][col](b1)
                toret.append(val_lo + val_hi)
            return toret
        else:
            val_lo = (1 - w)*self.interpolation_table[s8_lo][col](b1) 
            val_hi = w*self.interpolation_table[s8_hi][col](b1)
            return val_lo + val_hi
        
    
    #---------------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """
        Evaluate the stochasticity Pade model. If three arguments are given,
        evaluate the Pade model treating the arguments as (k, b1, s8_z). If
        only two provided, return the value of the parameters at the 
        input (b1, s8_z)
        
        Parameters
        ----------
        k : array_like
            The wavenumbers to evaluate the model at
        b1 : float
            The linear bias parameter
        s8_z : float
            The value of sigma8(z)
        """
        if len(args) == 3:
            k, b1, s8_z = args
            A0, A1, R = self._get_parameters(b1, s8_z)
            return (A0 + A1*(k*R)**2) / (1 + (k*R)**2)
        else:
            b1, s8_z = args
            col = kwargs.get('col', None)
            return self._get_parameters(b1, s8_z, col=col)
        
    #---------------------------------------------------------------------------
 
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
    def __init__(self, z, sigma8, cosmo, interpolated=False, stoch_type="B"):
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
        stoch_type : {`A`, `B`}
            Either type A or B stochasticity; default is `B`
        """             
        # initialize the Cache base class
        super(StochasticityGPModel, self).__init__()
        
        # set the parameters
        self.z            = z
        self.sigma8       = sigma8
        self.interpolated = interpolated
        self.cosmo        = cosmo
        self.stoch_type   = stoch_type
        
    #---------------------------------------------------------------------------
    @parameter
    def stoch_type(self, val):
        """
        Either type `A` or `B`
        """
        if val.lower() not in ["a", "b"]:
            raise ValueError("Stochasticity type must be either `A` or `B`")
        return val
        
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
    @cached_property("stoch_type")
    def gp(self):
        """
        The gaussian process holding the fits
        """
        if self.stoch_type.lower() == 'a':
            return sim_data.stochA_gp_model()
        else:
            return sim_data.stochB_gp_model()
            
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
    @cached_property("sigma8_z", "gp")
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