from ._cache import Cache, parameter, cached_property
from ._interpolate import RegularGridInterpolator
from .. import pygcl, numpy as np, data as sim_data
from . import tools

import itertools
import pandas as pd
from sklearn import preprocessing
import george

GEORGE_v03 = george.__version__ >= '0.3'

#-------------------------------------------------------------------------------
# simulation measurements, interpolated with a gaussian process
#-------------------------------------------------------------------------------
class GeorgeSimulationData(Cache):
    """
    Class to interpolate and predict functional data based on a training set
    of simulation data as a function of cosmological parameters

    Notes
    -----
    * this uses the `GP` class from the class `george` (see: http://dan.iel.fm/george)
    """
    def __init__(self,
                    independent_vars,
                    data,
                    theta,
                    use_errors=True,
                    dependent_col='y',
                    kernel=george.kernels.ExpSquaredKernel,
                    solver=george.BasicSolver):
        """
        Parameters
        ----------
        independent_vars : list of str
            the names of the independent variables to interpolate the data. Should be
            a column in `data`
        data : pandas.DataFrame
            the `pandas.DataFrame` holding the independent and dependent variables,
            which will be plugged into the Gaussian process
        theta : array_like
            the hyperparameters that d
        use_errors : bool, optional
            If `True`, use the errors associated with each dependent variable
        dependent_col : list of str
            the name of the dependent variable to interpolate the data. Should be
            a column in `data`
        kernel : `george.kernels.Kernel`, optional
            the kernel class to use in the Gaussian process covariance matrix
        solver : {`george.BasicSolver`, `george.HODLRSolver`}, optional
            the solver class to use when evaluating the Gaussian process
        """
        self.use_errors  = use_errors
        self.data        = data
        self.independent = independent_vars
        self.dependent   = dependent_col
        self.solver      = solver
        self.kernel      = kernel
        self.theta       = theta

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def kernel(self, val):
        """
        The kernel to use in the Gaussian process
        """
        return val

    @parameter
    def solver(self, val):
        """
        The solver to use in the Gaussian process
        """
        avail = [george.BasicSolver, george.HODLRSolver]
        if val not in avail:
            raise ValueError("the `solver` must be one of %s" %str(avail))
        return val

    @parameter
    def theta(self, val):
        """
        The Gaussian process hyperparameters
        """
        return val

    @parameter
    def data(self, val):
        """
        The `pandas.DataFrame` holding the data to interpolate
        """
        return val

    @parameter
    def use_errors(self, val):
        """
        Interpolate using the associated errors on the simulation values
        """
        return val

    @parameter
    def independent(self, val):
        """
        The names of the independent variables to interpolate
        """
        for col in val:
            if col not in self.data.columns:
                raise ValueError("the independent variable `%s` is not in the supplied data" %col)
        return val

    @parameter
    def dependent(self, val):
        """
        The name of the dependent variable to interpolate
        """
        if val not in self.data.columns:
            raise ValueError("the dependent variable `%s` is not in the supplied data" %val)
        if self.use_errors and val+'_err' not in self.data.columns:
            raise ValueError("trying to use error columns, but no error available for `%s`" %val)
        return val

    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("x")
    def xshape(self):
        """
        The shape of the second axis of `x`
        """
        x = self.x
        if x.ndim == 1: x = x.reshape(-1, 1)
        return x.shape[1]

    @cached_property("theta")
    def ndim(self):
        """
        The number of independent variables, also the size of `theta`
        """
        return len(self.theta)

    @cached_property("data")
    def x(self):
        """
        The unscaled independent variables
        """
        return self.data.loc[:,self.independent].values

    @cached_property("data")
    def y(self):
        """
        The unscaled dependent variable
        """
        return self.data.loc[:,self.dependent].values

    @cached_property("data")
    def yerr(self):
        """
        The unscaled error on the dependent variable
        """
        return self.data.loc[:,self.dependent+'_err'].values

    @cached_property("x")
    def x_scaler(self):
        """
        The class to scale the `x` attribute
        """
        x = self.x
        if x.ndim == 1: x = x.reshape(-1, 1)
        return preprocessing.StandardScaler(copy=True).fit(x)

    @cached_property("y")
    def y_scaler(self):
        """
        The class to scale the `y` attribute
        """
        return preprocessing.StandardScaler(copy=True).fit(self.y.reshape(-1, 1))

    @cached_property("x")
    def x_scaled(self):
        """
        The scaled independent variables
        """
        if self.x.ndim == 1:
            return np.squeeze(self.x_scaler.transform(self.x.reshape(-1,1)))
        else:
            return self.x_scaler.transform(self.x)

    @cached_property("y")
    def y_scaled(self):
        """
        The scaled dependent variable
        """
        return np.squeeze(self.y_scaler.transform(self.y.reshape(-1,1)))

    @cached_property("yerr")
    def yerr_scaled(self):
        """
        The scaled error on the dependent variable
        """
        return self.yerr / self.y_scaler.scale_

    @cached_property("data", "kernel", "solver")
    def gp(self):
        """
        The Gaussian process needed to do the interpolation
        """
        if self.ndim == self.xshape:
            kernel = self.kernel(self.theta, ndim=self.xshape)
        elif self.ndim == self.xshape+1:
            kernel = self.theta[0] * self.kernel(self.theta[1:], ndim=self.xshape)
        else:
            raise ValueError("size mismatch between supplied `x` variables and `theta` length")
        gp = george.GP(kernel, solver=self.solver)

        if GEORGE_v03:
            kws = {}
        else:
            kws = {'sort':False}
        if self.use_errors: kws['yerr'] = self.yerr_scaled
        gp.compute(self.x_scaled, **kws)
        return gp

    @tools.align_input
    @tools.unpacked
    def __call__(self, *args, **indep_vars):
        """
        Evaluate the Gaussian processes at the specified independent variables

        Parameters
        ----------
        indep_vars : keywords
            the independent variables to evaluate at
        """
        if len(args):
            if len(args) == 1 and len(self.independent) == 1:
                indep_vars[self.independent[0]] = args[0]
            else:
                raise ValueError("please pass variables as keywords")

        for p in self.independent:
            if p not in indep_vars:
                raise ValueError("please specify the `%s` independent variable" %p)

        # the domain point to predict
        pt = np.asarray([indep_vars[k] for k in self.independent]).T
        if pt.ndim == 1: pt = pt.reshape(1, -1)
        pt = self.x_scaler.transform(pt)

        if GEORGE_v03:
            kws = {'return_cov':False}
        else:
            kws = {'mean_only':True}
        return self.y_scaler.inverse_transform(self.gp.predict(self.y_scaled, pt, **kws))


class GeorgeSimulationDataSet(object):
    """
    Class designed to hold a `GeorgeSimulationData` instance
    for several parameters of the same model
    """
    def __init__(self, independent, dependent, data, theta, **kwargs):

        if len(dependent) != len(theta):
            raise ValueError("size mismatch between supplied parameter names and `theta`")
        self.dependents = dependent

        # initialize a Gaussian process for each dependent variable
        self._data = {}
        for i, dep in enumerate(dependent):
            self._data[dep] = GeorgeSimulationData(independent, data, theta[i], dependent_col=dep, **kwargs)

    @tools.unpacked
    def __call__(self, *args, **indep_vars):

        select = indep_vars.pop('select', None)
        # determine which parameters we are returning
        if select is None:
            select = self.dependents
        elif isinstance(select, str):
            select = [select]
        else:
            raise ValueError("do not understand `select` keyword")

        return [self._data[par](*args, **indep_vars) for par in select]


class Pmu4ResidualCorrection(GeorgeSimulationData):
    """
    Return the prediction for the Pmu4 model residual correction
    """
    def __init__(self):
        #theta = [11.76523097, 7.63002238, 3.74838973, 0.84367439]
        #independent = ['sigma8_z', 'b1', 'k']

        theta = [33.66747949, 3.95336447, 1.74027224, 0.62058417] # with f
        independent = ['f', 'sigma8_z', 'b1', 'k']
        data = sim_data.Pmu4_correction_data()
        super(Pmu4ResidualCorrection, self).__init__(independent, data, theta, use_errors=True)

class Pmu2ResidualCorrection(GeorgeSimulationData):
    """
    Return the prediction for the Pmu2 model residual correction
    """
    def __init__(self):
        #theta = [7.2492907, 4.48197495, 3.0182625, 0.67960878]
        #independent = ['sigma8_z', 'b1', 'k']

        theta = [11.84812224, 4.15569036, 1.26297742, 1.03950439] # with f
        independent = ['f', 'sigma8_z', 'b1', 'k']
        data = sim_data.Pmu2_correction_data()
        super(Pmu2ResidualCorrection, self).__init__(independent, data, theta, use_errors=True)

class VelocityDispersionFits(GeorgeSimulationData):
    """
    Return the halo velocity dispersion in Mpc/h, as measured from the
    runPB simulations, as a function of sigma8(z) and b1
    """
    def __init__(self):

        theta = [0.48087061,  0.21521814,  0.45073149]
        data = sim_data.velocity_dispersion_data()
        data['sigma_v'] /= data['f']
        data['sigma_v_err'] = 1e-5 * data['sigma_v']

        independent = ['sigma8_z', 'b1']
        kws = {'use_errors':True, 'dependent_col':'sigma_v'}
        super(VelocityDispersionFits, self).__init__(independent, data, theta, **kws)

class NonlinearBiasFits(GeorgeSimulationDataSet):
    """
    Return the nonlinear biases b2_00 and b2_01 as a function of
    sigma8(z) and b1
    """
    def __init__(self):

        data = sim_data.vlah_nonlinear_bias_fits()
        independent = ['b1']
        dependent = ['b2_00_a', 'b2_00_b', 'b2_00_c', 'b2_00_d', 'b2_01_a', 'b2_01_b']
        theta = [None]*6

        # best-fit thetas for b2_00 (see 1ad8705b0 in SimCalibrations/NonlinearBiases)
        theta[0] = [ 114.83271418,   45.4136823 ] # b2_00_a
        theta[1] = [ 78.74803514,  13.7686103 ]   # b2_00_b
        theta[2] = [ 519.43798005,   73.94048349] # b2_00_c
        theta[3] = [ 31.32395946,  11.08334321]   # b2_00_d

        # best-fit thetas for b2_01 (see 3084b38de in SimCalibrations/NonlinearBiases)
        theta[-2] = [ 7.48162433,  0.87971185]   # b2_01_a
        theta[-1] = [ 14.06494712,  12.08894434] # b2_01_b

        super(NonlinearBiasFits, self).__init__(independent, dependent, data, theta, use_errors=True)

class AutoStochasticityFits(GeorgeSimulationData):
    """
    Return the prediction for the auto stochasticity
    """
    def __init__(self):

        theta = [0.56384418, 0.76723559, 0.49555955, 5.7815692]
        data = sim_data.auto_stochasticity_data()
        independent = ['sigma8_z', 'b1', 'k']

        super(AutoStochasticityFits, self).__init__(independent, data, theta, use_errors=True)

class CrossStochasticityFits(GeorgeSimulationData):
    """
    Return the prediction for the cross stochasticity
    """
    def __init__(self):

        theta = [2.42796735, 0.59461745, 3.75844384, 1.5391186, 11.53257307]
        data = sim_data.cross_stochasticity_data()
        independent = ['sigma8_z', 'b1_1', 'b1_2', 'k']

        super(CrossStochasticityFits, self).__init__(independent, data, theta, use_errors=True)

#-------------------------------------------------------------------------------
# simulation data interpolated onto a grid
#-------------------------------------------------------------------------------
class InterpolatedSimulationData(Cache):
    """
    A base class for computing power moments from interpolated simulation data
    """
    def __init__(self, power_lin, z, sigma8_z, f):

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
