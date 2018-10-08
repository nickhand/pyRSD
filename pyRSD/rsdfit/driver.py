from .. import numpy as np, os
from . import MPILoggerAdapter, logging
from . import params_filename, model_filename

from .parameters import ParameterSet, Parameter
from .theory import GalaxyPowerTheory, QuasarPowerTheory
from .data import PowerData
from .solvers import *
from .util import rsd_io
from .results import EmceeResults, LBFGSResults
from ..rsd._cache import Cache, parameter
from pyRSD.rsdfit.theory import decorators
from pyRSD import __version__
from six import string_types
import warnings

logger = MPILoggerAdapter(logging.getLogger('rsdfit.fitting_driver'))

NMU = 41

def load_results(filename):
    """
    Load a result from file
    """
    try:
        result = EmceeResults.from_npz(filename)
    except:
        result = LBFGSResults.from_npz(filename)
    return result

class FittingDriverSchema(Cache):
    """
    The schema for the :class:`FittingDriver` class, defining the allowed
    initialization parameters
    """
    @staticmethod
    def help():
        """
        Print out the help information for the necessary initialization parameters
        """
        print("Initialization Parameters for FittingDriver" + '\n' + '-'*50)
        for name in sorted(FittingDriverSchema._param_names):
            par = getattr(FittingDriverSchema, name)
            doc = name+" :\n"+par.__doc__
            if hasattr(par, '_default'):
                doc += "\n\n\tDefault: %s\n" %str(par._default)
            print(doc)

    @parameter(default='galaxy')
    def tracer_type(self, val):
        """
        The type of tracer, either 'galaxy' or 'quasar'
        """
        if val not in ['galaxy', 'quasar']:
            raise ValueError("``tracer_type`` should be 'quasar' or 'galaxy'")
        return val

    @parameter(default={})
    def theory_decorator(self, val):
        """
        A decorator to wrap the default theory output.

        Users can use this to compute linear combinations of multipoles, for
        example.
        """
        from pyRSD.rsdfit.theory.decorators import valid

        if val is not None:
            assert isinstance(val, dict), "'theory_decorator' should be a dict"
            for k in val:
                # check value is a valid decorator name
                if not isinstance(val[k], string_types) or val[k] not in valid:
                    args = (str(valid), val[k])
                    raise ValueError("theory_decorator values should be one of %s, not %s" %args)
        else:
            return {}

        return val

    @parameter(default=0)
    def burnin(self, val):
        """
        An integer specifying the number of MCMC steps to consider as part
        of the "burn-in" period when saving the results

        This doesn't affect the parameter estimation at all; it simply
        changes what best-fit parameters are printed to the screen when
        the fitting is over. All iterations are saved to file regardless of
        the value of this parameter.
        """
        return val

    @parameter(default=0.02)
    def epsilon(self, val):
        """
        The Gelman-Rubin convergence criterion
        """
        return val

    @parameter(default='fiducial')
    def init_from(self, val):
        """
        How to initialize the optimization; can be 'nlopt', 'fiducial',
        'prior', or 'result'; default is None
        """
        # check for deprecated init_values
        if val == 'max-like':
            raise ValueError("``init_from = max-like`` has been deprecated; use `nlopt` instead")
        if val == 'chain':
            raise ValueError("``init_from = chain`` has been deprecated; use `result` instead")

        valid = ['nlopt', 'fiducial', 'prior', 'result', 'previous_run']
        if val is not None and val not in valid:
            raise ValueError("valid values for 'init_from' are %s" % str(valid))

        return val

    @parameter(default=0)
    def init_scatter(self, val):
        """
        The percentage of additional scatter to add to the initial fitting
        parameters; default is 0
        """
        return val

    @parameter(default=1e-4)
    def lbfgs_epsilon(self, val):
        """
        The step-size for derivatives in LBFGS; default is 1e-4

        A dictionary can supplied where keys specify the value to use
        for specific free parameters
        """
        return val

    @parameter(default={'factr': 1e5, 'gtol': 1e-5})
    def lbfgs_options(self, val):
        """
        Configuration options to pass to the LBFGS solver
        """
        return val

    @parameter(default=True)
    def lbfgs_use_priors(self, val):
        """
        Whether to use priors when solving with the LBFGS algorithm; default
        is ``True``
        """
        return val

    @parameter(default=False)
    def lbfgs_numerical(self, val):
        """
        If `True`, evaluate gradients of P(k,mu) numerically using finite difference
        """
        return val

    @parameter(default=False)
    def lbfgs_numerical_from_lnlike(self, val):
        """
        If `True`, evaluate the gradient by taking the numerical derivative
        of :func:`minus_lnlike`
        """
        return val

    @parameter(default=None)
    def start_from(self, val):
        """
        The name of a result file to initialize the optimization from
        """
        return val

    @parameter(default=False)
    def test_convergence(self, val):
        """
        Whether to test for convergence of the parameter fits

        For MCMC fits, the Gelman-Rubin criteria is used as specified
        by the 'epsilon' parameter. For the LBFGS solver, the
        convergence criteria is set by the 'lbfgs_options' attribute.
        """
        return val

    @parameter(default={})
    def stat_specific_params(self, val):
        """
        Model parameters to be updated for each statistic; keys of this
        dict should be the names of a statistic
        """
        if val is None: return {}
        return val

class FittingDriver(FittingDriverSchema):
    """
    A driver to run the parameter fitting pipeline, merging 
    together a model, theory, and fitting algorithm.

    Parameters
    ----------
    param_file : str
        a string specifying the name of the main parameter file
    init_model : bool, optional
        if `True`, initialize the theoretical model upon initialization; default is `True`
    """
    schema = FittingDriverSchema
    required_params = set([par for par in FittingDriverSchema._param_names if not hasattr(getattr(FittingDriverSchema, par), '_default')])

    def __init__(self,
                    param_file,
                    init_model=True,
                    **kwargs):

        extra_param_file = kwargs.get('extra_param_file', None)

        # generic params
        self.params = ParameterSet.from_file(param_file, tags='driver')
        stat_specific_params = self.params.get('stat_specific_params', {})
        if stat_specific_params is None:
            self.params['stat_specific_params'].value = {}

        # set all of the valid ones
        for name in FittingDriverSchema._param_names:
            if name in self.params:
                setattr(self, name, self.params[name].value)
            try:
                has_default = getattr(self, name)
            except ValueError:
                raise ValueError("FittingDriver class is missing the '%s' initialization parameter" %name)

        # initialize the data too
        self.data = PowerData(param_file)
        self.mode = self.data.mode # mode is either pkmu/poles

        # initialize the theory
        kwargs = {}
        kwargs['extra_param_file'] = extra_param_file
        kwargs['kmin'] = self.data.global_kmin
        kwargs['kmax'] = self.data.global_kmax
        if self.tracer_type == 'galaxy':
            self.theory = GalaxyPowerTheory(param_file, **kwargs)
        else:
            self.theory = QuasarPowerTheory(param_file, **kwargs)

        # log the DOF
        args = (self.Nb, self.Np, self.dof)
        logger.info("number of degrees of freedom: %d - %d = %d" %args, on=0)

        # setup the model for data
        if init_model: self._setup_for_data()

        # results are None for now
        self.results = None

        # check theory decorator keys
        for k in self.theory_decorator:
            if k not in self.data.statistics:
                msg = "keys of 'theory_decorator' should be one of "
                msg += "'data.statistics' = %s" % str(self.data.statistics)
                raise ValueError(msg)


        # get the model callables
        self.model_callable, self.grad_model_callable = self._get_theory_callables()

    #---------------------------------------------------------------------------
    # class methods to start from directory
    #---------------------------------------------------------------------------
    @classmethod
    def from_directory(cls, dirname, results_file=None, model_file=None, 
                        init_model=True, **kwargs):
        """
        Load a :class:`FittingDriver` from a results directory

        This reads ``params.dat`` file, optionally loading a pickled model and
        a results object from file

        Parameters
        ----------
        dirname : str
            the name of the directory holding the results
        results_file : str, optional
            the name of the file holding the results. Default is ``None``
        model_file : str, optional
            the name of the file holding the model to load. Default is ``None``
        init_model : bool, optional
            whether to initialize the RSD model upon loading. If a model
            file exists in the specified directory, the model is loaded and
            no new model is initialized
        """
        from pyRSD.rsd import DarkMatterSpectrum # import the base class to test for

        if not os.path.isdir(dirname):
            raise rsd_io.ConfigurationError('`%s` is not a valid directory' %dirname)
        params_path = os.path.join(dirname, params_filename)

        model_path = model_file
        existing_model = isinstance(model_path, string_types) and os.path.exists(model_path)
        existing_model = existing_model or isinstance(model_path, DarkMatterSpectrum)
        if model_path is not None:
            if not existing_model:
                raise rsd_io.ConfigurationError('provided model file `%s` does not exist' %model_path)
        else:
            model_path = os.path.join(dirname, model_filename)
        if not os.path.exists(params_path):
            raise rsd_io.ConfigurationError('parameter file `%s` must exist to load driver' %params_path)

        init_model = (not existing_model and init_model)
        driver = cls(params_path, init_model=init_model, **kwargs)

        # set the model and results
        if existing_model:
            driver.model = model_path
        if results_file is not None:
            if isinstance(results_file, string_types) and not os.path.exists(results_file):
                if not os.path.join(dirname, results_file):
                    raise rsd_io.ConfigurationError('specified results file `%s` does not exist' %results_file)
                else:
                    results_file = os.path.join(dirname, results_file)
            driver.results = results_file


        return driver

    @classmethod
    def default_params(cls, **params):
        """
        Return a :class:`ParameterSet` object using a mix of the default
        parameters and the parameters specified as keywords

        .. note::
            All required parameters (see :attr:`required_params`) must be
            specified as keywords or an exception will be raised

        Parameters
        ----------
        **params :
            parameter values specified as key/value pairs
        """
        for name in cls.schema._param_names:
            par = getattr(cls.schema, name)
            if hasattr(par, '_default'):
                params.setdefault(name, par._default)

        # check for missing parameters
        missing = cls.required_params - set(params)
        if len(missing):
            raise ValueError("missing the following required parameters: %s" %str(missing))

        # make the set of parameters
        toret = ParameterSet()
        for name in params:
            if name not in cls.schema._param_names:
                warnings.warn("'%s' is not a valid parameter name for FittingDriver" %name)
            toret[name] = Parameter(name=name, value=params[name])
        toret.tag = 'driver'

        return toret

    def _get_theory_callables(self):
        """
        Given the data, theory, and input parameters generate the 
        callable function that returns the theory measurements. 

        The hard part here is determining how many P(k,mu) calls 
        we need to make.
        """
        stat_specific_params = self.params.get('stat_specific_params', {})

        # determine the number of unique theory P(k,mu) calls
        stat_grps = []
        N1 = len(stat_specific_params)
        if N1 > 0:
            stat_grps += [[stat] for stat in stat_specific_params]
        missing = set(self.data.statistics) - set(stat_specific_params)
        if len(missing):
            stat_grps.append(list(missing))

        multiple_grps = len(stat_grps) > 1
        changed_params = []
        for k in stat_specific_params:
            changed_params += list(stat_specific_params[k])

        default_params = {}
        for par in set(changed_params):
            default_params[par] = self.theory.model_params[par]()
        
        # determine the transfers
        callables = []
        grad_callables = []
        for stat_grp in stat_grps:
            
            # get the transfers
            transfers, ids = self.data.calculate_transfer(stat_grp)
            
            # get the model parameters
            if len(default_params):
                model_params = default_params.copy()
            else:
                model_params = {}
            if len(ids) == 1:
                key = list(ids.keys())[0]
                if key in stat_specific_params:
                    model_params.update(stat_specific_params[key])
            
            # get the theory callable
            if not len(model_params):
                model_params = None
            c = self.theory.get_model_callable(self.data, transfers, ids, 
                                                model_params=model_params,
                                                theory_decorator=self.theory_decorator)
            callables.append(c)

            # get the grad theory callable
            c = self.theory.get_grad_model_callable(self.data, transfers, ids, 
                                                     model_params=model_params,
                                                     theory_decorator=self.theory_decorator)
            grad_callables.append(c)

        def final_model_callable():
            return np.concatenate([c() for c in callables], axis=0)
            
        def final_grad_callable(**kwargs):
            return np.concatenate([c(**kwargs) for c in grad_callables], axis=-1)

        return final_model_callable, final_grad_callable


    def apply(self, func, pattern):
        """
        Apply a function for several results files

        Parameters
        ----------
        func : callable
            the function
        pattern : str
            the pattern to match result files on

        Returns
        -------
        k, result : array_like
            the stacked results file
        """
        from glob import glob
        results = glob(pattern)

        toret = []
        for r in results:
            self.results = r
            self.set_fit_results()
            toret.append(func(self))

        return np.asarray(toret)

    def set_restart(self, restart_file, iterations):
        """
        Initialize the `restart` mode by loading the results from a previous run

        Parameters
        ----------
        restart_file : str
            the name of the file holding the results to restart from
        iterations : int
            the  number of additional iterations to run

        Returns
        -------
        solver_type : {'mcmc', 'nlopt'}
            the solver type
        """
        # tell driver we are starting from previous run
        self.init_from = 'previous_run'
        self.params['init_from'].value = self.init_from

        # set the results
        self.results = restart_file
        total_iterations = iterations + self.results.iterations

        if isinstance(self.results, LBFGSResults):
            solver_type = 'nlopt'
        else:
            solver_type = 'mcmc'

        # set the number of iterations to the total sum we want to do
        if solver_type == 'mcmc':
            self.params.add('iterations', value=total_iterations)
            self.params.add('walkers', value=self.results.walkers)

        elif solver_type == 'nlopt':
            options = self.lbfgs_options
            options['max_iter'] = total_iterations
            self.params.add('lbfgs_options', value=options)

        return solver_type

    def to_file(self, filename, mode='w'):
        """
        Write the parameters of this driver to a file

        Parameters
        ----------
        filename : str
            the name of the file to write out
        mode : {'a', 'w'}
            the mode to use when writing to file
        """
        kwargs = {'mode':mode, 'header_name':'driver_params', 'footer':True, 'as_dict':False}
        self.params.to_file(filename, **kwargs)
        self.data.to_file(filename, mode='a')
        self.theory.to_file(filename, mode='a')

    def run(self, solver_type, pool=None, chains_comm=None):
        """
        Run the whole fitting analysis, from start to finish

        Parameters
        ----------
        solver_type : {'mcmc', 'nlopt'}
            either run a MCMC fit with emcee or a nonlinear optimization using LBFGS
        pool : MPIPool, optional
            a MPI pool object to distribute tasks too
        chains_comm : MPI communicator, optional
            a communicator for communicating between multiple MCMC chains
        """
        if solver_type not in ['mcmc', 'nlopt']:
            raise ValueError("'solver_type' parameter must be 'mcmc' or 'nlopt', not '%s'" %solver_type)
        init_values = None

        # init from maximum probability solution
        if self.init_from == 'nlopt':
            init_values = self.find_peak_probability(pool=pool)

        # init from fiducial values
        elif self.init_from == 'fiducial':
            init_values = self.theory.free_fiducial

        # init from previous result
        elif self.init_from == 'result':

            # the result to start from
            if self.start_from is None:
                raise ValueError("if ``init_from = 'result'``, a `start_from` file path must be provided")

            # load the result and get the best-fit values
            try:
                result = EmceeResults.from_npz(self.start_from)
                best_values = result.max_lnprob_values()
            except:
                result = LBFGSResults.from_npz(self.start_from)
                best_values = result.min_chi2_values
            best_values = dict(zip(result.free_names, best_values))

            init_values = np.empty(self.Np)
            for i, key in enumerate(self.theory.free_names):
                if key in best_values:
                    init_values[i] = best_values[key]
                else:
                    if self.theory.fit_params[key].has_fiducial is not None:
                        init_values[i] = self.theory.fit_params[key].fiducial
                    else:
                        raise ValueError("cannot initiate from previous result -- `%s` parameter missing" %key)

            # log the file name
            logger.info("initializing run from previous result: '%s'" %self.start_from)

        elif self.init_from == 'prior' and solver_type == 'nlopt':
            raise ValueError("initializing from prior is only valid for MCMC solver type")

        # restart from previous result
        elif self.init_from  == 'previous_run':
            init_values = self.results.copy()

        kwargs = {'pool':pool}
        if init_values is not None:
            kwargs['init_values'] = init_values

        # emcee
        if solver_type == 'mcmc':

            # do not use any analytic approximations for bounds and priors
            # since MCMC can handle discontinuity in log-prob
            for name in self.theory.free_names:
                self.theory.fit_params[name].analytic = False

            solver = emcee_solver.run
            kwargs['chains_comm'] = chains_comm

        # lbfgs
        elif solver_type == 'nlopt':
            solver = lbfgs_solver.run

            # use analytic approximation for bounds and priors
            # to enforce continuity
            for name in self.theory.free_names:
                self.theory.fit_params[name].analytic = True

        # run the solver and store the results
        if self.init_from == 'previous_run':
            logger.info("Restarting '{}' solver from a previous result".format(solver_type))
        else:
            logger.info("Calling the '{}' solve function".format(solver_type))
        self.results, exception = solver(self.params, self.theory.fit_params, **kwargs)

        # store the model version in the results
        if self.results is not None:
            self.results.model_version = self.theory.model.__version__
            self.results.pyrsd_version = __version__

        logger.info("...fitting complete")
        return exception

    def finalize_fit(self, exception, results_file):
        """
        Finalize the fit, saving the results file
        """
        # save the results as a pickle
        logger.info('Saving the results to `%s`' %results_file)
        self.results.to_npz(results_file)

        # summarize if no exception
        if exception is None or isinstance(exception, KeyboardInterrupt):
            self.results.summarize_fit()

    def find_peak_probability(self, pool=None):
        """
        Find the peak of the probability distribution

        This uses either the maximum likelihood (ML) (excluding log priors) or
        maximum a posteriori probability (MAP) (including log priors) estimation
        """
        # use analytic approximation for bounds and priors
        # to enforce continuity
        for name in self.theory.free_names:
            self.theory.fit_params[name].analytic = True

        # get the solver and objective
        solver = lbfgs_solver.run

        # init values from fiducial
        init_values = self.theory.free_fiducial

        logger.info("using L-BFGS soler to find the maximum probability values to use as initialization")
        results, exception = solver(self.params, self.theory.fit_params, 
                                    pool=pool, init_values=init_values)
        logger.info("...done compute maximum probability")

        values = results.min_chi2_values
        del results
        return values

    #---------------------------------------------------------------------------
    # setup functions
    #---------------------------------------------------------------------------
    def _setup_for_data(self):
        """
        Setup the model callables for this set of data
        """
        # initialize the model
        logger.info("Initializing theoretical model", on=0)

        # update the model parameters
        pars = self.theory.fit_params
        self.theory.model.update(**pars.to_dict())

        # initialize
        self.theory.model.initialize()
        logger.info("...theoretical model initialized", on=0)

    @property
    def results(self):
        """
        The `results` object storing the fitting results
        """
        try:
            return self._results
        except AttributeError:
            return None

    @results.setter
    def results(self, val):
        """
        Set the results, checking to make sure we re-order the fitted params
        into the right order
        """
        if isinstance(val, string_types):
            val = load_results(val)

        # possibly reorder the results
        if hasattr(val, 'verify_param_ordering'):
            free_params = self.theory.free_names
            constrained_params = self.theory.constrained_names
            val.verify_param_ordering(free_params, constrained_params)
        self._results = val

    @property
    def model(self):
        """
        The `model` object which returns the P(k,mu) or multipoles theory
        """
        return self.theory.model

    @model.setter
    def model(self, val):
        """
        Set the theoretical model
        """
        # set it
        if isinstance(val, string_types):
            logger.info("setting the theoretical model from file `%s`" %val, on=0)
        else:
            logger.info("setting the theoretical model from existing instance", on=0)
        self.theory.model = val

        # print out the model parameters
        params = self.theory.model.config
        msg = "running with model parameters:\n\n"
        msg += "\n".join(["%-25s: %s" %(k, str(v)) for k,v in sorted(params.items())])
        logger.info(msg, on=0)
        logger.info("...theoretical model successfully read", on=0)

    @property
    def combined_model(self):
        """
        The model values for each measurement, flattened column-wise
        into a single array.

        Optionally, ``theory_decorator`` is applied.

        Notes
        -----
        The model callable should already returned the flattened `combined` values
        """
        return self.model_callable()

    @property
    def null_lnlike(self):
        """
        Return the log-likelihood value for a null model, to be used
        when the model parameters are in an invalid state
        """
        try:
            return self._null_lnlike
        except:
            d = self.data.combined_power
            self._null_lnlike = -0.5 * np.dot(d, np.dot(self.data.covariance_matrix.inverse, d))
            return self._null_lnlike

    @property
    def dof(self):
        """
        The number of degrees of freedom

        This is equal to the number of data points minus the number of free parameters
        """
        return self.Nb - self.Np

    @property
    def Nb(self):
        """
        The number of data points
        """
        return self.data.ndim

    @property
    def Np(self):
        """
        The number of free parameters
        """
        return self.theory.ndim

    #---------------------------------------------------------------------------
    # probability functions
    #---------------------------------------------------------------------------
    def lnprior(self):
        """
        Return the log of the prior, based on the current values of the free parameters
        """
        return self.theory.lnprior

    def chi2(self, theta=None):
        r"""
        The chi-squared for the specified model function

        This returns

        .. math::

            \chi^2 = (\mathcal{M} - \mathcal{D})^T C^{-1} (\mathcal{M} - \mathcal{D})

        Parameters
        ----------
        theta : array_like, optional
            an array of the free parameters to evaluate the statistic at; if ``None``,
            the current values of the free parameters in :attr:`theory.fit_params`
            is used
        """
        # set the free parameters
        if theta is not None:
            self.theory.set_free_parameters(theta)

        diff = self.combined_model - self.data.combined_power
        return np.dot(diff, np.dot(self.data.covariance_matrix.inverse, diff))

    def reduced_chi2(self):
        """
        The reduced chi squared value, using the current values of the free parameters
        """
        return self.chi2() / self.dof

    def lnlike(self, theta=None):
        """
        The log of the likelihood, equal to -0.5 * :func:`chi2`

        Parameters
        ----------
        theta : array_like, optional
            an array of the free parameters to evaluate the statistic at; if ``None``,
            the current values of the free parameters in :attr:`theory.fit_params`
            is used
        """
        return -0.5*self.chi2(theta=theta)

    def lnprob(self, theta=None):
        """
        Set the theory free parameters, update the model, and return the log of the posterior probability function

        This returns -0.5 :func:`chi2` + :func:`lnprior`

        Parameters
        ----------
        theta : array_like, optional
            an array of the free parameters to evaluate the statistic at; if ``None``,
            the current values of the free parameters in :attr:`theory.fit_params`
            is used
        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)
        else:
            in_bounds = all(p.within_bounds() for p in self.theory.free)

        # return -np.inf if any parameters are out of bounds
        if not in_bounds:
            return -np.inf

        # check the prior
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        # only compute lnlike if we have finite prior
        else:
            try:
                lnlike = self.lnlike()
                if np.isnan(lnlike):
                    raise ValueError("log-likelihood calculation resulted in NaN")
            except:
                import traceback
                msg = "exception while computing log-likelihood:\n"
                msg += "   current parameters:\n%s\n" %str(self.theory.fit_params)
                msg += "   traceback:\n%s" %(traceback.format_exc())
                raise RuntimeError(msg)

            return lp + lnlike

    def minus_lnlike(self, theta=None, use_priors=False):
        """
        Return the negative log-likelihood, optionally including priors

        Parameters
        ----------
        theta : array_like, optional
            an array of the free parameters to evaluate the statistic at; if ``None``,
            the current values of the free parameters in :attr:`theory.fit_params`
            is used
        use_priors : bool, optional
            whether to include the log priors in the objective function when
            minimizing the negative log probability
        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)

            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:

                with self.theory.preserve(theta):
                    return -self.null_lnlike - self.lnprior()

        # value at theta
        prob = -self.lnlike()
        if use_priors:
            prob += -self.lnprior()

        return prob

    #---------------------------------------------------------------------------
    # gradient calculations
    #---------------------------------------------------------------------------
    def grad_minus_lnlike(self, theta=None, epsilon=1e-4, pool=None, use_priors=True,
                            numerical=False, numerical_from_lnlike=False):
        """
        Return the vector of the gradient of the negative log likelihood, with respect to the free parameters

        This uses a central-difference finite-difference approximation to
        compute the numerical derivatives

        Parameters
        ----------
        theta : array_like, optional
            if provided, the values of the free parameters to compute the
            gradient at; if not provided, the current values of free parameters
            from :attr:`theory.fit_params` will be used
        epsilon : float or array_like, optional
            the step-size to use in the finite-difference derivative calculation;
            default is `1e-4` -- can be different for each parameter
        pool : MPIPool, optional
            a MPI Pool object to distribute the calculations of derivatives to
            multiple processes in parallel
        use_priors : bool, optional
            whether to include the log priors in the objective function when
            minimizing the negative log probability
        numerical : bool, optional
            if `True`, evaluate gradients of P(k,mu) numerically using finite difference
        numerical_from_lnlike : bool, optional
            if `True`, evaluate the gradient by taking the numerical derivative
            of :func:`minus_lnlike`

        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)

            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:

                # set parameters to get derivatives at invalid state
                with self.theory.preserve(theta):

                    # derivatives at invalid state
                    dlnprior = self.theory.dlnprior

                return -1.0 * dlnprior
        else:
            theta = self.theory.free_values

        # numerical gradient of minus_lnlike
        if numerical_from_lnlike:

            increments = np.identity(len(theta)) * epsilon
            tasks = np.concatenate([theta+increments, theta-increments], axis=0)
            results = np.array([self.minus_lnlike(t) for t in tasks])
            results = results.reshape((2, -1))

            grad_minus_lnlike = (results[0] - results[1]) / (2.*epsilon)

        # use as much analytic as possible
        else:

            # evaluate the gradient of lnlike
            kws = {}
            kws['pool'] = pool
            kws['epsilon'] = epsilon
            kws['numerical'] = numerical
            kws['theta'] = theta
            grad_lnlike = self.grad_model_callable(**kws)

            # transform from model gradient to log likelihood gradient
            diff = self.data.combined_power - self.combined_model
            grad_lnlike = np.dot(np.dot(self.data.covariance_matrix.inverse, diff), grad_lnlike.T)
            grad_minus_lnlike = -1 * grad_lnlike

        # test for inf
        has_inf = np.isinf(grad_minus_lnlike)
        if has_inf.any():
            bad = np.array(self.theory.free_names)[has_inf].tolist()
            raise ValueError("inf values detected in the gradient for: %s" %str(bad))

        # test for NaN
        has_nan = np.isnan(grad_minus_lnlike)
        if has_nan.any():
            bad = np.array(self.theory.free_names)[has_nan].tolist()
            raise ValueError("NaN values detected in the gradient for: %s" %str(bad))

        # add in log prior prob and derivatives of log priors
        if use_priors:
            grad_minus_lnlike += -1 * self.theory.dlnprior

        return grad_minus_lnlike

    def fisher(self, theta=None, epsilon=1e-4, pool=None, use_priors=True,
                numerical=False, numerical_from_lnlike=False):
        r"""
        Return the Fisher information matrix

        This is defined as the negative Hessian of the log likelihood with
        respect to the parameter vector

        .. math::

            F_{ij} = - \frac{\partial^2 \log \mathcal{L}}{\partial \theta_i \partial \theta_j}

        This uses a central-difference finite-difference approximation to
        compute the numerical derivatives

        Parameters
        ----------
        theta : array_like, optional
            if provided, the values of the free parameters to compute the
            gradient at; if not provided, the current values of free parameters
            from `theory.fit_params` will be used
        epsilon : float or array_like, optional
            the step-size to use in the finite-difference derivative calculation;
            default is `1e-4` -- can be different for each parameter
        pool : MPIPool, optional
            a MPI Pool object to distribute the calculations of derivatives to
            multiple processes in parallel
        use_priors : bool, optional
            whether to include the log priors in the objective function when
            minimizing the negative log probability
        numerical : bool, optional
            if `True`, evaluate gradients of P(k,mu) numerically using finite difference
        numerical_from_lnlike : bool, optional
            if `True`, evaluate the gradient by taking the numerical derivative
            of :func:`minus_lnlike`
        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)

            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:
                raise ValueError("the parameter vector is not valid (out of bounds); bad!")
        else:
            theta = self.theory.free_values

        # numerical gradient of minus_lnlike
        if numerical_from_lnlike:
            try:
                import numdifftools
            except ImportError:
                raise ImportError("``numdifftools`` is required to compute the Fisher Hessian numerically")

            # return the numerical Hessian of either the log prob (with priors) or log likelihood
            if use_priors:
                f = self.lnprob
            else:
                f = self.lnlike
            H = numdifftools.Hessian(f, step=epsilon)
            return -1.0 * H(theta)

        else:

            # get the gradient of Pgal wrt the parameters
            gradient = self.pkmu_gradient(theta, pool=pool, epsilon=epsilon, numerical=numerical)

            # evaluate gradient via the transfer function
            if self.data.transfer is not None:

                grad_lnlike = []
                for i in range(self.Np):
                    self.data.transfer.power = gradient[i]
                    grad_lnlike.append(self.data.transfer(flatten=True))
                grad_lnlike = np.asarray(grad_lnlike)

            # do pole calculation yourself
            elif self.mode == 'poles':
                from scipy.special import legendre
                from scipy.integrate import simps

                # reshape the arrays
                k        = self.pkmu_gradient.k.reshape(-1, NMU, order='F')
                mu       = self.pkmu_gradient.mu.reshape(-1, NMU, order='F')
                gradient = gradient.reshape(self.Np, -1, NMU, order='F')

                # do the integration over mu
                kern     = np.asarray([(2*ell+1.)*legendre(ell)(mu[i]) for i, ell in enumerate(self.data.combined_ell)])
                grad_lnlike = np.array([simps(t, x=mu[i]) for i,t in enumerate(kern*gradient)])

            else:
                grad_lnlike = gradient

            # compute Fisher from gradient
            F = np.zeros((self.Np, self.Np))
            for i in range(self.Np):
                for j in range(i, self.Np):
                    F[i,j] =  np.dot(grad_lnlike[i], np.dot(self.data.covariance_matrix.inverse, grad_lnlike[j]))
                    F[j,i] = F[i,j]

            # add priors??
            priors = np.zeros(self.Np)
            if use_priors:
                for ipar, par in enumerate(self.theory.free):
                    if par.prior.name == 'normal':
                        prior = par.prior.scale**(-2)
                    elif par.prior.name == 'uniform':
                        prior = 0.
                    else:
                        raise NotImplemented("adding prior of type '%s' to Fisher matrix not implemented" %par.prior.name)
                    priors[ipar] = prior
            priors = np.diag(priors)

            # add the (possibly zero) info from priors
            F += priors

            return F

    def jacobian(self, params, theta=None):
        """
        Return the Jacobian between the input (constrained) parameters and the
        free parameters, holding dp' / dp where p' is the set of constrained
        ``params`` and p is the set of free parameters

        Parameters
        ----------
        params : list
            list of the constrained parameters
        theta : array_like, optional
            free parameter values to evaluate Jacobian of transformation at

        Returns
        -------
        J : array_like (self.Np, len(params))
            the Jacobian
        """
        # set the free parameters
        if theta is not None:
            in_bounds = self.theory.set_free_parameters(theta)

            # if parameters are out of bounds, return the log-likelihood
            # for a null model + the prior results (which hopefully are restrictive)
            if not in_bounds:
                raise ValueError("the parameter vector is not valid (out of bounds); bad!")
        else:
            theta = self.theory.free_values

        J = np.zeros((self.Np, len(params)))
        for irow, freepar in enumerate(self.theory.free_names):
            for icol, newpar in enumerate(params):

                # transformation between two free parameters
                if freepar in self.theory.free_names and newpar in self.theory.free_names:
                    if freepar == newpar:
                        J[irow, icol] = 1.0
                    else:
                        J[irow, icol] = 0.
                else:
                    # this is dnewpar / dfreepar
                    deriv = self.theory.fit_params.constraint_derivative(newpar, freepar, theta=theta)
                    J[irow, icol] = deriv

        return J

    def marginalized_errors(self, params=None, theta=None, fixed=[], **kws):
        """
        Return the marginalized errors on the specified parameters, as computed from the Fisher matrix

        This is given by: :math:`(\mathcal{F}^{-1}))^(1/2)`

        Optionally, we can fix certain parameters, specified in ``fixed``

        Parameters
        ----------
        params : list, optional
            return errors for this parameters; default is all free parameters
        theta : array_like, optional
            the array of free parameters to evaluate the best-fit model at
        fixed : list, optional
            list of free parameters to fix when evaluating marginalized errors
        **kws :
            additional keywords passed to :func:`fisher`

        Returns
        -------
        errors : array_like
            the marginalized errors as computed from the inverse of the
            Fisher matrix
        """
        if isinstance(params, string_types):
            params = [params]
        if isinstance(fixed, string_types):
            fixed = [fixed]

        # return for all free parameters by default
        if params is None:
            params = self.theory.free_names

        # compute Fisher
        F = self.fisher(theta=theta, **kws)

        # Jacobian
        J = self.jacobian(params, theta=theta)

        # no fixing, just do inverse Fisher
        if not len(fixed):
            C = np.linalg.inv(F)

        # fixed = remove some row/columns from Fisher before inverting
        else:

            # fixed parameters must be free initially
            if not all(name in self.theory.free_names for name in fixed):
                raise ValueError("we can only fix free parameters")

            # remove fixed params from F
            indices = [self.theory.free_names.index(name) for name in fixed]
            F = np.delete(np.delete(F, indices, axis=0), indices, axis=1)
            C = np.linalg.inv(F)
            J = np.delete(J, indices, axis=0)

        # return the properly transformed errors
        return np.dot(J.T, np.dot(C, J))**0.5

    #---------------------------------------------------------------------------
    # setting results
    #---------------------------------------------------------------------------
    def set_fiducial(self):
        """
        Set the fiducial values as the current values of the free parameters
        """
        free = self.theory.free_names
        params = self.theory.fit_params
        theta = np.array([params[key].fiducial for key in free])

        if len(theta) != self.theory.ndim:
            logger.error("problem setting fiducial values; not correct number")
            raise ValueError("Number of fiducial values not equal to number of free params")
        self.theory.set_free_parameters(theta)

    def set_fit_results(self, method='median'):
        """
        Set the free parameters from the results objects and update the model
        """
        if self.results is not None:
            if isinstance(self.results, LBFGSResults):
                theta = self.results.min_chi2_values
            else:
                if method == 'median':
                    theta = self.results.values()
                elif method == 'peak':
                    theta = self.results.peak_values()
                elif method == 'max_lnprob':
                    theta = self.results.max_lnprob_values()
                else:
                    raise ValueError("`method` keyword must be one of ['median', 'peak', 'max_lnprob']")
            self.theory.set_free_parameters(theta)

    def preserve(self):
        """
        Context manager that preserves the state of the model
        upon exiting the context by first saving and then restoring it
        """
        return self.theory.model.preserve()

    def data_model_pairs(self):
        """
        Return the data - model pairs for each measurement

        Returns
        -------
        toret : list
            A list of (k_data, data, err, model) for each measurement
            in ``data.measurements``
        """
        # this tells you how to slice the flattened results
        flat_slices = self.data.flat_slices

        # get the model and data measurements
        toret = []
        for i, m in enumerate(self.data):
            model = self.combined_model[flat_slices[i]]
            toret.append((m.k, m.power, m.error, model))

        return toret

    #---------------------------------------------------------------------------
    # plotting functions
    #---------------------------------------------------------------------------
    def plot_residuals(self):
        """
        Plot the residuals of the best-fit theory and data
        """
        from matplotlib import pyplot as plt

        # get the data - model pairs (k, model, data, err)
        results = self.data_model_pairs()

        # loop over each measurement
        for i, result in enumerate(results):

            # make the residual
            k, data, errs, model = result
            residual = (data - model) / errs

            # make the plot
            if self.mode == 'pkmu':
                lab = r"$\mu = %.2f$" %(self.data[i].identifier)
            else:
                lab = r"$\ell = %d$" %(self.data[i].identifier)
            plt.plot(k, residual, "o", label=lab)

        # make it look nice
        ax = plt.gca()
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel("wavenumber k", fontsize=16)
        ax.set_ylabel("residuals (data - model)/error", fontsize=16)
        ax.legend(loc=0, ncol=2)

    def iplot(self, colors=None):
        """
        Make an interactive plot of the best-fit theory and data points using
        :mod:`plotly`.
        """
        from .plotly.fit import plot_fit_comparison
        import plotly.offline as py
        return py.iplot(plot_fit_comparison(self, colors=colors))

    def plot(self, usetex=False, ax=None, colors=None, use_labels=True, **kws):
        """
        Plot the best-fit theory and data points
        """
        from .util import plot

        # plot the fit comparison
        ax = plot.plot_fit_comparison(self, ax=ax, colors=colors, use_labels=use_labels, **kws)

        # set usetex
        if usetex:
            ax.figure.usetex = True
