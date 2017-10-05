import numpy
import logging
import functools
import abc
from six import add_metaclass

@add_metaclass(abc.ABCMeta)
class PkmuDerivative(object):
    """
    Abstract base class for derivatives of a ``P(k,mu)`` model
    """
    @classmethod
    def registry(cls):
        """
        Return the registered subclasses
        """
        d = {}
        for subclass in cls.__subclasses__():
            d[subclass.param] = subclass

        return d

    @staticmethod
    @abc.abstractmethod
    def eval(model, pars, k, mu):
        pass

def compute(registry, name, m, pars, k, mu):
    """
    Compute the total derivative of `Pgal` with
    respect to the input parameter `name`

    Parameters
    ----------
    registry : dict
        the dictionary of available analytic derivatives
    name : str
        the parameter to compute the derivative with respect to
    m : subclass of DarkMatterSpectrum
        the RSD model instance
    pars : ParameterSet
        the theory parameters
    k : array_like
        the array of `k` values to evaluate the derivative at
    mu : array_like
        the array of `mu` values to evaluate the derivative at
    """
    if name not in set(pars.valid_model_params)|set(pars.free_names):
        logging.debug("ignoring parameter '%s'" %name)
        if numpy.isscalar(k):
            return 0.
        else:
            return numpy.zeros(len(k))

    args = (m, pars, k, mu)
    par = pars[name]

    # this is dPgal/dpar
    logging.debug("computing dPkmu/d%s" %name)
    if name not in registry:
        raise ValueError("no registered subclass for dPkmu/d%s" %name)
    dclass = registry[name]
    dPkmu_dpar = dclass.eval(*args)

    # now compute the derivatives of parameters
    # that depend on par via constraints
    for child in par.children:
        childpar = pars[child]

        # compute dPkmu/dchild
        a = compute(registry, child, m, pars, k, mu)
        if numpy.count_nonzero(a):

            # this is dchild/dpar
            b = pars.constraint_derivative(child, name)
            logging.debug("  adding dPkmu/d{child} * d{child}/d{name}".format(child=child, name=name))
            dPkmu_dpar += a*b

    return dPkmu_dpar

def _call_power_from_driver(k, mu, theta):
    """
    Update the model and call power(k,mu) from the global driver
    instance

    This is defined at the module level so we can pickle it
    """
    from pyRSD.rsdfit import GlobalFittingDriver
    driver = GlobalFittingDriver.get()
    driver.theory.set_free_parameters(theta)
    return driver.model.power(k, mu)

class PkmuGradient(object):
    """
    Class to compute the gradient of `power(k,mu)`
    """
    def __init__(self, model, registry, pars):
        """
        Parameters
        ----------
        model : subclass of DarkMatterSpectrum
            the RSD model; should have a :func:`power` function of which we
            are taking the gradient
        registry : dict
            the dictionary of available analytic derivatives
        pars : ParameterSet
            a set of parameters specifying the free parameters, constraints, etc
        """
        self.model = model
        self.pars  = pars
        self.registry = registry

        # determine which parameters require numerical derivatives
        self._find_numerical()

    def _find_numerical(self):
        """
        Internal function to determine which derivatives require a
        numerical derivative
        """
        self.numerical_names   = []
        self.numerical_indices = []
        for i, name in enumerate(self.pars.free_names):
            try:
                d = compute(self.registry, name, self.model, self.pars, 0.05, 0.5)
            except Exception as e:
                logging.info("analytic derivative for parameter '%s' not available; %s" %(name, str(e)))
                self.numerical_names.append(name)
                self.numerical_indices.append(i)

    def __call__(self, k, mu, theta, epsilon=1e-4, pool=None, numerical=False):
        """
        Evaluate the gradient of ``power(k,mu)`` with respect to
        the input parameters

        Parameters
        -----------
        k : array_like
            the wavenumbers to evaluate the gradient at
        mu : array_like
            the mu values to evaluate the gradient at
        theta : array_like
            the values of the free parameters to compute the
            gradient at
        epsilon : float or array_like, optional
            the step-size to use in the finite-difference derivative calculation;
            default is `1e-4` -- can be different for each parameter
        pool : MPIPool, optional
            a MPI Pool object to distribute the calculations of derivatives to
            multiple processes in parallel; must have a :func:`map` function
        numerical : bool, optional
            if `True`, evaluate gradients of P(k,mu) numerically using finite difference
        """
        scalar = False
        if numpy.isscalar(k):
            k = numpy.array([k])
            scalar = True
        if numpy.isscalar(mu):
            mu = numpy.array([mu])
            scalar = True

        # the result value
        toret = numpy.zeros((len(theta), len(k)))

        if pool is not None:
            self._call_power_mpi = functools.partial(_call_power_from_driver, k, mu)

        # cache results for speed
        with self.model.use_cache():

            # loop over each free parameter
            for i, name in enumerate(self.pars.free_names):
                if numerical or name in self.numerical_names:
                    continue

                # the analytic derivative
                toret[i] = compute(self.registry, name, self.model, self.pars, k, mu)[:]

        # compute numerical derivatives
        # the increments to take
        try:
            increments = numpy.identity(len(theta)) * epsilon
            if not numerical:
                ii = self.numerical_indices
            else:
                ii = Ellipsis
            tasks = numpy.concatenate([(theta+increments)[ii], (theta-increments)[ii]], axis=0)

            # how to map
            if pool is None:
                results = numpy.asarray([self._call_power(k, mu, t) for t in tasks])
            else:
                results = numpy.array(pool.map(self._call_power_mpi, tasks))
            results = results.reshape((2, -1, len(k)))

            if numpy.isscalar(epsilon):
                epsilon = numpy.ones(len(theta)) * epsilon

            # compute the central finite-difference derivative
            toret[ii] = (results[0] - results[1]) / (2.*epsilon[ii][:,None])
        except:
            raise
        finally:
            self._update(theta)

        if scalar: toret = toret[0]
        return toret

    def _update(self, theta):
        """
        Internal function to update the parameters and the model
        """
        self.pars.update_values(**dict(zip(self.pars.free_names, theta)))
        self.model.update(**self.pars.to_dict())

    def _call_power(self, k, mu, theta):
        """
        Internal function that handles updating the model and calling power(k,mu)

        This is defined at the module level so we can pickle it
        """
        # update the parameters
        self._update(theta)
        return self.model.power(k, mu).values # return numpy array, not DataArray
