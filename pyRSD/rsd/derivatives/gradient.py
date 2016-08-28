import numpy 
import logging
import functools

from pyRSD.rsdfit import GlobalFittingDriver
from . import get_Pgal_derivative, get_constraint_derivative

def compute(name, m, pars, k, mu):
    """
    Compute the total derivative of `Pgal` with 
    respect to the input parameter `name`
    
    Parameters
    ----------
    name : str
        the parameter to compute the derivative with respect to
    m : GalaxySpectrum
        the model instance
    pars : ParameterSet
        the theory parameters
    k : array_like
        the array of `k` values to evaluate the derivative at
    mu : array_like
        the array of `mu` values to evaluate the derivative at
    """
    if name not in set(pars.model_params)|set(pars.free_names):
        logging.debug("ignoring parameter '%s'" %name)
        return numpy.zeros(len(k))

    args = (m, pars, k, mu)
    par = pars[name]
            
    # this is dPgal/dpar
    logging.debug("computing dPgal/d%s" %name)
    dclass     = get_Pgal_derivative(name)
    dPgal_dpar = dclass.eval(*args) 

    # now compute the derivatives of parameters
    # that depend on par via constraints
    for child in par.children:
        childpar = pars[child]
        
        # compute dPgal/dchild
        a = compute(child, m, pars, k, mu)
        if numpy.count_nonzero(a):
            
            # this is dchild/dpar
            dconstraint = get_constraint_derivative(childpar, par)
            b = dconstraint(*args)
            logging.debug("  adding dPgal/{child} * d{child}/d{name}".format(child=child, name=name))
            dPgal_dpar += a*b
        
    return dPgal_dpar
    
def _call_Pgal_from_driver(k, mu, theta):
    """
    Update the model and call Pgal(k,mu) from the global driver
    instance
    
    This is defined at the module level so we can pickle it
    """                            
    driver = GlobalFittingDriver.get()
    driver.theory.set_free_parameters(theta)
    return driver.model.Pgal(k, mu)


class PgalGradient(object):
    """
    Class to compute the gradient of `Pgal(k,mu)`
    """
    def __init__(self, model, pars, k, mu):
        
        self.model = model
        self.pars  = pars
        self.k     = k
        self.mu    = mu
        
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
                d = compute(name, self.model, self.pars, self.k, self.mu)[:]
            except Exception as e:
                logging.info("numerical derivative for parameter '%s' not available; %s" %(name, str(e)))
                self.numerical_names.append(name)
                self.numerical_indices.append(i)
        
    def __call__(self, theta, epsilon=1e-4, pool=None):
        """
        Evaluate the gradient of `Pgal(k,mu)` with respect to
        the input parameters
        
        Parameters
        -----------
        theta : array_like
            the values of the free parameters to compute the 
            gradient at
        epsilon : float or array_like, optional
            the step-size to use in the finite-difference derivative calculation; 
            default is `1e-4` -- can be different for each parameter
        pool : MPIPool, optional
            a MPI Pool object to distribute the calculations of derivatives to 
            multiple processes in parallel
        """
        # the result value
        toret = numpy.zeros((len(theta), len(self.k)))
    
        # cache results for speed
        with self.model.use_cache():
        
            # loop over each free parameter
            for i, name in enumerate(self.pars.free_names):
                if name in self.numerical_names:
                    continue
                
                # the analytic derivative
                toret[i] = compute(name, self.model, self.pars, self.k, self.mu)[:]
                                    
        # compute numerical derivatives
        # the increments to take
        try:
            increments = numpy.identity(len(theta)) * epsilon
            ii = self.numerical_indices
            tasks = numpy.concatenate([(theta+increments)[ii], (theta-increments)[ii]], axis=0)

            # how to map
            if pool is None:
                results = numpy.array([self._call_Pgal(t) for t in tasks])
            else:
                f = functools.partial(_call_Pgal_from_driver, self.k, self.mu)
                results = numpy.array(pool.map(f, tasks))
            results = results.reshape((2, -1, len(self.k)))
    
            # compute the central finite-difference derivative
            toret[ii] = (results[0] - results[1]) / (2.*epsilon)
        except:
            raise
        finally:
            self._update(theta)
    
        return toret
        
    def _update(self, theta):
        """
        Internal function to update the parameters and the model
        """
        self.pars.update_values(**dict(zip(self.pars.free_names, theta)))
        self.model.update(**self.pars.to_dict())
        
    def _call_Pgal(self, theta):
        """
        Internal function that handles updating the model and calling Pgal(k,mu)
    
        This is defined at the module level so we can pickle it
        """                            
        # update the parameters
        self._update(theta)
        return self.model.Pgal(self.k, self.mu)
        
        
        
