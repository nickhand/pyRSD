import logging
import numpy

from .Pgal import *
from .constraints import *

valid_parameters = ['fcB', 'NcBs', 'NsBsB', 'fs', 'Nsat_mult', 'f1h_sBsB', 'f1h_cBs', 'fsB']

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
    if name not in valid_parameters:
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
            
