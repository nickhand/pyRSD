import numpy as np
import copy
import logging

logger = logging.getLogger('pyRSD.analysis.lmfit_results')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
class LmfitResults(object):
    """
    Class to hold the fitting results from an `lmfit` nonlinear optimization run.
    
    Notes
    -----
    Individual parameters can be accessed in a dict-like fashion, using
    the parameter name as the key passed to the `LmfitResults` object. 
    
    The relevant attributes for each parameter are:
        
            correl : dict
                A dict where the keys are the other free parameters and values
                are the correlations b/w the two variables
            max : float
                The maximum allowed value
            min : float
                The minimu allowed value
            name : str
                The name of the parameter
            value : float
                The best-fit value of the parameter
            stderr : float
                The estimated standard error on the parameter
    
    """
    def __init__(self, minimizer):
        """
        Initialize with the `lmfit.Minimizer` object and the fitting parameters
        """              
        # store the params
        self.params = minimizer.params
        
        # store the parameter names
        self.free_parameter_names = self.params.keys()
        
        # chi-squared
        self.reduced_chi2 = minimizer.redchi
        self.chi2 = minimizer.chisqr
        
        # some other attributes
        self.has_errorbars = minimizer.errorbars
        self.residual = minimizer.residual
        
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string method
        """
        free_params = [self[name] for name in self.free_parameter_names]
        
        # first get the parameters
        toret = "Free parameters\n" + "_"*15 + "\n"
        toret += "\n".join(map(str, free_params))
        return toret
            
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation
        """
        return "<LmfitResults: {} free paremeters>".format(self.ndim)
        
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        
        # check if key is the name of a free or constrained param
        if key in (self.free_parameter_names):
            return self.params[key]
        else:
            return getattr(self, key)
                    
    #---------------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy of the `EmceeResults` object
        """
        return copy.deepcopy(self)
            
    #---------------------------------------------------------------------------        
    @property
    def ndim(self):
        """
        The number of free parameters
        """
        return len(self.free_parameter_names)
        
    #---------------------------------------------------------------------------
    def correl(self, param1, param2):
        """
        Return the correlation between the two parameters
        """
        if not self.has_errorbars:
          return np.nan  
        return self[param1].correl[param2]
        
    #---------------------------------------------------------------------------
    def summarize_fit(self, *args, **kwargs):
        """
        Summarize the fit by calling `lmfit.fit_report`
        """
        logger.info("\n" + lmfit.fit_report(self.params))
        
    #---------------------------------------------------------------------------
    def values(self):
        """
        Convenience function to return the values for the free parameters
        as an array
        """
        return np.array([self.params[name].value for name in self.params])
        
    #---------------------------------------------------------------------------
#endclass LmfitResults

#-------------------------------------------------------------------------------
    
