import numpy as np
import copy
import logging

logger = logging.getLogger('pyRSD.analysis.lbfgs_results')
logger.addHandler(logging.NullHandler())

class LBFGSResults(object):
    """
    Class to hold the fitting results from an `scipy.optimize` L-BFGS-B 
    nonlinear optimization run.    
    """
    def __init__(self, data, fit_params):
        """
        Initialize
        """                  
        # store the parameter names
        self.free_names        = fit_params.free_names
        self.constrained_names = fit_params.constrained_names
        
        # store the results
        self.data = data
        self.min_chi2 = data['curr_state'].F
        self.min_chi2_values = data['curr_state'].X
        
        # constrained
        self.min_chi2_constrained_values = self._get_constrained_values(fit_params)
        
    @property
    def iterations(self):
        """
        The total number of iterations ran
        """
        return self.data['iteration']
        
    def verify_param_ordering(self, free_params, constrained_params):
        """
        Verify the ordering of of parameters, making sure that the
        ordering specified by `free_params` and  `constrained_params` is 
        respected in the results
        """
        if sorted(self.free_names) != sorted(free_params):
            raise ValueError("mismatch in `LBFGSResults` free parameters")
        if sorted(self.constrained_names) != sorted(constrained_params):
            raise ValueError("mismatch in `LBFGSResults` constrained parameters")
        
        reordered = False
        # reorder ``min_chi2_values``
        if self.free_names != free_params:
            inds = [self.free_names.index(k) for k in free_params]
            self.min_chi2_values = self.min_chi2_values[inds]
            reordered = True
        
        # reorder ``min_chi2_constrained_values``
        if self.constrained_names != constrained_params:
            reordered = True
        
        if reordered:
            self.free_names = free_params
            self.constrained_names = constrained_params
            
    def to_npz(self, filename):
        """
        Save the relevant information of the class to a numpy ``npz`` file
        """
        atts = ['free_names', 'constrained_names', 'min_chi2', 'data',
                'min_chi2_values', 'min_chi2_constrained_values']
        d = {k:getattr(self, k) for k in atts}
        d['model_version'] = getattr(self, 'model_version', None)
        np.savez(filename, **d)
        
    @classmethod
    def from_npz(cls, filename):
        """
        Load a numpy ``npz`` file and return the corresponding ``EmceeResults`` object
        """
        toret = cls.__new__(cls)
        with np.load(filename, encoding='latin1') as ff:
            for k, v in ff.items():
                setattr(toret, k, v)
        
        for a in ['min_chi2', 'free_names', 'constrained_names', 'data']:
            v = getattr(toret, a)
            setattr(toret, a, v.tolist())
            
        # remove old structured arrays from constrained values
        x = toret.min_chi2_constrained_values
        if x.dtype.char == "V":
            toret.min_chi2_constrained_values = np.array([x[name] for name in toret.constrained_names])
            
        return toret
        
    def __iter__(self):
        return iter(self.free_names + self.constrained_names)
        
    def _get_constrained_values(self, fit_params):
        """
        Get the values for the constrained parameters at min chi2
        """
        if len(self.constrained_names) == 0:
            return None
        
        # set the free parameters
        for val, name in zip(self.min_chi2_values, self.free_names):
            fit_params[name].value = val

        # update constraints
        fit_params.update_values()
                        
        # return the constrained vals
        return fit_params.constrained_values

    def __str__(self):
        
        # first get the parameters
        toret = "Free parameters [ mean ]\n" + "_"*15 + "\n"
        toret += "\n".join(["%-15s: %s" %(p, str(self[p])) for p in self.free_names])
        
        toret += "\n\nConstrained parameters [ mean ]\n" + "_"*25 + "\n"
        toret += "\n".join(["%-15s: %s" %(p, str(self[p])) for p in self.constrained_names])
        
        s = "minimum chi2 = %s\n\n" %str(self.min_chi2)
        return s + toret
    
    def __repr__(self):
        N = len(self.constrained_names)
        name = self.__class__.__name__
        return "<{}: {} free parameters, {} constrained parameters>".format(name, self.ndim, N)
        
    def __getitem__(self, key):
        
        # check if key is the name of a free or constrained param
        if key in self.free_names:
            i = self.free_names.index(key)
            return self.min_chi2_values[i]
        elif key in self.constrained_names:
            i = self.constrained_names.index(key)
            return self.min_chi2_constrained_values[i]
        else:
            return getattr(self, key)
                    
    def copy(self):
        """
        Return a deep copy of the `EmceeResults` object
        """
        return copy.deepcopy(self)
                   
    @property
    def ndim(self):
        """
        The number of free parameters
        """
        return len(self.free_names)
                
    def summarize_fit(self, *args, **kwargs):
        """
        Summarize the fit by printing self
        """
        logger.info("\n" + self.__str__())
