from ._cache import parameter
from scipy.interpolate import InterpolatedUnivariateSpline as spline
    
#-------------------------------------------------------------------------------
class SimLoaderMixin(object):
    """
    A mixin class to help deal with loading of power terms from simulations
    """
    _loadable_power_terms = ["Pdv", "P00_mu0", "P01_mu2", "P11_mu2", "P11_mu4"]
    
    def __init__(self):
        self._loaded_data   = {}
        self.Pdv_loaded     = False
        self.P00_mu0_loaded = False
        self.P01_mu2_loaded = False
        self.P11_mu2_loaded = False
        self.P11_mu4_loaded = False
        
    #---------------------------------------------------------------------------
    # THE INTERPOLATION FUNCTIONS
    #---------------------------------------------------------------------------
    @parameter
    def Pdv_loaded(self, val):
        return val
    
    @parameter
    def P00_mu0_loaded(self, val):
        return val
    
    @parameter
    def P01_mu2_loaded(self, val):
        return val
        
    @parameter
    def P11_mu2_loaded(self, val):
        return val
        
    @parameter
    def P11_mu4_loaded(self, val):
        return val
        
    #---------------------------------------------------------------------------
    # UTILITY FUNCTIONS    
    #---------------------------------------------------------------------------
    def _get_loaded_data(self, name, k):
        """
        Evaluate the spline for the given loaded name
        """
        if name not in self._loaded_data:
            raise ValueError("Cannot evaluate loaded data for `%s`; no data found" %name)
        return self._loaded_data[name](k)
        
    #---------------------------------------------------------------------------
    def _load(self, name, k, P):
        """
        Load data into a given power attribute, as specified by power_term
        and mu_term.
        """        
        if name not in self._loadable_power_terms:
            raise ValueError("`%s` not a valid term to be loaded;"
                            " must be one of %s" %(name, self._loadable_power_terms))   
        self._loaded_data[name] = spline(k, P)
        setattr(self, name+'_loaded', True)
        
    #---------------------------------------------------------------------------
    def _unload(self, name):
        """
        Delete the given power attribute, as specified by power_term.
        """ 
        if name in self._loaded_data:
            del self._loaded_data[name]
            setattr(self, name+'_loaded', False)

    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------
