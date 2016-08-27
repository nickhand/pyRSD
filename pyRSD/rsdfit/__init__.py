params_filename = 'params.dat'
model_filename = 'model.pickle'

class GlobalFittingDriver(object):
    """
    The global :class:`~pyRSD.rsdfit.driver.FittingDriver` instance
    """
    _instance = None 
    
    @classmethod
    def get(cls):
        """
        Get global driver, raising an exception if it is None
        """
        if cls._instance is None:
            raise ValueError("global driver has not been set yet")
        return cls._instance
        
    @classmethod
    def set(cls, driver):
        """
        Set the global driver to the input value
        """
        cls._instance = driver

import logging
from .util.rsd_logging import MPILoggerAdapter

# import the drivers and run functions
from .driver import FittingDriver

# import the specific modules as well
from . import data
from . import solvers
from . import parameters
from . import results
from . import theory
from . import util
