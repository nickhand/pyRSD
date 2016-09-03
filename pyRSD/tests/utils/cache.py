from ... import os, sys
from ...rsd import load_model, OutdatedModelWarning
import logging
import warnings

logging.basicConfig(level=logging.DEBUG)

def user_cache_dir(appname):
    r"""

    This function is copied from:
    https://github.com/pypa/pip/blob/master/pip/utils/appdirs.py

    Return full path to the user-specific cache dir for this application.
    
    Parameters
    ----------
    appname : str 
        the name of application
    
    Notes
    -----
    Typical user cache directories are:
        
        - Mac OS X: ~/Library/Caches/<AppName>
        - Unix: ~/.cache/<AppName> (XDG default)
    """
    from os.path import expanduser
    WINDOWS = (sys.platform.startswith("win") or
               (sys.platform == 'cli' and os.name == 'nt'))

    if WINDOWS:
        raise OSError("sorry, not supported on Windows")
    elif sys.platform == "darwin":
        # Get the base path
        path = expanduser("~/Library/Caches")

        # Add our app name to it
        path = os.path.join(path, appname)
    else:
        # Get the base path
        path = os.getenv("XDG_CACHE_HOME", expanduser("~/.cache"))

        # Add our app name to it
        path = os.path.join(path, appname)

    return path

cache_dir = user_cache_dir('pyRSD')

class cache_manager():
    """
    Context for managing loading/saving RSD models to cache
    """                
    def __init__(self, model, filename, autocache=True):
        """
        Parameters
        ----------
        model : DarkMatterSpectrum or subclass
            the RSD model that will be initialized and used if
            no existing model exists in the cache
        filename : str 
            the name of the file to look for in the cache directory
        autocache : bool, optional
            whether to save the new model to the cache directory if
            the current model is out of date or non-existent; default
            is ``True``
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.model      = model
        self.filename   = filename
        self.save_model = False
        self.autocache  = autocache
        self.logger     = logging.getLogger("cache-manager")
        
        
    def __enter__(self):
        
        # try to load the model 
        path = os.path.join(cache_dir, self.filename)

        if os.path.exists(path):
            with warnings.catch_warnings(record=True) as w:
                exception = False
                try: model = load_model(path)
                except: exception = True
                    
                if exception or len(w) and issubclass(w[-1].category, OutdatedModelWarning):
                    self.logger.info("the model in the cache is out of date")
                    if len(w): self.logger.info(str(w[-1].message))
                    self.save_model = True
                else:
                    self.model = model
                    self.logger.info("successfully loaded cached model with name '%s'" %path)
        else:
            self.logger.info("no model in cache with name '%s'" %path)
            self.save_model = True
        
        # initialize the model
        if self.save_model:
            self.logger.info("initializing new model")
            self.model.initialize()
        
        return self.model
            
    def __exit__(self, type, value, traceback):
        
        if self.save_model and self.autocache:
            path = os.path.join(cache_dir, self.filename)
            self.logger.info("dumping new model to '%s'" %path)
            self.model.to_npy(path)
