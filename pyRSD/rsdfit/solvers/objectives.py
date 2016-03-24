_driver = None
def get_rsdfit_driver():
    return _driver
    
def set_rsdfit_driver(d):
    global _driver
    _driver = d
    
def neg_lnlike(x=None):
    """
    Wrapper for the negative log-likelihood
    """
    return _driver.neg_lnlike(x, use_priors=False)
    
def neg_lnprob(x=None):
    """
    Wrapper for the negative log-probability (including priors)
    """
    return _driver.neg_lnlike(x, use_priors=True)

def lnprob(x=None):
    """
    Wrapper for the log-probability (including priors)
    """
    return _driver.lnprob(x)
            
def gradient(x, **kwargs):
    """
    Wrapper for ``FittingDriver.gradient`` which explictly
    grabs the pickeable ``nlopt_lnlike``
    """
    return _driver.gradient(neg_lnlike, x, **kwargs)