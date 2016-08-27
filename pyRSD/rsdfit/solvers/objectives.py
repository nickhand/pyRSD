from pyRSD.rsdfit import GlobalFittingDriver

def neg_lnlike(x=None):
    """
    Wrapper for the negative log-likelihood
    """
    driver = GlobalFittingDriver.get()
    return driver.neg_lnlike(x, use_priors=False)
    
def neg_lnprob(x=None):
    """
    Wrapper for the negative log-probability (including priors)
    """
    driver = GlobalFittingDriver.get()
    return driver.neg_lnlike(x, use_priors=True)

def lnprob(x=None):
    """
    Wrapper for the log-probability (including priors)
    """
    driver = GlobalFittingDriver.get()
    return driver.lnprob(x)
            
def gradient(x, **kwargs):
    """
    Wrapper for ``FittingDriver.gradient`` which explictly
    grabs the pickeable ``nlopt_lnlike``
    """
    driver = GlobalFittingDriver.get()
    return driver.gradient(x, **kwargs)