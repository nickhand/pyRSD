from pyRSD.rsdfit import GlobalFittingDriver


def minus_lnlike(x=None, scaling=False):
    """
    Wrapper for the negative log-likelihood
    """
    driver = GlobalFittingDriver.get()
    if scaling:
        x = driver.theory.fit_params.inverse_scale(x)
    return driver.minus_lnlike(x, use_priors=False)


def minus_lnprob(x=None, scaling=False):
    """
    Wrapper for the negative log-probability (including priors)
    """
    driver = GlobalFittingDriver.get()
    if scaling:
        x = driver.theory.fit_params.inverse_scale(x)
    return driver.minus_lnlike(x, use_priors=True)


def lnprob(x=None, scaling=False):
    """
    Wrapper for the log-probability (including priors)
    """
    driver = GlobalFittingDriver.get()
    if scaling:
        x = driver.theory.fit_params.inverse_scale(x)
    return driver.lnprob(x)


def grad_minus_lnlike(x, **kwargs):
    """
    Wrapper for ``FittingDriver.gradient`` which explictly
    grabs the pickeable ``nlopt_lnlike``
    """
    scaling = kwargs.pop('scaling', False)
    driver = GlobalFittingDriver.get()

    if scaling:
        x = driver.theory.fit_params.inverse_scale(x)
    grad = driver.grad_minus_lnlike(x, **kwargs)
    if scaling:
        grad = driver.theory.fit_params.scale_gradient(grad)

    return grad
