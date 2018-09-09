from ... import numpy as np

#-------------------------------------------------------------------------------


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

#-------------------------------------------------------------------------------


def univariate_init(fit_params, nwalkers, draw_from='prior', logger=None):
    """
    Initialize variables from univariate prior

    Parameters
    ----------
    fit_params : ParameterSet
        the theoretical parameters
    nwalkers : int
        the number of walkers used in fitting with `emcee`
    draw_from : str, {'prior', 'posterior'}, optional
        either draw from the prior or the posterior to initialize
    logger : logging.Logger, optional
        Log output to this logger
    """
    # get the free params
    pars = fit_params.free

    # draw function: if it's from posteriors and a par has no posterior, fall
    # back to prior
    if draw_from == 'posterior':
        draw_funcs = ['get_value_from_posterior' if par.has_posterior(
        ) else 'get_value_from_prior' for par in pars]
        get_funcs = ['get_posterior' if par.has_posterior()
                     else 'get_prior' for par in pars]
    else:
        draw_funcs = ['get_value_from_prior' for par in pars]
        get_funcs = ['get_prior' for par in pars]

    # create an initial set of parameters from the priors (shape: nwalkers x npar)
    p0 = np.array([getattr(par, draw_func)(size=nwalkers)
                   for par, draw_func in zip(pars, draw_funcs)]).T

    # we do need to check if all the combinations produce realistic models
    exceed_max_try = 0
    difficult_try = 0

    # loop over each parameter
    for i, walker in enumerate(p0):
        max_try = 100
        current_try = 0

        # check the model for this set of parameters
        while True:

            # Set the values
            for par, value in zip(pars, walker):
                par.value = value

            # if it checks out, continue checking the next one
            if fit_params.check() or current_try > max_try:
                if current_try > max_try:
                    exceed_max_try += 1
                elif current_try > 50:
                    difficult_try += 1
                p0[i] = walker
                break

            current_try += 1

            # else draw a new value: for traces, we remember the index so that
            # we can take the parameters from the same representation always
            walker = []
            for ii, par in enumerate(pars):
                value = getattr(par, draw_funcs[ii])(size=1)
                walker.append(value)

    # Perhaps it was difficult to initialise walkers, warn the user
    if exceed_max_try or difficult_try:
        if logger is not None:
            args = len(p0), difficult_try, exceed_max_try
            logger.warning(("Out {} walkers, {} were difficult to initialise, and "
                            "{} were impossible: probably your priors are very "
                            "wide and allow many unphysical combinations of "
                            "parameters.").format(*args))

    # report what was used to draw from:
    if np.all(np.array(get_funcs) == 'get_prior'):
        drew_from = 'prior'
    elif np.all(np.array(get_funcs) == 'get_posterior'):
        drew_from = 'posterior'
    else:
        drew_from = 'mixture of priors and posteriors'

    return p0, drew_from

#-------------------------------------------------------------------------------


def multivariate_init(fit_params, nwalkers, draw_from='prior', logger=None):
    """
    Initialize parameters with multivariate normals

    Parameters
    ----------
    fit_params : ParameterSet
        the theoretical parameters
    nwalkers : int
        the number of walkers used in fitting with `emcee`
    draw_from : str, {'prior', 'posterior'}
        either draw from the prior or the posterior to initialize
    logger : logging.Logger, optional
        Log output to this logger
    """
    # get the free params
    pars = fit_params.free
    npars = len(pars)

    # draw function
    draw_func = 'get_value_from_' + draw_from

    # getter
    get_func = draw_from

    # check if distributions are traces, otherwise we can't generate
    # multivariate distributions
    for par in pars:
        this_dist = getattr(par, get_func)
        if this_dist is None:
            raise ValueError(("No {} defined for parameter {}, cannot "
                              "initialise "
                              "multivariately").format(draw_from, par.name))
        if not this_dist.name == 'trace':
            raise ValueError(("Only trace distributions can be used to "
                              "generate multivariate walkers ({} "
                              "distribution given for parameter "
                              "{})").format(this_dist, par.name))

    # extract averages and sigmas
    averages = [getattr(par, get_func).loc for par in pars]
    sigmas = [getattr(par, get_func).scale for par in pars]

    # Set correlation coefficients
    cor = np.zeros((npars, npars))
    for i, ipar in enumerate(pars):
        for j, jpar in enumerate(pars):
            prs = scipy.stats.pearsonr(getattr(ipar, get_func).trace,
                                       getattr(jpar, get_func)().trace)[0]
            cor[i, j] = prs * sigmas[i] * sigmas[j]

    # sample is shape nwalkers x npars
    sample = np.random.multivariate_normal(averages, cor, nwalkers)

    # Check if all initial values satisfy the limits and priors. If not,
    # draw a new random sample and check again. Don't try more than 100 times,
    # after that we just need to live with walkers that have zero probability
    # to start with...
    for i, walker in enumerate(sample):
        max_try = 100
        current_try = 0
        while True:
            # adopt the values in the system
            for par, value in zip(pars, walker):
                par.value = value
                sample[i] = walker
            # perform the check; if it doesn't work out, retry
            if not fit_params.check() and current_try < max_try:
                walker = np.random.multivariate_normal(averages, cor, 1)[0]
                current_try += 1
            else:
                break
        else:
            if logger is not None:
                logger.warning(
                    "Walker {} could not be initalised with valid parameters".format(i))

    return sample, draw_from
