from matplotlib import pyplot as plt
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn as sns

def pkmu_normalization(driver):
    """
    Return P(k,mu) normalization, using the current
    values of `f` and `b1`, and the properly normalized
    no-wiggle linear power spectrum

    Returns
    -------
    callable :
        the P(k,mu) function callable
    """
    # get f, b1 from the fit params
    f = driver.theory.fit_params['f'].value
    b1 = driver.theory.fit_params['b1'].value
    beta = f/b1

    # freeze the power norm
    m = driver.theory.model
    power_norm = (m.sigma8_z / m.cosmo.sigma8())**2

    return lambda k, mu: (1. + beta*mu**2)**2 * b1**2 * power_norm*m.power_lin_nw(k)

def poles_normalization(driver):
    """
    Return P(k,ell=0) monopole normalization, using the current
    values of `f` and `b1`, and the properly normalized
    no-wiggle linear power spectrum

    Returns
    -------
    callable :
        the P(k,ell=0) function callable
    """
    # get f, b1 from the fit params
    f = driver.theory.fit_params['f'].value
    b1 = driver.theory.fit_params['b1'].value
    beta = f/b1

    # freeze the power norm
    m = driver.theory.model
    power_norm = (m.sigma8_z / m.cosmo.sigma8())**2

    return lambda k: (1. + 2./3*beta + 1./5*beta**2) * b1**2 * power_norm*m.power_lin_nw(k)

def plot_normalized_data(driver, offset=0., use_labels=True, labels=None, norm=None, **kwargs):
    """
    Plot the normalized data from the input `FittingDriver`

    Parameters
    ----------
    driver : FittingDriver
        the driver instances used to run the fitting procedure
    offset : float, optional
        whether to offset the individual data statistics
    use_labels : bool, optional
        whether to label the individual statistics
    **kwargs :
        additional keywords to pass to the ``errorbar()`` function
    """
    kwargs.setdefault('ls', "")
    kwargs.setdefault('capthick', 2)
    label = ""

    # check for a color list
    colors = None
    if 'color' in kwargs and isinstance(kwargs['color'], list):
        colors = kwargs['color']
    elif 'c' in kwargs and isinstance(kwargs['c'], list):
        colors = kwargs['c']
    if colors is not None:
        if len(colors) < driver.data.size:
            raise ValueError("color list must have a length of at least %d" %driver.data.size)

    # get the normalization
    if norm is None:
        if driver.mode == 'pkmu':
            norm = pkmu_normalization(driver)
        else:
            norm = poles_normalization(driver)

    # loop over the data
    for i, m in enumerate(driver.data):

        if driver.mode == 'pkmu':
            n = norm(m.k, m.mu)
        else:
            n = norm(m.k)

        # plot the measurement
        if use_labels:
            if labels is not None:
                label = labels[i]
            else:
                if driver.mode == 'pkmu':
                    label = r"$\mu = %.2g$" %(m.identifier)
                else:
                    label = r"$P_{%d}$" %(m.identifier)
        if colors is not None: kwargs['color'] = colors[i]
        kwargs['label'] = label
        yield m.k, m.power/n + offset*i, m.error/n, kwargs


def plot_normalized_theory(driver, offset=0., norm=None, label="", **kwargs):
    """
    Plot the normalized theory from the input `FittingDriver`

    Parameters
    ----------
    driver : FittingDriver
        the driver instances used to run the fitting procedure
    offset : float, optional
        whether to offset the individual data statistics
    **kwargs :
        additional keywords to pass to the ``plot()`` function
    """
    if 'capthick' in kwargs:
        kwargs.pop('capthick')

    # check for a color list
    colors = None
    if 'color' in kwargs and isinstance(kwargs['color'], list):
        colors = kwargs['color']
    elif 'c' in kwargs and isinstance(kwargs['c'], list):
        colors = kwargs['c']
    if colors is not None:
        if len(colors) < driver.data.size:
            raise ValueError("color list must have a length of at least %d" %driver.data.size)

    # get the normalization
    if norm is None:
        if driver.mode == 'pkmu':
            norm = pkmu_normalization(driver)
        else:
            norm = poles_normalization(driver)

    # check the label
    labels = [""]*driver.data.size
    if label:
        labels[0] = label

    slices = driver.data.flat_slices
    theory = driver.combined_model

    # loop over eah statistic
    for i, m in enumerate(driver.data):

        if driver.mode == 'pkmu':
            n = norm(m.k, m.mu)
        else:
            n = norm(m.k)

        # plot the theory
        if colors is not None: kwargs['color'] = colors[i]
        kwargs['label'] = labels[i]
        yield m.k, theory[slices[i]]/n + offset*i, kwargs

def get_xlabel(driver, with_latex=True):
    """
    Return the x axis label.
    """
    if with_latex:
        return r"$k \ [h \ \mathrm{Mpc}^{-1}]$"
    else:
        return "k [h/Mpc]"

def get_ylabel(driver, with_latex=True):
    """
    Return the y-axis label.
    """
    if not with_latex:
        if driver.mode == 'pkmu':
            return "normalized %s P(k,mu)" % driver.tracer_type
        else:
            return "normalized %s multipoles" % driver.tracer_type
    else:
        lab = None
        if driver.tracer_type == 'galaxy':
            lab = 'gg'
        elif driver.tracer_type == 'quasar':
            lab = 'QSO'
        assert lab is not None

        if driver.mode == 'pkmu':
            return r"$P^{\ %s} / P^\mathrm{EH} (k, \mu)$" % lab
        else:
            ell_str = ",".join([str(m.identifier) for m in driver.data])
            return r"$P^{\ %s}_{\ell=%s} / P^\mathrm{EH}_{\ell=0} (k)$" % (lab,ell_str)

def get_title(driver, with_latex=True):
    """
    Add a title to the axes with fit information
    """
    args = (driver.lnprob(), driver.Np, driver.Nb, driver.reduced_chi2())
    if with_latex:
        return r'$\ln\mathcal{L} = %.2f, \ N_p = %d, \ N_b = %d, \ \chi^2_\mathrm{red} = %.2f$' %args
    else:
        return "log-likelihood = %.2f, # params = %d, # bins = %d, reduced chi sq = %.2f" %args

def plot_fit_comparison(driver, ax=None, colors=None, use_labels=True, **kws):
    """
    Plot the model and data points for any P(k,mu) measurements
    """
    # use a Paired color
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = sns.color_palette("Paired", 14)

    # offset
    offset = -0.1 if driver.mode == 'pkmu' else 0.

    # plot the theory
    c = colors[::2][:driver.data.size]
    for res in plot_normalized_theory(driver, offset=offset, color=c, zorder=1, **kws):
        x, y, meta = res
        ax.plot(x, y, **meta)

    # plot the data
    c = colors[1::2][:driver.data.size]
    kws_ = {'offset':offset, 'use_labels':use_labels, 'zorder':10, 'color':c}
    kws_.update(kws)
    for res in plot_normalized_data(driver, **kws_):
        x, y, yerr, meta = res
        ax.errorbar(x, y, yerr, **meta)

    # format the axes
    ax.set_xlabel(get_xlabel(driver), fontsize=14)
    ax.set_ylabel(get_ylabel(driver), fontsize=16)
    ax.set_title(get_title(driver), fontsize=12)

    ncol = 1 if driver.data.size < 4 else 2
    ax.legend(loc=0, ncol=ncol)

    return ax
