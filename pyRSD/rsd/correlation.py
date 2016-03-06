"""
    correlation.py
    pyRSD.rsd.correlation

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : compute the smoothed correlation function multipoles
"""
from .. import pygcl, numpy as np
from . import ExtrapolatedPowerSpectrum

KMAX = 10.
        
def SmoothedXiMultipoles(power_model, r, ells, R=0., **kwargs):
    """
    Compute the smoothed correlation function multipoles, by extrapolating
    a power spectrum in `k` and Fourier transforming the power
    spectrum multipoles, with a Gaussian smoothing kernel
    
    :math: \Xi_\ell(s) = i^\ell \int (dq/2\pi^2) q^2 W(q*R) P_\ell(q) j_\ell(q*s)
    
    Parameters
    ----------
    model : rsd.DarkMatterSpectrum or its derived classes
        the power spectrum model instance which will be extrapolated and integrated
        over to compute the power spectrum multipoles specified by `ells`
    r : array_like
        the separations to compute the correlation function multipoles at [units: `Mpc/h`]
    power : pyRSD.rsd.power_biased
        The power spectrum class specifying the redshift-space power spectrum
        to integrate over
    ells : int or array_like
        the desired multipole numbers to compute
    R : float, optional
        the radius of the Gaussian smoothig kernel to use; default is `0`
    kwargs: passed to `ExtrapolatedPowerSpectrum` constructor
            model_func : str, optional (`Pgal`)
                the name of the function, which is a class method of ``model``
                and takes (k, mu) as its arguments
            k_lo : float 
                wavenumber to extrapolate below; default is ``model.kmin``
            k_hi : float
                wavenumber to extrapolate above; default is ``model.kmax``
            kcut_lo : float
                only include wavenumbers below this value when fitting
                the extrapolation parameters; default is `5*k_lo`
            kcut_hi : float
                only include wavenumbers above this value when fitting
                the extrapolation parameters; default is `0.5*k_hi`
    """
    if np.isscalar(ells): ells = [ells]
    
    # the extrapolated power spectrum
    extrap_model = ExtrapolatedPowerSpectrum(power_model, **kwargs)
    
    # compute the power spectrum multipoles
    k_spline = np.logspace(-5, np.log10(KMAX), 1000)
    poles = extrap_model.to_poles(k_spline, ells)
    
    toret = []
    for i, ell in enumerate(ells):
        xi = pygcl.pk_to_xi(int(ell), k_spline, poles[:,i], r, smoothing=R)
        toret.append(xi)
        
    toret = np.vstack(toret).T
    return np.squeeze(toret)
    

