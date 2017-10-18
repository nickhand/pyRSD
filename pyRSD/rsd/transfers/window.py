from pyRSD.rsd.transfers import PkmuGrid
from pyRSD.rsd.transfers.grid import GriddedMultipoleTransfer
from pyRSD.rsd.window import WindowConvolution
from pyRSD import pygcl, numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import xarray as xr

class WindowFunctionTransfer(GriddedMultipoleTransfer):
    """
    A transfer function object to go from unconvolved to convolved multipoles.

    Parameters
    ----------
    window : array_like
        the window function multipoles in configuration space, as columns;
        the first column should be the separation vector ``s``
    ells : int, list of int
        the multipole numbers to compute
    kmin : float, optional
        the minimum ``k`` value on the grid used when FFTing during the convolution
    kmax : float, optional
        the maximum ``k`` value on the grid used when FFTing during the convolution
    Nk : int, optional
        the number of grid points in the ``k`` direction
    Nmu : int, optional
        the number of grid points in the ``mu`` direction (from 0 to 1)
    max_ellprime : int, optional
        the maximum multipole number to include when determining the leakage
        of higher-order multipoles into a multipole of order ``ell``
    """
    def __init__(self, window, ells, kmin=1e-4, kmax=0.7, Nk=1024, Nmu=40, max_ellprime=4):

        # make the grid
        # NOTE: we want to use the centers of the mu bins here!
        k = np.logspace(np.log10(kmin), np.log10(kmax), Nk)
        mu_edges = np.linspace(0., 1., Nmu+1)
        mu = 0.5 * (mu_edges[1:] + mu_edges[:-1])
        grid_k, grid_mu =  np.meshgrid(k, mu, indexing='ij')
        weights = np.ones_like(grid_k)
        grid = PkmuGrid([k,mu], grid_k, grid_mu, weights)

        # init the base class
        GriddedMultipoleTransfer.__init__(self, grid, ells, kmin=kmin, kmax=kmax)

        # the convolver object
        self.convolver = WindowConvolution(window[:,0], window[:,1:],
                                            max_ellprime=max_ellprime,
                                            max_ell=max(ells))

    def __call__(self, power, k_out=None, extrap=False, mcfit_kwargs={}, **kws):
        """
        Evaluate the convolved multipoles.

        Parameters
        ----------
        power : xarray.DataArray
            a DataArray holding the :math:`P(k,\mu)` values on a
            coordinate grid with ``k`` and ``mu`` dimensions.
        k_out : array_like, optional
            if provided, evaluate the convolved multipoles at these
            ``k`` values using a spline
        **kws :
            additional keywords for testing purposes

        Returns
        -------
        Pell : xarray.DataArray
            a DataArray holding the convolved :math:`P_\ell(k)` on a
            coordinate grid with ``k`` and ``ell`` dimensions.
        """
        from pyRSD.extern import mcfit

        # get testing keywords
        dry_run = kws.get('dry_run', False)
        no_convolution = kws.get('no_convolution', False)

        # get the unconvovled theory multipoles
        Pell0 = GriddedMultipoleTransfer.__call__(self, power)

        # create additional logspaced k values for zero-padding up to k=100 h/Mpc
        oldk = Pell0['k'].values
        dk = np.diff(np.log10(oldk))[0]
        newk = 10**(np.arange(np.log10(oldk.max()) + dk, 2 + 0.5*dk, dk))
        newk = np.concatenate([oldk, newk])

        # now copy over with zeros
        Nk = len(newk); Nell = Pell0.shape[1]
        Pell = xr.DataArray(np.zeros((Nk,Nell)), coords={'k':newk, 'ell':Pell0.ell}, dims=['k', 'ell'])
        Pell.loc[dict(k=Pell0['k'])] = Pell0[:]

        # do the convolution
        if not no_convolution:

            # FFT the input power multipoles
            xi = np.empty((Nk, Nell), order='F') # column-continuous
            for i, ell in enumerate(self.ells):
                P2xi = mcfit.P2xi(newk, l=ell, **mcfit_kwargs)
                rr, xi[:,i] = P2xi(Pell.sel(ell=ell).values, extrap=extrap)


            # the linear combination of multipoles
            if dry_run:
                xi_conv = xi.copy()
            else:
                xi_conv = self.convolver(self.ells, rr, xi, order='F')

            # FFTLog back
            Pell_conv = np.empty((Nk, Nell), order='F')
            for i, ell in enumerate(self.ells):
                xi2P = mcfit.xi2P(rr, l=ell, **mcfit_kwargs)
                kk, Pell_conv[:,i] = xi2P(xi_conv[:,i], extrap=extrap)

        else:
            Pell_conv = Pell

        # interpolate to k_out
        coords = coords={'ell':Pell0.ell}
        if k_out is not None:

            shape = (len(k_out), len(self.ells))
            toret = np.ones(shape) * np.nan
            for i, ell in enumerate(self.ells):
                idx = np.isfinite(newk)
                spl = spline(newk[idx], Pell_conv[idx,i])
                toret[:,i] = spl(k_out)
            coords['k'] = k_out
        else:
            toret = Pell_conv
            coords['k'] = newk

        return xr.DataArray(toret, coords=coords, dims=['k', 'ell'])
