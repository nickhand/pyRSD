from pyRSD import pygcl, numpy as np
from pyRSD.rsd.transfers import PkmuGrid, TransferBase

import xarray as xr
from scipy.special import legendre


class MultipoleTransfer(TransferBase):
    """
    A transfer function object to go from :math:`P(k,\mu)` to :math:`P_\ell(k)`.

    For data on a discrete grid, see :class:`GriddedMultipoleTransfer`.

    Parameters
    ----------
    k : array_like
        the desired ``k`` values to evaluate the multipoles at
    ells : int, list of int
        the multipole numbers we wish to compute
    Nmu : int, optional
        the number of ``mu`` points to use when performing the multipole integration
    """
    def __init__(self, k, ells, Nmu=40):

        # the multipoles
        self.ells = np.array(ells, ndmin=1)

        # use odd number of samples for simpson's rule
        if Nmu % 2 == 0: Nmu += 1

        # make the grid
        # NOTE: use mu edges so we include full integration range, [0,2]
        mu = np.linspace(0., 1., Nmu, endpoint=True)
        grid_k, grid_mu =  np.meshgrid(k, mu, indexing='ij')
        weights = np.ones_like(grid_k) # unity weights
        self.grid = PkmuGrid([k,mu], grid_k, grid_mu, weights)

    def __call__(self, power):
        """
        Parameters
        ----------
        power : xarray.DataArray
            a DataArray holding the :math:`P(k,\mu)` values on a
            coordinate grid with ``k`` and ``mu`` dimensions

        Returns
        -------
        Pell : xarray.DataArray
            a DataArray holding the :math:`P_\ell(k)` on a coordinate grid
            with ``k`` and ``ell`` dimensions.
        """
        self.power = power

        # input coordinate grid
        k = self.power['k']
        mu = self.power['mu']

        # return array
        Nk = len(self.power)
        shape = (len(k), len(self.ells))
        Pell = xr.DataArray(np.empty(shape), coords=[('k', k), ('ell', self.ells)])

        # compute each Pell
        for ell in self.ells:
            kern = (2*ell+1.)*legendre(ell)(mu)
            val = np.array([pygcl.SimpsIntegrate(mu, kern*d) for d in self.power])
            Pell.loc[dict(ell=ell)] = val

        return Pell
