from pyRSD import pygcl, numpy as np
from pyRSD.rsd._cache import parameter, cached_property
from pyRSD.rsd.transfers import PkmuGrid, TransferBase
import xarray as xr


class WedgeTransfer(TransferBase):
    """
    A transfer function object to bin :math:`P(k,\mu)` in wide :math:`\mu` bins.

    For data on a discrete grid, see :class:`GriddedWedgeTransfer`.

    Parameters
    ----------
    k : array_like
        the desired ``k`` values to evaluate the wedges at
    mu_bounds : array_like
        a list of tuples specifying (lower, upper) for each desired
        mu bin, i.e., [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    Nmu : int, optional
        the number of ``mu`` points to use when performing the multipole integration
    """
    def __init__(self, k, mu_bounds, Nmu=40):

        # handle a single bin
        if not isinstance(mu_bounds, list) and isinstance(mu_bounds, tuple):
            mu_bounds = [mu_bounds]

        # centers of the returned array
        self.mu_cen = np.array([0.5*(lo+hi) for (lo,hi) in mu_bounds])
        self.mu_bounds = mu_bounds

        # make the grid
        mu = np.linspace(0., 1., Nmu+1, endpoint=True)
        grid_k, grid_mu =  np.meshgrid(k, mu, indexing='ij')
        weights = np.ones_like(grid_k) # unity weights
        self.grid = PkmuGrid([k,mu], grid_k, grid_mu, weights)

    @parameter
    def mu_bounds(self, val):
        """
        A list of tuples specifying the lower and upper limits for each
        desired ``mu`` bin.

        .. note::
            A small tolerance is added to bins that end at :math:`\mu=1` to
            make that bin inclusive.
        """
        # if mu = 1.0 is upper bound of bin, increase by epsilon to include edge
        for i, (lo, hi) in enumerate(val):
            if np.isclose(hi, 1.0):
                val[i] = (val[i][0], 1.0005*val[i][1])
        return val

    @cached_property("mu_bounds")
    def mu_edges(self):
        """
        The edges of the ``mu`` bins desired for output.
        """
        toret = np.asarray(self.mu_bounds).ravel()
        if not all(x<=y for x, y in zip(toret, toret[1:])):
            raise ValueError("specified `mu` bounds are not monotonically increasing")
        return toret

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

        # the bin shape
        Nk = self.grid.Nk; Nmu = len(self.mu_edges)-1
        binshape =  (Nk+2, Nmu+2)

        # compute the bin indices
        k_idx = np.arange(self.grid.Nk, dtype=int)[:,None]
        dig_k = np.repeat(k_idx, self.grid.Nmu, axis=1)[self.grid.notnull] + 1
        dig_mu = np.digitize(self.grid.mu[self.grid.notnull], self.mu_edges)
        indices = np.ravel_multi_index([dig_k, dig_mu], binshape)

        # sum up power
        toret = np.zeros(binshape)
        minlength = np.prod(binshape)
        P = self.power.values
        toret.flat = np.bincount(indices, weights=P[self.grid.notnull], minlength=minlength)
        toret = toret.reshape(binshape)[1:-1, 1:-1][..., ::2]

        # and the normalization
        N = np.zeros(binshape)
        N.flat = np.bincount(indices, minlength=minlength)
        N = N.reshape(binshape)[1:-1, 1:-1][..., ::2]

        # normalize properly
        toret /= N

        # return array
        shape = (len(k), len(self.mu_cen))
        Pwedge = xr.DataArray(np.empty(shape), coords=[('k', k), ('mu', self.mu_cen)])
        for i in range(Pwedge.shape[1]):
            Pwedge[:,i] = toret[:,i]

        return Pwedge
