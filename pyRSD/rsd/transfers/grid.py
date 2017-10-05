from pyRSD import numpy as np
from pyRSD.rsd._cache import parameter, cached_property
from pyRSD.rsd.transfers import TransferBase

import xarray as xr
from scipy.special import legendre
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class GriddedWedgeTransfer(TransferBase):
    """
    Class to facilitate the manipulations of :math:`P(k,\mu)` measurements
    on a :math:`(k,\mu)` grid.

    Parameters
    ----------
    grid : :class:`PkmuGrid`
        the grid instance defining the :math:`(k,\mu)` grid
    mu_bounds : array_like
        a list of tuples specifying (lower, upper) for each desired
        mu bin, i.e., [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    kmin : float, optional
        the minimum wavenumber to include
    kmax : float, optional
        the maximum wavenumber to include
    """
    def __init__(self, grid, mu_bounds, kmin=-np.inf, kmax=np.inf):
        self.grid = grid

        # handle a single bin
        if not isinstance(mu_bounds, list) and isinstance(mu_bounds, tuple):
            mu_bounds = [mu_bounds]

        # centers of the returned array
        self.mu_cen = np.array([0.5*(lo+hi) for (lo,hi) in mu_bounds])
        self.k_cen = self.grid.k_cen

        self.mu_bounds = mu_bounds
        self.kmin = kmin
        self.kmax = kmax

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

    @parameter
    def kmin(self, val):
        """
        The minimum wavenumber ``k`` to include.

        Shape can be either a float or array of length :attr:`N2`
        """
        if val is None: val = -np.inf
        toret = np.empty(self.N2)
        toret[:] = val
        return toret

    @parameter
    def kmax(self, val):
        """
        The maximum wavenumber ``k`` to include.

        Shape can be either a float or array of length :attr:`N2`.
        """
        if val is None: val = np.inf
        toret = np.empty(self.N2)
        toret[:] = val
        return toret

    @cached_property("mu_bounds")
    def mu_edges(self):
        """
        The edges of the ``mu`` bins desired for output.
        """
        toret = np.asarray(self.mu_bounds).ravel()
        if not all(x<=y for x, y in zip(toret, toret[1:])):
            raise ValueError("specified `mu` bounds are not monotonically increasing")
        return toret

    @property
    def size(self):
        """
        The size of the flattened (valid) grid points, i.e., the number of
        (k,mu) or (k,ell) data points.
        """
        return len(self.flatcoords[0])

    @property
    def N1(self):
        """
        First axis is ``k``.
        """
        return self.grid.Nk

    @property
    def N2(self):
        """
        Second axis is the ``mu`` bins.
        """
        return len(self.mu_bounds)

    @property
    def flatcoords(self):
        """
        A list holding the flattened coordinates (``k``, ``mu``), where
        ``mu`` refers to the center of the ``mu`` wedge.

        .. note::
            These coordinates corresponds to the final power with NaNs removed.
        """
        return [x[self.in_k_range] for x in self.coords]

    @property
    def coords(self):
        """
        The (``k``,``mu``) coordinates with shape (``N1``, ``N2``).
        """
        return np.broadcast_arrays(self.k_cen[:,None], self.mu_cen[None,:])

    @cached_property("kmin", "kmax")
    def in_k_range(self):
        """
        A boolean array with elements set to ``True`` specifying the
        elements on the underlying (k,mu) grid within the desired
        :attr:`kmin` and :attr:`kmax`.
        """
        k_cen = self.grid.k_cen
        return (k_cen[:,None] >= self.kmin)&(k_cen[:,None] <= self.kmax)

    @cached_property("mu_edges")
    def binshape(self):
        """
        Convenience attribute to store the dimensions of the bin counting arrays.
        """
        Nk = self.grid.Nk; Nmu = len(self.mu_edges)-1
        return (Nk+2, Nmu+2)

    @cached_property("mu_edges")
    def mu_indices(self):
        """
        Multi-index for re-binning the ``mu`` grid.
        """
        k_idx = np.arange(self.grid.Nk, dtype=int)[:,None]
        dig_k = np.repeat(k_idx, self.grid.Nmu, axis=1)[self.grid.notnull] + 1
        dig_mu = np.digitize(self.grid.mu[self.grid.notnull], self.mu_edges)
        return np.ravel_multi_index([dig_k, dig_mu], self.binshape)

    def sum(self, d):
        """
        Sum the input data over the `mu` bins specified by `mu_edges`

        Parameters
        ----------
        d : array_like, (``grid.Nk``, ``grid.Nmu``)
            the data array to sum over the grid
        """
        toret = np.zeros(self.binshape)
        minlength = np.prod(self.binshape)
        toret.flat = np.bincount(self.mu_indices, weights=d[self.grid.notnull], minlength=minlength)
        return toret.reshape(self.binshape)[1:-1, 1:-1][..., ::2]

    def average(self, d, w=None):
        """
        Average the input data, with optional weights, as a function of ``mu``
        on the grid, using :attr:`mu_edges` to specify the edges of the new
        ``mu`` bins.

        Parameters
        ----------
        d : array_like, (``grid.Nk``, ``grid.Nmu``)
            the data array to average over the grid
        w : array_like, (``grid.Nk``, ``grid.Nmu``)
            optional weight array to apply before averaging
        """
        if w is None: w = np.ones(d.shape)
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.sum(d*w) / self.sum(w)

    def __call__(self, power):
        """
        Return the ``power`` attribute re-binned into the ``mu`` bins
        corresponding to `mu_edges`, optionally flattening the return array

        Parameters
        ----------
        power : array_like
            the power values defined on the grid -- can have shape
            (Nk,Nmu) or (N,), in which case it is interpreted
            as the values at all valid grid points
        """
        # make sure power is the right size
        self.power = power

        # all null?
        if not np.sum(~self.power.isnull()):
            raise ValueError("please set ``power`` arrary; all NaN right now")

        # compute the wedges
        wedges = self.average(self.power.values, self.grid.modes)

        # convert to a DataArray
        k_cen = self.grid.k_cen
        mu_cen = self.mu_cen # wedge centers
        toret = xr.DataArray(wedges, coords={'k':k_cen, 'mu':mu_cen}, dims=['k', 'mu'])

        # make sure there are no null values in the valid k-range!
        if toret.isnull().values[self.in_k_range].any():
            raise ValueError("NaN values in GriddedWedgeTransfer result within valid k range!")

        # set out of range to null
        toret.values[~self.in_k_range] = np.nan

        return toret


class GriddedMultipoleTransfer(GriddedWedgeTransfer):
    """
    Class to facilitate the manipulations of multipole measurements
    on an underlying :math:`(k, \mu)` grid.

    Parameters
    ----------
    grid : :class:`PkmuGrid`
        the grid instance defining the :math:`(k,\mu)` grid
    ells : array_like
        a list of multipole numbers to return
    kmin : float, optional
        the minimum wavenumber to include
    kmax : float, optional
        the maximum wavenumber to include
    """
    def __init__(self, grid, ells, kmin=-np.inf, kmax=np.inf):

        self.ells = ells
        mu_bounds = [(0., 1.)] # a single mu bin from 0 to 1
        GriddedWedgeTransfer.__init__(self, grid, mu_bounds, kmin=kmin, kmax=kmax)

    @parameter
    def ells(self, val):
        """
        The multipole numbers to compute -- this corresponds to the
        second axis (columns).
        """
        return np.array(val, ndmin=1)

    @cached_property()
    def k_mean(self):
        """
        The ``k`` values averaged over the ``mu`` dimension
        """
        return self.average(self.grid.k, self.grid.modes)

    @property
    def N2(self):
        """
        The number of multipoles (2nd axis).
        """
        return len(self.ells)

    @cached_property("ells")
    def legendre_weights(self):
        """
        Legendre weighting for the multipoles.
        """
        return np.array([(2*ell+1)*legendre(ell)(self.grid.mu) for ell in self.ells])

    @property
    def coords(self):
        """
        The :math:`(k,\ell)`` coordinates with shape (``N1``, ``N2``)
        """
        return np.broadcast_arrays(self.k_cen[:,None], self.ells[None,:])

    def __call__(self, power):
        """
        Return the Legendre-weighted mean power on the :math:`(k, \mu)` grid.

        Parameters
        ----------
        power : array_like
            the power values defined on the grid -- can have shape
            (Nk,Nmu) or (N,), in which case it is interpreted
            as the values at all valid grid points
        """
        # make sure power is the right size
        self.power = power

        # all null?
        if not np.sum(~self.power.isnull()):
            raise ValueError("please set ``power`` arrary; all NaN right now")

        # compute the poles
        tobin = self.legendre_weights*self.power.values
        poles = np.asarray([np.squeeze(self.average(d, self.grid.modes)) for d in tobin]).T

        # convert to a DataArray
        coords = {'k':self.grid.k_cen, 'ell':self.ells}
        toret = xr.DataArray(poles, coords=coords, dims=['k', 'ell'])

        # make sure there are no null values in the valid k-range!
        if toret.isnull().values[self.in_k_range].any():
            raise ValueError("NaN values in GriddedMultipoleTransfer result within valid k range!")

        # set out of range to null
        toret.values[~self.in_k_range] = np.nan

        return toret
