from .. import numpy as np
from ._cache import Cache, parameter, cached_property
from scipy.special import legendre
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import warnings

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------
def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def _flatten(arr):
    """
    Flatten and remove any null entries
    """
    flat = arr.ravel(order='F')
    return flat[np.isfinite(flat)]

def trim_zeros_indices(filt):
    """
    Return the indices (first, last) specifying
    the indices to trim leading or trailing zeros
    """
    first = 0
    for i in filt:
        if i != 0.:
            break
        else:
            first = first + 1
    last = len(filt)
    for i in filt[::-1]:
        if i != 0.:
            break
        else:
            last = last - 1
    return first, last

#------------------------------------------------------------------------------
# P(k,mu) grid
#------------------------------------------------------------------------------
class PkmuGrid(object):
    """
    A class to represent a 2D grid of (``k``, ``mu``).

    The ``k`` axis is 0 and the ``mu`` axis is 1.

    Parameters
    ----------
    coords : list
        list of (k_cen, mu_cen) representing the bin centers
        of the grid coordinates
    k : array_like, (Nk, Nmu)
        the mean values of `k` at each grid point
    mu : array_like, (Nk, Nmu)
        the mean values of `mu` at each grid point
    modes : array_like, (Nk, Nmu)
        the number of modes in each grid point
    """
    def __init__(self, coords, k, mu, modes):
        if np.shape(mu)[1] < 40:
            warnings.warn(("initializing PkmuGrid with fewer mu bins than "
                            "recommended; should have >= 40"))
        self.k_cen, self.mu_cen = coords
        self.k = k
        self.mu = mu
        self.modes = modes

    def to_plaintext(self, filename):
        """
        Write out the grid to a plaintext file.
        """
        with open(filename, 'wb') as ff:
            header = "%d %d\n" %(self.Nk, self.Nmu)
            ff.write(header.encode())
            np.savetxt(ff, self.k_cen)
            np.savetxt(ff, self.mu_cen)
            d = np.dstack([self.k, self.mu, self.modes])
            np.savetxt(ff, d.reshape((-1, 3)))

    @classmethod
    def from_pkmuresult(cls, pkmu):
        """
        Convienence method to return a ``PkmuGrid`` from a
        ``nbodykit.PkmuResult`` instance.
        """
        coords = [pkmu.coords['k'], pkmu.coords['mu']]
        return cls(coords, pkmu['k'], pkmu['mu'], pkmu['modes'])

    @classmethod
    def from_structured(cls, coords, data):
        """
        Return a ``PkmuGrid`` from a list of coordinates and
        a structured array.

        Parameters
        ----------
        coords : list
            list of (k_cen, mu_cen) representing the bin centers
            of the grid coordinates
        data : array_like
            structured array with `k`, `mu`, and `modes` fields
        """
        return cls(coords, data['k'], data['mu'], data['modes'])

    @classmethod
    def from_plaintext(cls, filename):
        """
        Return a `PkmuGrid` instance, reading the data from a plaintext.
        """
        with open(filename, 'r') as ff:
            lines = ff.readlines()

        Nk, Nmu = [int(l) for l in lines[0].split()]
        k_cen = np.array([float(l) for l in lines[1:Nk+1]])
        mu_cen = np.array([float(l) for l in lines[Nk+1:Nk+Nmu+1]])
        data = np.array([[float(ll) for ll in l.split()] for l in lines[Nk+Nmu+1:]]).reshape((Nk, Nmu, 3))

        return cls([k_cen, mu_cen], *np.rollaxis(data, 2))

    @property
    def shape(self):
        return self.k.shape

    @property
    def Nk(self):
        return self.shape[0]

    @property
    def Nmu(self):
        return self.shape[1]

    @property
    def notnull(self):
        return (np.isfinite(self.modes))&(self.modes > 0)

    def __str__(self):
        """
        Builtin string representation.
        """
        cls = self.__class__.__name__
        args = (cls, self.Nk, self.Nmu, (1.-1.*self.notnull.sum()/np.prod(self.shape))*100.)
        return "<%s: size (%dx%d), %.2f%% empty grid points>" %args

    def __repr__(self):
        return self.__str__()

class PkmuTransfer(Cache):
    """
    Class to facilitate the manipulations of P(k,mu) measurements
    on a (k, mu) grid.

    Parameters
    ----------
    grid : :class:`~pyRSD.rsd.PkmuGrid`
        the grid instance defining the (k,mu) grid
    power : array_like
        the power values defined on the grid -- can have shape
        (Nk,Nmu) or (N,), in which case it is interpreted
        as the values at all valid grid points
    """
    def __init__(self, grid, power=None):
        self.grid      = grid
        self.power     = power

    @classmethod
    def from_structured(cls, coords, data, mu_bounds, **kwargs):
        """
        Convenience function to teturn a `PkmuTransfer` object
        from the coords+data instead of a `PkmuGrid`.
        """
        grid = PkmuGrid.from_structured(coords, data)
        return cls(grid, mu_bounds, **kwargs)

    #--------------------------------------------------------------------------
    # parameters
    #--------------------------------------------------------------------------
    @parameter
    def power(self, val):
        """
        The power array holding P(k,mu) on the grid.

        Shape is (grid.Nk, grid.Nmu), with NaNs for any null grid points.
        """
        toret = np.ones(self.grid.shape)*np.nan
        if val is None:
            return toret

        valid = self.grid.notnull.sum()
        if np.ndim(val) == 1:
            if len(val) == valid:
                toret[self.grid.notnull] = val
            else:
                raise ValueError("if 1D array is passed for ``power``, must have length %d" %valid)
        else:
            toret[self.grid.notnull] = val[self.grid.notnull]
        return toret

    @parameter
    def mu_bounds(self, val):
        """
        A list of tuples specifying the lower and upper limits for each
        desired ``mu`` bin.
        """
        # if mu = 1.0 is upper bound of bin, increase by epsilon to include edge
        for i, (lo, hi) in enumerate(val):
            if np.isclose(hi, 1.0):
                val[i] = (val[i][0], 1.005*val[i][1])
        return val

    @parameter
    def kmin(self, val):
        """
        The minimum wavenumber ``k`` to include.

        Shape can be either a float or array of length ``N2``
        """
        if val is None: val = -np.inf
        toret = np.empty(self.N2)
        toret[:] = val
        return toret

    @parameter
    def kmax(self, val):
        """
        The maximum wavenumber ``k`` to include.

        Shape can be either a float or array of length ``N2``.
        """
        if val is None: val = np.inf
        toret = np.empty(self.N2)
        toret[:] = val
        return toret

    #--------------------------------------------------------------------------
    # cached properties
    #--------------------------------------------------------------------------
    @cached_property("mu_bounds")
    def mu_edges(self):
        """
        The edges of the ``mu`` bins desired for output.
        """
        toret = np.asarray(self.mu_bounds).ravel()
        if not non_decreasing(toret):
            raise ValueError("specified `mu` bounds are not monotonically increasing")
        return toret

    @cached_property()
    def size(self):
        """
        The size of the flattened (valid) grid points, i.e., the number of
        (k,mu) or (k,ell) data points.
        """
        return len(self.coords_flat[0])

    @cached_property()
    def N1(self):
        return self.grid.Nk

    @cached_property("mu_bounds")
    def N2(self):
        return len(self.mu_bounds)

    @cached_property("coords")
    def coords_flat(self):
        """
        List of the flattened coordinates, with NaNs removed.
        """
        return [_flatten(x) for x in self.coords]

    @cached_property("mu_edges", "in_range_idx")
    def coords(self):
        """
        The (``k``,``mu``) coordinates with shape (``N1``, ``N2``).
        """
        # broadcast (k_cen, mu_cen) to the shape (N1, N2)
        idx = self.in_range_idx
        k_cen = np.ones((self.N1, self.N2)) * self.grid.k_cen[:,None]
        mu_cen = 0.5*(self.mu_edges[1:] + self.mu_edges[:-1])[::2]
        mu_cen = np.ones((self.N1, self.N2)) * mu_cen[None,:]

        # restrict the k-range and return
        return self.restrict_k(k_cen), self.restrict_k(mu_cen)

    @cached_property("kmin", "kmax")
    def in_range_idx(self):
        """
        A boolean array with elements set to ``True`` specifying the
        elements on the underlying (k,mu) grid within the desired
        :attr:`kmin` and :attr:`kmax`.
        """
        k_cen = self.grid.k_cen
        return np.squeeze((k_cen[:,None] >= self.kmin)&(k_cen[:,None] <= self.kmax))

    @cached_property("mu_edges")
    def ndims(self):
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
        return np.ravel_multi_index([dig_k, dig_mu], self.ndims)

    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def sum(self, d):
        """
        Sum the input data over the `mu` bins specified by `mu_edges`

        Parameters
        ----------
        d : array_like, (``grid.Nk``, ``grid.Nmu``)
            the data array to sum over the grid
        """
        toret = np.zeros(self.ndims)
        minlength = np.prod(self.ndims)
        toret.flat = np.bincount(self.mu_indices, weights=d[self.grid.notnull], minlength=minlength)
        return np.squeeze(toret.reshape(self.ndims)[1:-1, 1:-1][..., ::2])

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

    def restrict_k(self, arr):
        """
        Restrict the k-range of ``arr`` to the specified :attr:`kmin` and
        :attr:`kmax`.

        Parameters
        ----------
        arr : array_like (``grid.Nk``, ``grid.Nmu``)
            trim the array defined on the grid to the desired k-range,
            filling with NaNs if not aligned properly
        """
        cnts = self.in_range_idx.astype(int)
        first, last = trim_zeros_indices(cnts.sum(axis=1))

        toret = arr.copy()
        toret[~self.in_range_idx] = np.nan
        return toret[first:last,...]

    def __call__(self, mu_bounds, kmin=-np.inf, kmax=np.inf, full_grid_output=False):
        """
        Return the ``power`` attribute re-binned into the ``mu`` bins
        corresponding to `mu_edges`, optionally flattening the return array

        Notes
        -----
        The returned array will be restricted to (`kmin`, `kmax`)

        Parameters
        ----------
        mu_bounds : array_like
            a list of tuples specifying (lower, upper) for each desired
            mu bin, i.e., [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        kmin : float
            the minimum wavenumber to include
        kmax : float
            the maximum wavenumber to include
        full_grid_output : bool
            if ``True``, do not restrict the ``k`` range
        """
        if not isinstance(mu_bounds, list):
            mu_bounds = [mu_bounds]

        # set the state
        self.mu_bounds = mu_bounds
        self.kmin = kmin
        self.kmax = kmax

        if not (~np.isnan(self.power)).sum():
            raise ValueError("please set ``power`` arrary; all NaN right now")

        toret = self.average(self.power, self.grid.modes)
        if not full_grid_output:
            toret = self.restrict_k(toret)

        return np.squeeze(toret)

class PolesTransfer(PkmuTransfer):
    """
    Class to facilitate the manipulations of multipole measurements
    on an underlying (k, mu) grid

    Parameters
    ----------
    grid : :class:`~pyRSD.rsd.PkmuGrid`
        the grid instance defining the (k,mu) grid
    power : array_like
        the power values defined on the grid -- can have shape
        (Nk,Nmu) or (N,), in which case it is interpreted
        as the values at all valid grid points
    """
    def __init__(self, grid, power=None):
        super(PolesTransfer, self).__init__(grid, power=power)
        self.mu_bounds = [(0., 1.)] # one single mu bin

    @classmethod
    def from_structured(cls, coords, data, ells, **kwargs):
        """
        Convenience function to teturn a `PolesTransfer` object
        from the coords+data instead of a `PkmuGrid`.
        """
        grid = PkmuGrid.from_structured(coords, data)
        return cls(grid, ells, **kwargs)

    #--------------------------------------------------------------------------
    # parameters
    #--------------------------------------------------------------------------
    @parameter
    def ells(self, val):
        """
        The multipole numbers to compute -- this corresponds to the
        second axis (columns).
        """
        return np.asarray(val)

    @cached_property()
    def k_mean(self):
        """
        The ``k`` values averaged over the `mu` dimension
        """
        return self.average(self.grid.k, self.grid.modes)

    @cached_property("ells")
    def N2(self):
        return len(self.ells)

    @cached_property("ells")
    def legendre_weights(self):
        """
        Legendre weighting for multipoles.
        """
        return np.array([(2*ell+1)*legendre(ell)(self.grid.mu) for ell in self.ells])

    @cached_property("ells", "in_range_idx")
    def coords(self):
        """
        The (k,ell) coordinates with shape (`N1`, `N2`)
        """
        # broadcast (k_cen, ells) to the shape (N1, N2)
        idx = self.in_range_idx
        k_cen = np.ones((self.N1, self.N2)) * self.grid.k_cen[:,None]
        ells = np.ones((self.N1, self.N2)) * self.ells[None,:]

        # restrict the k-range and return
        return self.restrict_k(k_cen), self.restrict_k(ells)

    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def __call__(self, ells, kmin=-np.inf, kmax=np.inf, full_grid_output=False):
        """
        Return the multipoles corresponding to `ells`, by weighting the
        `power` attribute by the appropriate Legendre weights and summing
        over the `mu` dimension

        Notes
        -----
        The returned array will be restricted to (`kmin`, `kmax`)

        Parameters
        ----------
        ells : array_like
            a list of multipole numbers to return
        kmin : float
            the minimum wavenumber to include
        kmax : float
            the maximum wavenumber to include
        full_grid_output : bool
            if ``True``, do not restrict the ``k`` range
        """
        if not (~np.isnan(self.power)).sum():
            raise ValueError("please set ``power`` arrary; all NaN right now")

        # set the state
        self.ells = ells
        self.kmin = kmin
        self.kmax = kmax

        # average over grid
        tobin = self.legendre_weights*self.power
        toret = np.asarray([self.average(d, self.grid.modes) for d in tobin]).T

        # restrict k range
        if not full_grid_output:
            toret = self.restrict_k(toret)

        # return
        return np.squeeze(toret)
