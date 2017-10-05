from pyRSD.rsd._cache import Cache, parameter
from pyRSD import numpy as np
import warnings

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
                            "recommended; should have >= 40 to avoid inaccuracies"))
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
    def from_plaintext(cls, filename):
        """
        Return a :class:`PkmuGrid` instance, reading the data from a
        plaintext file.
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
        cls = self.__class__.__name__
        args = (cls, self.Nk, self.Nmu, (1.-1.*self.notnull.sum()/np.prod(self.shape))*100.)
        return "<%s: size (%dx%d), %.2f%% empty grid points>" %args

    def __repr__(self):
        return self.__str__()

class TransferBase(Cache):
    """
    A base class for a P(k,mu) transfer function.
    """
    def __init__(self):
        Cache.__init__(self)

    @property
    def Nk(self):
        return self.grid.Nk

    @property
    def Nmu(self):
        return self.grid.Nmu

    @property
    def gridshape(self):
        return self.grid.shape

    @property
    def N1(self):
        return self.Nk

    @property
    def flatk(self):
        return self.grid.k[self.grid.notnull]

    @property
    def flatmu(self):
        return self.grid.mu[self.grid.notnull]

    @parameter
    def power(self, val):
        """
        The power array holding :math:`P(k,\mu)` on the grid.

        Shape is (:attr:`Nk`, :attr`Nmu`), with NaNs for any null grid points.
        """
        import xarray as xr

        # get the data as a numpy array
        if isinstance(val, xr.DataArray):
            val = val.values

        # create a DataArray on the grid with null values
        toret = np.ones(self.gridshape)*np.nan
        coords = {'k':self.grid.k_cen, 'mu':self.grid.mu_cen}
        toret = xr.DataArray(toret, coords=coords, dims=['k', 'mu'])

        if val is None:
            return toret

        # the number of grid points that are not null
        valid = self.grid.notnull.sum()

        # if flat data, we are setting the valid data points
        if np.ndim(val) == 1:
            if len(val) == valid:
                toret.values[self.grid.notnull] = val
            else:
                raise ValueError("if 1D array is passed for ``power``, must have length %d" %valid)
        else:
            toret.values[self.grid.notnull] = val[self.grid.notnull]
        return toret

from .grid import GriddedWedgeTransfer, GriddedMultipoleTransfer
from .poles import MultipoleTransfer
from .window import WindowFunctionTransfer

gridded_transfers = (GriddedWedgeTransfer, GriddedMultipoleTransfer)
