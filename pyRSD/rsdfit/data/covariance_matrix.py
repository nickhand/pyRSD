import numpy as np
from pyRSD.rsd._cache import Cache, cached_property, parameter
import itertools
from collections import OrderedDict
from six import string_types
import xarray as xr
from . import indexing, tools

def is_array_like(d, shape):
    return not np.isscalar(d) and np.shape(d) == shape

def is_square(d):
    shape = np.shape(d)
    return np.ndim(d) == 2 and all(s == shape[0] for s in shape)

def condensed_to_full(data, N):
    toret = np.empty((N, N))

    # set the upper triangle and the diagonal
    upper_inds = np.triu_indices(N)
    toret[upper_inds] = data[:]

    # and now set the lower triangle
    i, j = np.tril_indices(N)
    toret[i, j] = toret[j, i]
    return toret

def flat_and_nonnull(arr):
    """
    Flatten the input array using `Fortran` format
    (i.e., col #1 then col #2, etc), and remove
    any NaNs along the way
    """
    flat = arr.ravel(order='F')
    return flat[np.isfinite(flat)]

class CovarianceMatrix(Cache):
    """
    Class to represent a covariance matrix. The class has coordinates associated
    with the covariance matrix, which can be used to index the matrix
    """

    def __init__(self, data, coords=[], names=[], attrs=None, verify=True):
        """
        Parameters
        ----------
        data : array_like, (N, N) or (N,)
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        coords : list, optional
            A list of coordinates to associate with the matrix index. Must
            have same shape as input data
        names : list
            A list of names to associate with the coordinates of the index. Must
            have same shape as supplied coordinates. If coordinates are
            provided and no names, they are named `coord_0`, `coord_1`, etc
        attrs : dict_like or None, optional
            Attributes to assign to the new variable. By default, an empty
            attribute dictionary is initialized.
        verify : bool, optional
            If `True`, verify that the matrix is positive-semidefinite and
            symmetric
        """
        # make local copies
        data = np.asarray(data).copy()

        # make a diagonal 2D matrix from input
        if data.ndim == 1: data = np.diag(data)

        # has to be square
        if not is_square(data):
            raise ValueError("sorry, the input data must at least be square")

        # check basic properties
        if verify: self.check_properties(data)

        # initialize the backend
        self._setup(data, coords=coords, names=names)
        self.inverse_rescaling = 1.0

        # setup the attrs for metadata
        if attrs is None: attrs = OrderedDict()
        self.attrs = attrs

    #--------------------------------------------------------------------------
    # Parameters
    #--------------------------------------------------------------------------
    @parameter
    def _data(self, val):
        """
        The main data attribute, which stores the upper triangle + diagonal
        elements of the symmetric matrix
        """
        return val

    @parameter
    def attrs(self, val):
        """
        Dictionary of local attributes on this variable.
        """
        return val

    @parameter
    def inverse_rescaling(self, val):
        """
        Rescale the inverse of the covariance matrix by this factor
        """
        return val

    #--------------------------------------------------------------------------
    # Internal functions
    #--------------------------------------------------------------------------
    @classmethod
    def __construct_direct__(cls, data, coords=[], names=[], attrs=None):
        """
        Shortcut around __init__ for internal use
        """
        obj = cls.__new__(cls)
        obj._setup(data, coords=coords, names=names)
        obj.inverse_rescaling = 1.0
        if attrs is None:
            obj.attrs = OrderedDict()
        else:
            obj.attrs = attrs.copy()

        return obj

    def __finalize__(self, data, indices):
        """
        Finalize a new instance, by slicing the `coords` and `attrs` and
        returning the new instance
        """
        attrs = self.__slice_attrs__(indices)
        coords = self.__slice_coords__(indices)
        return self.__class__.__construct_direct__(data, coords=coords, names=self.dims, attrs=attrs)

    def __write_attrs__(self, ff):
        """
        Write out the `attrs` dictionary to an open file
        """
        if len(self.attrs):
            ff.write(("%d\n" %len(self.attrs)).encode())
            for k, v in self.attrs.items():
                if np.isscalar(v):
                    N = (0,)
                    cast = type(v).__name__
                else:
                    N = np.shape(v)
                    cast = np.asarray(v).dtype.type.__name__
                N = (0,) if np.isscalar(v) else np.shape(v)
                args = (k, max(1, np.product(N)), cast, " ".join(map(str, N)))
                ff.write(("%s %d %s %s\n" %args).encode())

                if np.isscalar(v):
                    ff.write(("%s\n" %str(v)).encode())
                else:
                    np.savetxt(ff, v.ravel())

    @classmethod
    def __read_attrs__(cls, ff):
        """
        Read the `attrs` dictionary from an open file, returning
        a new ``OrderedDict``
        """
        toret = OrderedDict()
        tot = ff.readline()
        if tot:
            tot = int(tot)
            for i in range(tot):

                # read the header, which has format
                # 1. attribute name
                # 2. lines to read
                # 3. name of data type for casting
                # 4.-...: shape
                info = ff.readline().split()
                key, N_lines, dtype = info[0], int(info[1]), info[2]
                shape = tuple(map(int, info[3:]))

                # get the cast function
                if dtype in __builtins__:
                    dtype = __builtins__[dtype]
                elif hasattr(np, dtype):
                    dtype = getattr(np, dtype)
                else:
                    raise TypeError("``attrs`` values must have builtin or numpy type")

                if shape == (0,):
                    toret[key] = dtype(ff.readline())
                else:
                    toret[key] = np.array([dtype(ff.readline()) for i in range(N_lines)]).reshape(shape)

        return toret

    def _get_contracted_index(self, i, j):
        """
        Given the matrix element (i, j) return the index of the corresponding
        element in `_data`
        """
        if i < 0: i += self.N
        if j < 0: j += self.N

        if not 0 <= i < self.N: raise IndexError("index value %d out of range" %i)
        if not 0 <= j < self.N: raise IndexError("index value %d out of range" %j)

        if i > j: i, j = j, i
        return i*(self.N-1) - sum(range(i)) + j

    def _setup(self, data, coords=[], names=[]):
        """
        Setup the class by veryifying the input and storing the
        upper triangle + diagonal elements of the input data (N*(N+1)/2 elements)

        This also initializes the GridIndexer object as the `_indexer` attribute
        """
        self.N = np.shape(data)[0]

        if not len(coords):
            coords = [range(self.N)] * np.ndim(data)

        # coords and names must have same length
        if len(coords) != len(names):
            if len(coords) and not len(names):
                names = ['dim_%d' %i for i in range(len(coords))]
            else:
                raise ValueError("size mismatch between supplied `coords` and `names`")

        # initialize the grid indexer
        self._indexer = indexing.GridIndexer(names, coords)

        # only store the upper triangle (including diagonals)
        inds = np.triu_indices(self.N)
        self._data = data[inds]

    #---------------------------------------------------------------------------
    # builtin math manipulations
    #---------------------------------------------------------------------------
    def __add__(self, other):
        toret = self.copy()
        if isinstance(other, CovarianceMatrix):
            assert self.shape == other.shape
            other = other._data
        toret._data = toret._data + other
        return toret

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        toret = self.copy()
        if isinstance(other, CovarianceMatrix):
            assert self.shape == other.shape
            other = other._data
        toret._data = toret._data - other
        return toret

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        toret = self.copy()
        if isinstance(other, CovarianceMatrix):
            assert self.shape == other.shape
            other = other._datas
        toret._data = toret._data * other
        return toret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        toret = self.copy()
        if isinstance(other, CovarianceMatrix):
            assert self.shape == other.shape
            other = other._data
        toret._data = toret._data / other
        return toret

    def __truediv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        return self.__div__(other)

    def __pow__(self, other):
        toret = self.copy()
        toret._data = toret._data**other
        return toret

    #---------------------------------------------------------------------------
    # utility functions
    #---------------------------------------------------------------------------
    def __repr__(self):
        name = str(self.__class__).split('.')[-1].split("'")[0]
        return "<{name}: size {N}x{N}>".format(name=name, N=self.N)

    def __str__(self):
        return str(self.values)

    def check_properties(self, data):
        """
        Verify that the input data is symmetric and positive-semidefinite, as
        any covariance matrix must be
        """
        if not np.allclose(data, data.T):
            raise ValueError("Covariance matrix is not symmetric")

        if not np.all(np.linalg.eigvals(data) >= 0.):
            raise ValueError("Covariance matrix is not positive-semidefinite")

    def copy(self):
        """
        Return a deep copy of `self`
        """
        import copy
        return copy.deepcopy(self)

    def to_pickle(self, filename):
        """
        Save the covariance matrix as a pickle file
        """
        import pickle
        pickle.dump(self, open(filename, 'w'))

    @classmethod
    def from_pickle(cls, filename):
        """
        Return a pickled CovarianceMatrix
        """
        import pickle
        return pickle.load(open(filename, 'r'))

    def to_plaintext(self, filename):
        """
        Save the covariance matrix to a plain text file
        """
        with open(filename, 'wb') as ff:

            # first line is the data size
            header = "%d\n" %self.N
            ff.write(header.encode())

            # then the data
            ff.write(("\n".join(map(str, self._data)) + "\n").encode())

            # then the dims and coords
            dims = self._indexer.dims_flat
            coords = np.vstack(self[d] for d in dims).T
            ff.write((" ".join(dims) + "\n").encode())
            np.savetxt(ff, coords)

            # then the attrs
            self.__write_attrs__(ff)

    @classmethod
    def from_plaintext(cls, filename):
        """
        Load the covariance matrix from a plain text file
        """
        with open(filename, 'r') as ff:

            # read the data
            N = int(ff.readline())
            data = np.array([float(ff.readline()) for i in range(N*(N+1)//2)])

            # and the dimension/coordinates
            dims = ff.readline().split()
            dims = np.squeeze(np.hsplit(np.array(dims), 2))
            if dims.ndim > 1:
                dims = [list(d) for d in dims]
            else:
                dims = list(dims)
            coords = np.array([[float(x) for x in ff.readline().split()] for i in range(N)])
            coords = np.squeeze(np.hsplit(coords, 2))
            coords = [y.T.tolist() for y in coords]

            # and the meta data
            attrs = cls.__read_attrs__(ff)

        data = condensed_to_full(data, N)
        covar = cls.__construct_direct__(data, coords=coords, names=dims, attrs=attrs)
        return covar

    #---------------------------------------------------------------------------
    # indexing functions
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        """
        Access individual elements through array-like interface:

        >>> C = CovarianceMatrix(data)
        >>> element = C[i, j]
        >>> first_col = C[:, 0]

        where `i` and `j` integers between `0` and `N`
        """
        # return the coordinates arrays if provided the name
        if isinstance(key, string_types):
            if key in self._indexer.dims_flat:
                for axis, dims in enumerate(self.dims):
                    if key in dims:
                        return self.coords[axis][key]
            else:
                raise KeyError("are you trying to access one of these dimensions %s?" %self.dims)

        # if only one key provided, duplicate it
        if not isinstance(key, tuple):
            key = (key, key)

        key = list(key)
        if len(key) == 2 and all(isinstance(k, (int, slice, list, np.ndarray)) for k in key):
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    key[i] = range(*k.indices(self.N))
                elif isinstance(k, (list, np.ndarray)):
                    if isinstance(k, list): k = np.array(k)
                    if k.dtype == bool:
                        if len(k) != self.N:
                            raise ValueError("to index with a boolean array, total length of array must be equal to `N`")
                        k = list(k.nonzero()[0])
                    key[i] = k
                else:
                    key[i] = [k]
            shapes = tuple(len(k) for k in key)
            toret = []
            for i, j in itertools.product(*key):
                toret.append(self._data[self._get_contracted_index(i,j)])
            return np.squeeze(np.array(toret).reshape(shapes))
        else:
            raise KeyError("exactly two integer keys must be supplied")

    def __slice_attrs__(self, indices):
        """
        Slice the ``attrs``, returning a new `OrderedDict` instance
        """
        toret = OrderedDict()
        for k, v in self.attrs.items():
            if is_array_like(v, self.shape):
                for axis, idx in enumerate(indices):
                    v = np.take(v, idx, axis=axis)
                toret[k] = v
            else:
                toret[k] = v
        return toret

    def __slice_coords__(self, indices):
        """
        Slice the coordinates, returning a new `OrderedDict` instance
        """
        coords = []
        for axis, idx in enumerate(indices):
            icoords = []
            for dim in self.dims[axis]:
                c = self.coords[axis][dim]
                if idx is None:
                    icoords.append(c)
                else:
                    icoords.append(np.array(np.take(c, idx, axis=0), ndmin=1))
            coords.append(icoords)
        return coords

    def nearest(self, dim, coord):
        """
        Return the nearest match along the specified dimension for the
        input coordinate value

        Parameters
        ----------
        dim : str
            string specifying the dimension name
        coord : float
            value to find the nearest match to

        Returns
        -------
        idx : int
            the integer index specifying the nearest element
        nearest : float
            the nearest element value
        """
        return self._indexer.nearest(dim, coord)

    def sel(self, **kwargs):
        """
        Label-based indexing by coordinate name

        Notes
        -----
        Returns a new class instance if the resulting sliced data
        is symmetric, otherwise, returns the sliced data array

        Parameters
        ----------
        kwargs : dict
            key, value pairs where the keys represent dimension names and
            the values represent indexing objects, i.e., an integer, slice object,
            or array
        """
        data, indices = self._indexer.sel(self.values, return_indices=True, **kwargs)
        if not is_square(data):
            return data
        else:
            return self.__finalize__(data, indices)

    def isel(self, **kwargs):
        """
        Interger-based indexing by coordinate name

        Notes
        -----
        Returns a new class instance if the resulting sliced data
        is symmetric, otherwise, returns the sliced data array

        Parameters
        ----------
        kwargs : dict
            key, value pairs where the keys represent dimension names and
            the values represent indexing objects, i.e., an integer, slice object,
            or array
        """
        data, indices = self._indexer.isel(self.values, return_indices=True, **kwargs)
        if not is_square(data):
            return data
        else:
            return self.__finalize__(data, indices)

    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property()
    def index(self):
        """
        A list of GridIndex objects for each axis
        """
        return self._indexer.index

    @cached_property()
    def dims(self):
        """
        The dimension names
        """
        return self._indexer.dims

    @cached_property()
    def coords(self):
        """
        A list of dictionaries holding (dim, coord) pairs for each axis
        """
        return self._indexer.coords

    @cached_property()
    def shape(self):
        return (self.N, self.N)

    @cached_property('_data')
    def diag(self):
        """
        The diagonal elements of the matrix
        """
        inds = [self._get_contracted_index(i, i) for i in range(self.N)]
        return self._data[inds]

    @cached_property('_data', 'inverse_rescaling')
    def inverse(self):
        """
        The inverse of the covariance matrix, returned as a 2D ndarray
        """
        return np.linalg.inv(self.values) * self.inverse_rescaling

    @cached_property('_data')
    def values(self):
        """
        Return the total 2D covariance matrix array
        """
        return condensed_to_full(self._data, self.N)

    @cached_property('_data')
    def normalized(self):
        """
        Return the normalized covariance matrix, i.e., the correlation matrix
        """
        ii, jj = np.triu_indices(self.N)
        toret = self.copy()

        # normalize by the diagonals
        diag = self.diag
        norm = np.array([np.sqrt(diag[i]*diag[j]) for (i, j) in zip(ii, jj)])
        toret._data = toret._data/norm
        return toret

    def plot(self, vmin=-1.0, vmax=1.0, include_diagonal=True):
        """
        Plot the correlation matrix (normalized covariance matrix), optionally
        saving if `filename != None`
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import seaborn as sns
        sns.set(style="white")

        # compute the correlation matrix
        corr = self.normalized.values

        # generate a mask for the upper triangle, optionally
        # masking the diagonal
        mask = np.zeros_like(corr, dtype=np.bool)
        k = 0 if not include_diagonal else 1
        mask[np.triu_indices_from(mask, k=k)] = True

        # generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # draw the heatmap with the mask and no labels
        ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=vmax, vmin=vmin, square=True,
                    xticklabels=False, yticklabels=False)

        return ax


#--------------------------------------------------------------------------
# covariance matrix for P(k,mu) measurements
#--------------------------------------------------------------------------
class PkmuCovarianceMatrix(CovarianceMatrix):
    """
    Class to hold a covariance matrix for P(k, mu)

    Parameters
    ----------
    data : array_like (N, ) or (N, N)
        The data representing the covariance matrix. If 1D, the input
        is interpreted as the diagonal elements, with all other elements
        zero
    k_center : array_like
        The wavenumbers where the center of the measurement k-bins are defined,
        for each element of the data matrix along one axis
    mu_center : array_like,
        The values where the center of the measurement mu-bins are defined,
        for each element of the data matrix along one axis
    """
    def __init__(self, data, k_center, mu_center, **kwargs):
        """
        Parameters
        ----------
        data : array_like (N, ) or (N, N)
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        k_center : array_like, (N, )
            The wavenumbers where the center of the measurement k-bins are defined,
            for each element of the data matrix along one axis
        mu_center : array_like, (N, )
            The values where the center of the measurement mu-bins are defined,
            for each element of the data matrix along one axis
        """
        # check the input
        N = np.shape(data)[0]

        # assume statistics are concatenated
        Nk, Nmu = len(k_center), len(mu_center)
        if Nk != N and Nmu != N and Nk*Nmu==N:
            k_center = np.concatenate([k_cen]*Nmu)
            mu_center = np.repeat(mu_center, Nk)

        if len(k_center) != N:
            raise ValueError("size mismatch between supplied `k` index array and data")
        if len(mu_center) != N:
            raise ValueError("size mismatch between supplied `mu` index array and data")

        # setup coords and dims
        names = [['mu1', 'k1'], ['mu2', 'k2']]
        coords = [[mu_center, k_center], [mu_center, k_center]]

        # initialize the base class
        super(PkmuCovarianceMatrix, self).__init__(data, names=names, coords=coords, **kwargs)

    @classmethod
    def periodic_gaussian_covariance(cls, model, k, mu_edges, nbar, volume, Nmu=100):
        r"""
        Return the Gaussian prediction for the covariance between
        :math:`P(k,\mu)` wedges for a periodic box simulation, where the number
        density is constant.

        See eq. 17 of Grieb et al. 2015 arxiv:1509.04293

        Parameters
        ----------
        model : GalaxySpectrum, QuasarSpectrum
            the model instance used to evaluate the theoretical :math:`P(k,\mu)`
        k : array_like
            the array of wavenumbers (units of :math:`h/\mathrm{Mpc}`) where
            the covariance matrix will be evaluated
        mu_edges : array_like
            the edges of the :math:`\mu` bins
        nbar : float
            the constant number density in the box in units of
            :math:`(\mathrm{Mpc}/h)^{-3}` -- the shot noise contribution
            to the covariance is the inverse of this value
        volume : float
            the volume of the box, in units of :math:`(\mathrm{Mpc}/h)^3`
        Nmu : int, optional
            the number of mu bins to use when performing the multipole integration
            over :math:`\mu`

        Returns
        -------
        PkmuCovarianceMatrix :
            the covariance matrix object holding the Gaussian prediction for
            the covariance between the specified wedges

        Examples
        --------
        >>> import numpy
        >>> from pyRSD.rsd import GalaxySpectrum
        >>> from pyRSD.rsdfit.data import PkmuCovarianceMatrix
        >>> volume = 1380.0**3
        >>> nbar = 3e-4
        >>> model = GalaxySpectrum(params='boss_dr12_fidcosmo.ini')
        >>> k = numpy.arange(0., 0.4, 0.005) + 0.005/2
        >>> mu_edges = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> C = PkmuCovarianceMatrix.periodic_gaussian_covariance(model, k, ells, nbar, volume)
        """
        Nwedge = len(mu_edges) - 1
        ndims = (len(k)+2, Nwedge+2)
        mu_edges = np.asarray(mu_edges)

        # the best-fit P(k,mu)
        fine_mu_edges = np.linspace(0, 1, Nmu+1)
        mu_cen = 0.5*(fine_mu_edges[1:] + fine_mu_edges[:-1])
        Pkmu = model.power(k, mu_cen)
        _, mus = np.meshgrid(k, mu_cen, indexing='ij')

        # determine the mu indices for binning
        k_idx = np.arange(len(k), dtype=int)[:,None]
        dig_k = np.repeat(k_idx, Nmu, axis=1) + 1
        dig_mu = np.digitize(mus, mu_edges)
        mu_indices = np.ravel_multi_index([dig_k, dig_mu], ndims)

        # determine the number of modes
        dk = np.diff(k).mean()
        Vk  = 4*np.pi*k**2*dk
        N = Vk * volume / (2*np.pi)**3
        N = np.repeat(N[:,None], Nmu, axis=1) * np.diff(fine_mu_edges)

        Psq = (Pkmu + 1./nbar)**2

        def bin(d, average=False):
            toret = np.zeros(ndims)
            minlength = np.prod(ndims)
            toret.flat = np.bincount(mu_indices.flat, weights=d.flat, minlength=minlength)
            norm = 1.0
            if average:
                norm = np.zeros(ndims)
                norm.flat = np.bincount(mu_indices.flat, minlength=minlength)
                norm = np.squeeze(norm.reshape(ndims)[1:-1, 1:-1])
            toret = np.squeeze(toret.reshape(ndims)[1:-1, 1:-1])
            return toret/norm

        Psq = bin(Psq, average=True)
        modes = bin(N, average=False)

        N1 = len(k); N2 = Nwedge
        C = numpy.zeros((N1,N2)*2)
        for i in range(Nwedge):
            C[:,i,:,i] = np.diag(np.nan_to_num(2*Psq[:,i]/modes[:,i]))

        C = C.reshape((N1*N2,)*2, order='F')

        # the coordinate arrays
        k_coord = np.concatenate([k for i in range(Nwedge)])
        mu_cen = (mu_edges[1:] + mu_edges[:-1])*0.5
        mu_coord = np.concatenate([np.ones(len(k), dtype=int)*mu for mu in mu_cen])

        return cls(C, k_coord, mu_coord, verify=False)

    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def __iter__(self):
        """
        Iterate across mu blocks
        """
        mu1 = self.mus(unique=True, name='mu1')
        mu2 = self.mus(unique=True, name='mu2')
        for i in range(len(mu1)):
            for j in range(len(mu2)):
                yield self.sel(mu1=mu1[i], mu2=mu2[j])

    def enumerate(self, upper=True):
        """
        Enumerate across the upper mu triangle of blocks
        """
        mu1 = self.mus(unique=True, name='mu1')
        mu2 = self.mus(unique=True, name='mu2')

        for iblock, block in enumerate(self):
            i,j = np.unravel_index(iblock, (len(mu1), len(mu2)))
            if upper and i > j: continue
            yield mu1[i], mu2[j], block

    def trim_k(self, kmin=-np.inf, kmax=np.inf):
        """
        Trim the k bounds of the covariance matrix, returning a new, trimmed
        PkmuCovarianceMatrix object

        Parameters
        ----------
        kmin : float or array_like
            the minimum k value; if an array is provided, it is interpreted as the values
            for each mu bin
        kmax : float or array_like
            the maximum k value; if an array is provided, it is interpreted as the values
            for each mu bin
        """
        # get the unique mu values
        mus = self.mus()
        Nmu = len(mus)

        # format the kmin and kmax
        kmin_ = np.empty(Nmu)
        kmax_ = np.empty(Nmu)
        kmin_[:] = kmin
        kmax_[:] = kmax

        total = []
        for i, mu in enumerate(mus):
            k_slice = slice(kmin_[i], kmax_[i])
            _, indices = self._indexer.sel(self.values, return_indices=True, mu1=mu, mu2=mu, k1=k_slice, k2=k_slice)
            total.append(indices)

        total = np.concatenate(total, axis=1).tolist()
        data = self.values
        for axis, idx in enumerate(total):
            data = np.take(data, idx, axis=axis)
        attrs = self.__slice_attrs__(total)
        coords = self.__slice_coords__(total)

        return self.__class__.__construct_direct__(data, coords=coords, names=self.dims, attrs=attrs)

    def mus(self, unique=True, name='mu1'):
        """
        The unique center values of the `mu` bins, optionally
        returning all (non-unique) values
        """
        if not unique:
            return self[name]
        else:
            for index in self.index:
                if name in index:
                    return index.to_pandas(key=name, unique=True)

    def ks(self, unique=True, name='k1'):
        """
        The unique center values of the `k` bins, optionally
        returning all (non-unique) values
        """
        if not unique:
            return self[name]
        else:
            for index in self.index:
                if name in index:
                    return index.to_pandas(key=name, unique=True)

    #---------------------------------------------------------------------------
    # plotting
    #---------------------------------------------------------------------------
    def plot_diagonals(self, ax=None, mu=None, **options):
        """
        Plot the square-root of the diagonal elements for a specfic mu value

        Parameters
        ---------------
        ax : Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        mu : {float, int},
            If not `None`, only plot the diagonals for a single `mu` value, else
            plot for all `mu` values
        options : dict
            Dictionary of options with the following keywords:
                show_gaussian : bool, (`False`)
                    Overplot the Gaussian prediction, i.e.,
                    :math: 2/N_{modes} * [P(k) + Pshot(k)]**2
                norm_by_gaussian : bool, (`False`)
                    Normalize the diagonal elements by the Gaussian prediction
                norm_by_power : bool, (`False`)
                    Normalize the diagonal elements by the mean power,
                    P(k) + Pshot(k)
                subtract_gaussian : bool, (`False`)
                    Subtract out the Gaussian prediction
        """
        from matplotlib import pyplot as plt
        # get the current axes
        if ax is None: ax = plt.gca()

        # setup the default options
        options.setdefault('show_gaussian', False)
        options.setdefault('norm_by_gaussian', False)
        options.setdefault('norm_by_power', False)
        options.setdefault('subtract_gaussian', False)

        # get the mus to plot
        if mu is None:
            mus = self.mus()
        else:
            if isinstance(mu, int):
                mu = self.mus()[mu]
            mus = [mu]

        if options['show_gaussian']:
            if ax.color_cycle != "Paired": ax.color_cycle = "Paired"

        # loop over each mu
        for mu in mus:

            # get sigma for this mu sub matrix
            this_mu = self.sel(mu1=mu, mu2=mu)
            sigma = this_mu.diag**0.5

            norm = 1.
            if options['subtract_gaussian']:
                norm = sigma.copy()
                sigma -= np.diag(this_mu.gaussian_covariance())**0.5
            elif options['norm_by_gaussian']:
                norm = np.diag(this_mu.gaussian_covariance())**0.5
            elif options['norm_by_power']:
                norm = np.diag(this_mu.attrs['mean_power'])
                if not np.count_nonzero(norm):
                    raise ValueError("cannot normalize by power -- all elements are zero")

            # do the plotting
            label = r"$\mu = {}$".format(mu)
            plt.plot(this_mu.ks(), sigma/norm, label=label)
            if options['show_gaussian']:
                plt.plot(this_mu.ks(), np.diag(this_mu.gaussian_covariance())**0.5/norm, ls='--')

            if np.isscalar(norm):
                ax.set_xscale('log')
                ax.set_yscale('log')

        # add the legend and axis labels
        ax.legend(loc=0, ncol=2)
        ax.set_xlabel(r'$k$ ($h$/Mpc)', fontsize=16)
        if options['norm_by_power']:
            ax.set_ylabel(r"$\sigma_P / P(k, \mu)$", fontsize=16)

        elif options['norm_by_gaussian']:
            ax.set_ylabel(r"$\sigma_P / \sigma_P^\mathrm{Gaussian}$", fontsize=16)
        else:
            if not options['subtract_gaussian']:
                ax.set_ylabel(r"$\sigma_P$ $(\mathrm{Mpc}/h)^3$", fontsize=16)
            else:
                ax.set_ylabel(r"$(\sigma_P - \sigma_P^\mathrm{Gaussian})/\sigma_P$", fontsize=16)

        return ax

    def plot_off_diagonals(self, kbar, ax=None, mu=None, mu_bar=None, **options):
        r"""
        Plot the off diagonal elements, specifically:

        :math: \sqrt(Cov(k, kbar) / P(k, mu) / P(kbar, mu))

        Parameters
        ---------------
        kbar : {float, int}
            The specific value of k to plot the covariances at
        ax : Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        mu : {float, int},
            the first mu value
        mu_bar : {float, int},
            the second mu value
        options : dict
            Dictionary of options with the following keywords:
                show_diagonal : bool, (`True`)
                    If `True`, plot the diagonal value as a stem plot
                show_binned : bool, (`False`)
                    If `True`, plot a smoothed spline interpolation
                show_zero_line : bool, (`False`)
                    If `True`, plot a y=0 horizontal line
        """
        from matplotlib import pyplot as plt
        import scipy.stats

        # get the current axes
        if ax is None: ax = plt.gca()

        # setup the default options
        options.setdefault('show_diagonal', True)
        options.setdefault('show_binned', False)
        options.setdefault('show_zero_line', False)

        # determine the mus to plot
        if mu is None:
            mus = self.mus()
            mu_bars = mus
        else:
            if isinstance(mu, int):
                mu = self.mus()[mu]
            mus = [mu]

            if mu_bar is not None:
                if isinstance(mu_bar, int):
                    mu_bar = self.mus()[mu_bar]
                mu_bars = [mu_bar]
            else:
                mu_bars = mus

        # loop over each mu
        for mu, mu_bar in zip(mus, mu_bars):

            # slice at mu1 = mu, mu2 = mu_bar
            this_slice = self.sel(mu1=mu, mu2=mu_bar)

            # ks for this mu
            ks = this_slice.ks(name='k1')
            kbars = this_slice.ks(name='k2')

            # get the right kbar
            if isinstance(kbar, int): kbar = kbars[kbar]

            # get the closest value in the matrix
            kbar_index, kbar = this_slice.nearest('k2', kbar)

            # covariance at the kbar slice
            cov = this_slice.values[:,kbar_index]

            # the normalization
            P_k = np.diag(this_slice.attrs['mean_power'])
            P_kbar = P_k[kbar_index]
            norm = P_k*P_kbar

            # remove the diagonal element
            toplot = cov/norm
            diag_element = toplot[kbar_index]
            toplot = np.delete(toplot, kbar_index)
            ks = ks[~np.isclose(ks, kbar)]

            # plot the fractional covariance
            if mu == mu_bar:
                args = (mu, kbar)
                label = r"$\mu = \bar{{\mu}} = {0}, \ \bar{{k}} = {1:.3f}$ $h$/Mpc".format(*args)
            else:
                args = (mu, mu_bar, kbar)
                label = r"$\mu = {0}, \ \bar{{\mu}} = {1}, \ \bar{{k}} = {2:.3f}$ $h$/Mpc".format(*args)
            plt.plot(ks, toplot, label=label)

            # get the color to use
            this_color = ax.last_color

            # plot the diagonal element
            if options['show_diagonal']:
                markerline, stemlines, baseline = ax.stem([kbar], [diag_element], linefmt=this_color)
                plt.setp(markerline, 'markerfacecolor', this_color)
                plt.setp(markerline, 'markeredgecolor', this_color)

            # plot a smoothed spline
            if options['show_binned']:
                nbins = tools.histogram_bins(toplot)
                y, binedges, w = scipy.stats.binned_statistic(ks, toplot, bins=nbins)
                plt.plot(0.5*(binedges[1:]+binedges[:-1]), y, color=this_color)

        if options['show_zero_line']:
            ax.axhline(y=0, c='k', ls='--', alpha=0.6)

        # add the legend and axis labels
        ax.legend(loc=0, ncol=2)
        ax.set_xlabel(r'$k$ ($h$/Mpc)', fontsize=16)
        ax.set_ylabel(r"$\mathrm{Cov}(k, \bar{k}) / (P(k, \mu) P(\bar{k}, \bar{\mu}))$", fontsize=16)

        return ax

#--------------------------------------------------------------------------
# covariance matrix for multipole measurements
#--------------------------------------------------------------------------
class PoleCovarianceMatrix(CovarianceMatrix):
    """
    Class to hold a covariance matrix for multipole measurements

    Parameters
    ----------
    data : array_like (N, ) or (N, N)
        The data representing the covariance matrix. If 1D, the input
        is interpreted as the diagonal elements, with all other elements
        zero
    k_center : array_like, (N, )
        The wavenumbers where the center of the measurement k-bins are defined,
        for each element of the data matrix along one axis
    ells : array_like, (N, )
        The multipole numbers for each element of the data matrix along one axis

    Examples
    --------

    >>> from pyRSD.rsdfit.data import PoleCovarianceMatrix
    >>> import numpy as np

    >>> # generate 20 k bins from 0.01 to 0.4
    >>> Nk = 20
    >>> k_edges = np.linspace(0.01, 0.4, Nk+1)
    >>> k_cen = 0.5 * (k_edges[1:] + k_edges[:-1])

    >>> # generate 100 fake monopoles
    >>> P0 = np.random.random(size=(100, Nk)) # Nk = 20, see above

    >>> # 100 fake quadrupoles
    >>> P2 = np.random.random(size=(100, Nk))

    >>> # 100 fake hexadecapoles
    >>> P4 = np.random.random(size=(100, Nk))

    >>> # make the full data vector
    >>> D = np.concatenate([P0, P2, P4], axis=-1) # shape is (100, 60)

    >>> # compute the covariance matrix
    >>> cov = np.cov(D, rowvar=False) # shape is (20,20)

    >>> # initialize the PoleCovarianceMatrix
    >>> ells = [0,2,4]
    >>> C = PoleCovarianceMatrix(cov, k_cen, ells)
    """
    def __init__(self, data, k_center, ells, **kwargs):
        """
        Parameters
        ----------
        data : array_like (N, ) or (N, N)
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        k_center : array_like, (N, )
            The wavenumbers where the center of the measurement k-bins are defined,
            for each element of the data matrix along one axis
        ells : array_like, (N, )
            The multipole numbers for each element of the data matrix along one axis
        """
        # check the input
        N = np.shape(data)[0]

        # assume statistics are concatenated
        Nk, Nell = len(k_center), len(ells)
        if Nk != N and Nell != N and Nk*Nell==N:
            k_center = np.concatenate([k_center]*Nell)
            ells = np.repeat(ells, Nk)

        if len(k_center) != N:
            raise ValueError("size mismatch between supplied `k` index array and data")
        if len(ells) != N:
            raise ValueError("size mismatch between supplied `ell` index array and data")

        # setup coords and dims
        names = [['ell1', 'k1'], ['ell2', 'k2']]
        coords = [[ells, k_center], [ells, k_center]]

        # initialize the base class
        super(PoleCovarianceMatrix, self).__init__(data, names=names, coords=coords, **kwargs)

    @classmethod
    def periodic_gaussian_covariance(cls, model, k, ells, nbar, volume, Nmu=100):
        r"""
        Return the Gaussian prediction for the covariance between multipoles
        for a periodic box simulation, where the number density is constant.

        See eq. 16 of Grieb et al. 2015 arxiv:1509.04293

        Parameters
        ----------
        model : GalaxySpectrum, QuasarSpectrum
            the model instance used to evaluate the theoretical :math:`P(k,\mu)`
        k : array_like
            the array of wavenumbers (units of :math:`h/\mathrm{Mpc}`) where
            the covariance matrix will be evaluated
        ells : list of int
            the list of multipole numbers to compute the covariance of
        nbar : float
            the constant number density in the box in units of
            :math:`(\mathrm{Mpc}/h)^{-3}` -- the shot noise contribution
            to the covariance is the inverse of this value
        volume : float
            the volume of the box, in units of :math:`(\mathrm{Mpc}/h)^3`
        Nmu : int, optional
            the number of mu bins to use when performing the multipole integration
            over :math:`\mu`

        Returns
        -------
        PoleCovarianceMatrix :
            the covariance matrix object holding the Gaussian prediction for
            the covariance between the specified multipoles

        Examples
        --------
        >>> import numpy
        >>> from pyRSD.rsd import GalaxySpectrum
        >>> from pyRSD.rsdfit.data import PoleCovarianceMatrix
        >>> volume = 1380.0**3
        >>> nbar = 3e-4
        >>> model = GalaxySpectrum(params='boss_dr12_fidcosmo.ini')
        >>> k = numpy.arange(0., 0.4, 0.005) + 0.005/2
        >>> ells = [0,2,4]
        >>> C = PoleCovarianceMatrix.periodic_gaussian_covariance(model, k, ells, nbar, volume)
        """
        from scipy.special import legendre

        if np.isscalar(ells):
            ells = [ells]

        # the best-fit P(k,mu)
        mus = np.linspace(0, 1, Nmu+1)
        Pkmu = model.power(k, mus)
        _, mus = np.meshgrid(k, mus, indexing='ij')

        N1 = Pkmu.shape[0]
        N2 = len(ells)

        # determine the number of modes
        dk = np.diff(k).mean()
        Vk  = 4*np.pi*k**2*dk
        N = Vk * volume / (2*np.pi)**3

        # initialize the return array
        Psq = np.zeros((N2, N1)*2)
        modes = np.zeros((N2, N1)*2)

        # leg weights
        leg = np.array([(2*ell+1)*legendre(ell)(mus) for ell in ells])

        # P(k,mu)^2 * L_ell(mu)*L_ellprime
        weights = leg[:,None]*leg[None,:]
        power = (Pkmu + 1./nbar)**2
        tobin = weights * power[None,...]

        # fill the covariance for each ell, ell_prime
        for i in range(N2):
            for j in range(i, N2):

                Psq[i,:,j,:] = np.diag(np.nanmean(tobin[i,j,:], axis=-1))
                modes[i,:,j,:] = N
                if i != j:
                    Psq[j,:,i,:] = Psq[i,:,j,:]
                    modes[j,:,i,:] = modes[i,:,j,:]

        # reshape squared power and modes
        Psq = Psq.reshape((N1*N2,)*2)
        modes = modes.reshape((N1*N2,)*2)
        C = np.nan_to_num(2*Psq/modes)

        # the coordinate arrays
        k_coord = np.concatenate([k for i in range(len(ells))])
        ell_coord = np.concatenate([np.ones(len(k), dtype=int)*ell for ell in ells])

        return cls(C, k_coord, ell_coord, verify=False)


    @classmethod
    def cutsky_gaussian_covariance(cls, model, k, ells, nbar, fsky, zmin, zmax,
                                    FKP_P0=1e4, Nmu=100, Nz=50):
        r"""
        Return the Gaussian prediction for the covariance between multipoles
        for a "cutsky" survey, i.e., a survey with a varying :math:`n(z)`
        distribution

        Parameters
        ----------
        model : GalaxySpectrum, QuasarSpectrum
            the model instance used to evaluate the theoretical :math:`P(k,\mu)`
        k : array_like
            the array of wavenumbers (units of :math:`h/\mathrm{Mpc}`) where
            the covariance matrix will be evaluated
        ells : list of int
            the list of multipole numbers to compute the covariance of
        nbar : callable
            a callable function returning the number density as a function of
            redshift
        fsky : float
            the sky fraction, used to normalized the effective volume calculation
        zmin : float
            the minimum redshift value to include when performing the effective
            volume calculation
        zmax : float
            the maximum redshift value to include when performing the effective
            volume calculation
        FKP_P0 : float, optional
            the FKP P0 value to use, where the FKP weights are :math:`1/(1+n(z)P_0)`;
            default is 1e4
        Nmu : int, optional
            the number of mu bins to use when performing the multipole integration
            over :math:`\mu`
        Nz : int, optional
            the number of redshift bins to use when performing the integral over
            redshift

        Returns
        -------
        PoleCovarianceMatrix :
            the covariance matrix object holding the Gaussian prediction for
            the covariance between the specified multipoles

        Examples
        --------
        >>> import numpy
        >>> from scipy.interpolate import InterpolatedUnivariateSpline as spline
        >>> from pyRSD.rsd import GalaxySpectrum
        >>> from pyRSD.rsdfit.data import PoleCovarianceMatrix
        >>> filename = 'nbar_DR12v5_CMASSLOWZ_North_om0p31_Pfkp10000.dat'
        >>> nbar = numpy.loadtxt(filename, skiprows=3)
        >>> nbar = spline(nbar[:,0], nbar[:,3])
        >>> fsky = 0.1436
        >>> model = GalaxySpectrum(params='boss_dr12_fidcosmo.ini')
        >>> k = numpy.arange(0., 0.4, 0.005) + 0.005/2
        >>> ells = [0,2,4]
        >>> zmin = 0.2
        >>> zmax = 0.5
        >>> C = PoleCovarianceMatrix.cutsky_gaussian_covariance(model, k, ells, nbar, fsky, zmin, zmax)
        """
        from scipy.special import legendre

        if not callable(nbar):
            raise ValueError("``nbar`` must be a callable function returning n(z)")

        if np.isscalar(ells):
            ells = [ells]

        # the model cosmology
        cosmo = model.cosmo

        # the best-fit P(k,mu)
        mus = np.linspace(0, 1, Nmu+1)
        Pkmu = model.power(k, mus)
        if isinstance(Pkmu, xr.DataArray):
            Pkmu = Pkmu.values
        _, mus = np.meshgrid(k, mus, indexing='ij')

        N1 = Pkmu.shape[0]
        N2 = len(ells)

        # volume of redshift shells for integral over z
        zbins = np.linspace(zmin, zmax, Nz+1)
        R_hi = cosmo.Dc_z(zbins[1:]) * cosmo.h() # in Mpc/h
        R_lo = cosmo.Dc_z(zbins[:-1]) * cosmo.h() # in Mpc/h
        dV = (4./3.)*np.pi*(R_hi**3 - R_lo**3)

        # compute nbar
        zcen = 0.5*(zbins[:-1] + zbins[1:])
        nbar_ = nbar(zcen)

        # weights
        w = 1. / (1 + nbar_*FKP_P0) # FKP weights

        # properly calibrate fsky
        dV *= fsky

        # effective volume
        W2 = ((nbar_*w)**2 * dV).sum()
        W4 = ((nbar_*w)**4 * dV).sum()
        Veff = W2**2 / W4

        # k-shell volume
        dk = np.diff(k).mean()
        Vk  = 4*np.pi*k**2*dk
        N = Vk * Veff / (2*np.pi)**3

        # initialize the return array
        Psq = np.zeros((N2, N1)*2)
        modes = np.zeros((N2, N1)*2)

        # leg weights
        leg = np.array([(2*ell+1)*legendre(ell)(mus) for ell in ells])

        # P(k,mu)^2 * L_ell(mu)*L_ellprime
        weights = leg[:,None]*leg[None,:]
        power = (Pkmu[...,None] + 1./nbar_)**2
        tobin = weights[...,None] * power[None,...]

        # fill the covariance for each ell, ell_prime
        for i in range(N2):
            for j in range(i, N2):

                # do the sum over redshift first
                x = ( (w*nbar_)**4 * dV * tobin).sum(axis=-1) / W4
                Psq[i,:,j,:] = np.diag(np.nanmean(x[i,j,:], axis=-1))
                modes[i,:,j,:] = N
                if i != j:
                    Psq[j,:,i,:] = Psq[i,:,j,:]
                    modes[j,:,i,:] = modes[i,:,j,:]

        # reshape squared power and modes
        Psq = Psq.reshape((N1*N2,)*2)
        modes = modes.reshape((N1*N2,)*2)
        C = np.nan_to_num(2*Psq/modes)

        # the coordinate arrays
        k_coord = np.concatenate([k for i in range(len(ells))])
        ell_coord = np.concatenate([np.ones(len(k), dtype=int)*ell for ell in ells])

        return cls(C, k_coord, ell_coord, verify=False)

    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def __iter__(self):
        """
        Iterate across ``ell`` blocks
        """
        ell1 = self.ells(unique=True, name='ell1')
        ell2 = self.ells(unique=True, name='ell2')
        for i in range(len(ell1)):
            for j in range(len(ell2)):
                yield self.sel(ell1=ell1[i], ell2=ell2[j])

    def enumerate(self, upper=True):
        """
        Enumerate across the upper triangle of blocks
        """
        ell1 = self.ells(unique=True, name='ell1')
        ell2 = self.ells(unique=True, name='ell2')

        for iblock, block in enumerate(self):
            i,j = np.unravel_index(iblock, (len(ell1), len(ell2)))
            if upper and i > j: continue
            yield ell1[i], ell2[j], block

    def trim_k(self, kmin=-np.inf, kmax=np.inf):
        """
        Trim the k bounds of the covariance matrix, returning a new, trimmed
        PkmuCovarianceMatrix object

        Parameters
        ----------
        kmin : float or array_like
            the minimum k value; if an array is provided, it is interpreted as the values
            for each mu bin
        kmax : float or array_like
            the maximum k value; if an array is provided, it is interpreted as the values
            for each mu bin
        """
        # get the unique mu values
        ells = self.ells()
        Nell = len(ells)

        # format the kmin and kmax
        kmin_ = np.empty(Nell)
        kmax_ = np.empty(Nell)
        kmin_[:] = kmin
        kmax_[:] = kmax

        total = []
        for i, ell in enumerate(ells):
            k_slice = slice(kmin_[i], kmax_[i])
            _, indices = self._indexer.sel(self.values, return_indices=True, ell1=ell, ell2=ell, k1=k_slice, k2=k_slice)
            total.append(indices)

        total = np.concatenate(total, axis=1).tolist()
        data = self.values
        for axis, idx in enumerate(total):
            data = np.take(data, idx, axis=axis)
        attrs = self.__slice_attrs__(total)
        coords = self.__slice_coords__(total)

        return self.__class__.__construct_direct__(data, coords=coords, names=self.dims, attrs=attrs)

    def ells(self, unique=True, name='ell1'):
        """
        The unique center values of the `ell` bins, optionally
        returning all (non-unique) values
        """
        if not unique:
            return self[name]
        else:
            for index in self.index:
                if name in index:
                    return index.to_pandas(key=name, unique=True)

    def ks(self, unique=True, name='k1'):
        """
        The unique center values of the `k` bins, optionally
        returning all (non-unique) values
        """
        if not unique:
            return self[name]
        else:
            for index in self.index:
                if name in index:
                    return index.to_pandas(key=name, unique=True)

    #---------------------------------------------------------------------------
    # plotting
    #---------------------------------------------------------------------------
    def plot_diagonals(self, ax=None, ell=None, ell_bar=None, **options):
        """
        Plot the square-root of the diagonal elements for a specfic ell value

        Parameters
        ---------------
        ax : Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        ell : int
            multipole number for the first axis
        ell_bar : int
            multipole number for the second axis
        options : dict
            Dictionary of options with the following keywords:
                show_gaussian : bool, (`False`)
                    Overplot the Gaussian prediction, i.e.,
                    :math: 2/N_{modes} * [P(k) + Pshot(k)]**2
                norm_by_gaussian : bool, (`False`)
                    Normalize the diagonal elements by the Gaussian prediction
                norm_by_power : bool, (`False`)
                    Normalize the diagonal elements by the mean power,
                    P(k) + Pshot(k)
                subtract_gaussian : bool, (`False`)
                    Subtract out the Gaussian prediction
        """
        from matplotlib import pyplot as plt

        # get the current axes
        if ax is None: ax = plt.gca()

        # setup the default options
        options.setdefault('show_gaussian', False)
        options.setdefault('norm_by_gaussian', False)
        options.setdefault('norm_by_power', False)
        options.setdefault('subtract_gaussian', False)
        options.setdefault('label', "")

        # get the ells to plot
        ells = self.ells()
        if ell is None and ell_bar is None:
            ells = [(ells[i], ells[j]) for i in range(len(ells)) for j in range(i, len(ells))]
        elif ell_bar is None:
            ells = [(ell, ells[i]) for i in range(len(ells))]
        else:
            ells = [(ell, ell_bar)]

        if options['show_gaussian']:
            if ax.color_cycle != "Paired": ax.color_cycle = "Paired"

        # loop over each ell pair
        for (ell, ell_bar) in ells:

            # get sigma for this mu sub matrix
            this_slice = self.sel(ell1=ell, ell2=ell_bar)
            sigma = this_slice.diag**0.5

            norm = 1.
            if options['subtract_gaussian']:
                norm = sigma.copy()
                gauss_cov = this_slice.gaussian_covariance()
                sigma -= np.diag(gauss_cov)**0.5
            elif options['norm_by_gaussian']:
                gauss_cov = this_slice.gaussian_covariance()
                norm = np.diag(gauss_cov)**0.5
            elif options['norm_by_power']:
                norm = abs(np.diag(this_slice.attrs['mean_power']))
                if not np.count_nonzero(norm):
                    raise ValueError("cannot normalize by power -- all elements are zero")

            # do the plotting
            label = options['label']
            if not options['label'] and ell == ell_bar:
                label = r"$\ell = \bar{\ell} = %d$" %ell
            elif not options['label']:
                label = r"$\ell = %d, \ \bar{\ell} = %d$" %(ell, ell_bar)
            plt.plot(this_slice.ks(), sigma/norm, label=label)
            if options['show_gaussian']:
                gauss_cov = this_slice.gaussian_covariance()
                plt.plot(this_slice.ks(), np.diag(gauss_cov)**0.5/norm, ls='--')

            if np.isscalar(norm):
                ax.set_xscale('log')
                ax.set_yscale('log')

        # add the legend and axis labels
        ax.legend(loc=0, ncol=2)
        ax.set_xlabel(r'$k$ ($h$/Mpc)', fontsize=16)
        if options['norm_by_power']:
            ax.set_ylabel(r"$\sigma_P / |P_\ell(k)|$", fontsize=16)

        elif options['norm_by_gaussian']:
            ax.set_ylabel(r"$\sigma_P / \sigma_P^\mathrm{Gaussian}$", fontsize=16)
        else:
            if not options['subtract_gaussian']:
                ax.set_ylabel(r"$\sigma_P$ $(\mathrm{Mpc}/h)^3$", fontsize=16)
            else:
                ax.set_ylabel(r"$(\sigma_P - \sigma_P^\mathrm{Gaussian})/\sigma_P$", fontsize=16)

        return ax

    def plot_off_diagonals(self, kbar, ax=None, ell=None, ell_bar=None, **options):
        r"""
        Plot the off diagonal elements, specifically:

        :math: \sqrt(Cov(k, kbar) / P(k, ell) / P(kbar, ell_bar))

        Parameters
        ---------------
        kbar : {float, int}
            The specific value of k to plot the covariances at
        ax : Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        ell : int
            multipole axis for the first axis
        ell_bar : int
            multipole axis for the second axis
        options : dict
            Dictionary of options with the following keywords:
                show_diagonal : bool, (`True`)
                    If `True`, plot the diagonal value as a stem plot
                show_binned : bool, (`False`)
                    If `True`, plot a smoothed spline interpolation
                show_zero_line : bool, (`False`)
                    If `True`, plot a y=0 horizontal line
        """
        from matplotlib import pyplot as plt
        import scipy.stats

        # get the current axes
        if ax is None: ax = plt.gca()

        # setup the default options
        options.setdefault('show_diagonal', True)
        options.setdefault('show_binned', False)
        options.setdefault('show_zero_line', False)
        options.setdefault('label', "")
        options.setdefault('norm_by_power', False)

        # get the ells to plot
        ells = self.ells()
        if ell is None and ell_bar is None:
            ells = [(ells[i], ells[j]) for i in range(len(ells)) for j in range(i, len(ells))]
        elif ell_bar is None:
            ells = [(ell, ells[i]) for i in range(len(ells))]
        else:
            ells = [(ell, ell_bar)]

        # loop over each mu
        for (ell, ell_bar) in ells:

            # slice
            this_slice = self.sel(ell1=ell, ell2=ell_bar)

            # ks for this ell
            ks = this_slice.ks(name='k1')
            kbars = this_slice.ks(name='k2')

            # get the right kbar
            if isinstance(kbar, int): kbar = kbars[kbar]

            # get the closest value in the matrix
            kbar_index, kbar = this_slice.nearest('k2', kbar)

            # covariance at the kbar slice
            cov = this_slice.values[:,kbar_index]

            # the normalization
            norm = 1.
            if options['norm_by_power'] and 'mean_power' in this_slice.attrs:
                norm = np.diag(this_slice.attrs['mean_power'])**2

            # remove the diagonal element
            toplot = cov/norm
            diag_element = toplot[kbar_index]
            toplot = np.delete(toplot, kbar_index)
            ks = ks[~np.isclose(ks, kbar)]

            # plot the fractional covariance
            label = options['label']
            if ell == ell_bar:
                args = (ell, kbar)
                if not options['label']:
                    label = r"$\ell = \bar{{\ell}} = %d, \ \bar{{k}} = %.3f$ $h$/Mpc" %args
            else:
                args = (ell, ell_bar, kbar)
                if not options['label']:
                    label = r"$\ell = %d, \ \bar{{\ell}} = %d, \ \bar{{k}} = %.3f$ $h$/Mpc" %args
            plt.plot(ks, toplot, label=label)

            # get the color to use
            this_color = ax.last_color

            # plot the diagonal element
            if options['show_diagonal']:
                markerline, stemlines, baseline = ax.stem([kbar], [diag_element], linefmt=this_color)
                plt.setp(markerline, 'markerfacecolor', this_color)
                plt.setp(markerline, 'markeredgecolor', this_color)

            # plot a smoothed spline
            if options['show_binned']:
                nbins = tools.histogram_bins(toplot)
                y, binedges, w = scipy.stats.binned_statistic(ks, toplot, bins=nbins)
                plt.plot(0.5*(binedges[1:]+binedges[:-1]), y, color=this_color)

        if options['show_zero_line']:
            ax.axhline(y=0, c='k', ls='--', alpha=0.6)

        # add the legend and axis labels
        ax.legend(loc=0, ncol=2)
        ax.set_xlabel(r'$k$ ($h$/Mpc)', fontsize=16)
        if options['norm_by_power'] and 'mean_power' in this_slice.attrs:
            ax.set_ylabel(r"$\mathrm{Cov}(k, \bar{k}) / (P(k, \ell) P(\bar{k}, \bar{\ell}))$", fontsize=16)
        else:
            ax.set_ylabel(r"$\mathrm{Cov}(k, \bar{k})$  $(\mathrm{Mpc}/h)^3$", fontsize=16)

        return ax
