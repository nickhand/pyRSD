from ... import numpy as np
from ...rsd._cache import Cache, cached_property, parameter
import itertools

class CovarianceMatrix(Cache):
    """
    Class to represent a covariance matrix. The class has coordinates associated
    with the covariance matrix, which can be used to index the matrix
    """
    
    def __init__(self, data, coords=[], names=[], verify=True):
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
        verify : bool, optional
            If `True`, verify that the matrix is positive-semidefinite and
            symmetric
        """
        # initialize the base Cache
        Cache.__init__(self,)
        
        # make local copies
        data = np.asarray(data).copy()
        
        # make a diagonal 2D matrix from input
        if data.ndim == 1: data = np.diag(data)
            
        # check basic properties
        if verify: self.check_properties(data)
        
        # initialize the backend        
        self._setup(data, coords=coords, names=names)
        self.inverse_rescaling = 1.0
        
    @parameter
    def _data(self, val):
        """
        The main data attribute, which stores the upper triangle + diagonal
        elements of the symmetric matrix
        """
        return val
    
    @parameter
    def inverse_rescaling(self, val):
        """
        Rescale the inverse of the covariance matrix by this factor
        """
        return val
        
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
        """
        self.N = np.shape(data)[0]
        
        # coords and names must have same length
        if len(coords) != len(names):
            if len(coords) and not len(names):
                names = ['coord_%d' %i for i in range(len(coords))]
            else:
                raise ValueError("size mismatch between supplied `coords` and `names`")
        if len(coords):
            self.names = names
            self.coords = dict(zip(names, coords))
        else:
            self.names = self.coords = None
        
        # determine which coordinates are unique
        self._unique = None
        if self.coords is not None:            
            self._unique = {}
            for name in self.names:
                if len(self.coords[name]) != self.N:
                    raise ValueError("size mismatch between data and `%s` coordinate for covariance matrix" %name)
                self._unique[name] = len(np.unique(self.coords[name])) == len(self.coords[name])
            
        # only store the upper triangle (including diagonals)
        inds = np.triu_indices(self.N)
        self._data = data[inds]

    def check_properties(self, data):
        """
        Verify that the input data is symmetric and positive-semidefinite, as
        any covariance matrix must be
        """
        if not np.allclose(data, data.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        if not np.all(np.linalg.eigvals(data) >= 0.):
            raise ValueError("Covariance matrix is not positive-semidefinite")
    
    #---------------------------------------------------------------------------
    # builtin math manipulations
    #---------------------------------------------------------------------------
    def __add__(self, other):
        toret = self.copy()
        toret._data += other
        return toret

    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        toret = self.copy()
        toret._data -= other
        return toret

    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        toret = self.copy()
        toret._data *= other
        return toret

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __div__(self, other):
        toret = self.copy()
        toret._data /= other
        return toret

    def __rdiv__(self, other):
        return self.__div__(other)
        
    def __pow__(self, other):
        toret = self.copy()
        toret._data **= other
        return toret
    
    #---------------------------------------------------------------------------
    # utility functions
    #---------------------------------------------------------------------------
    def __repr__(self):
        name = str(self.__class__).split('.')[-1].split("'")[0]
        return "<{name}: size {N}x{N}>".format(name=name, N=self.N)
    
    def __str__(self):
        return str(self.full())

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
        return pickle.load(open(filename, 'r'))
        
    def to_plaintext(self, filename):
        """
        Save the covariance matrix to a plain text file
        """
        with open(filename, 'w') as ff:
            ff.write("%d\n" %self.N)
            ff.write("\n".join(map(str, self._data)) + "\n")
            if self.names is not None:
                ff.write(" ".join(self.coords.keys()) + "\n")
                np.savetxt(ff, zip(*self.coords.values()))
    
    @classmethod
    def from_plaintext(cls, filename):
        """
        Load the covariance matrix from a plain text file
        """
        names, coords = [], []
        with open(filename, 'r') as ff:
            N = int(ff.readline())
            data = np.array([float(ff.readline()) for i in range(N*(N+1)/2)])
            names = ff.readline()
            if names:
                names = names.split()
                coords = np.loadtxt(ff)
        if len(coords):
            coords = [coords[:,i] for i in range(len(names))]
        
        covar = cls(np.empty((N, N)), coords=coords, names=names, verify=False)
        covar._data = data
        return covar
        
    #---------------------------------------------------------------------------
    # indexing functions
    #---------------------------------------------------------------------------
    def nearest(self, **kwargs):
        """
        Return a tuple of the nearest match to the coordinate values
        """
        coords = zip(*[self.coords[k] for k in kwargs])
        coord0 = kwargs.values()        
        def distance(tup, coord):
            toret = 0
            for i, x in enumerate(tup):
                toret += abs(x-coord[i])
            return toret
        min_tuple = min(coords, key=lambda x: distance(x, coord0))
        return coords.index(min_tuple), min_tuple
        
    def sel(self, **kwargs):
        """
        Value indexing by coordinate name
        """
        # verify the keywords are all valid coordinate names
        for k in kwargs:
            if k not in self.names:
                raise IndexError("name `%s` is not a valid coordinate name" %k)
        
        # can't index on one coordinate, if it isn't unique
        if len(kwargs) == 1 and not self._unique[kwargs.keys()[0]]:
            raise IndexError("cannot index only on `%s` coordinate, since it is not unique" %(kwargs.keys()[0]))
        
        # get the coordinates as zipped tuples    
        indices = []
        for k in kwargs:
            if np.isscalar(kwargs[k]): 
                kwargs[k] = [kwargs[k]]
        for tup in zip(*kwargs.values()):
            index, _ = self.nearest(**dict(zip(self.names, tup)))
            indices.append(index)
        
        if len(indices) == 1:
            return self[indices[0], :]
        else:
            return self[indices[0], indices[-1]]
        
    def __getitem__(self, key):
        """
        Access individual elements through array-like interface:

        >>> C = CovarianceMatrix(data)
        >>> element = C[i, j]
        >>> first_col = C[:, 0]

        where `i` and `j` integers between `0` and `N`
        """
        # return the coordinates arrays if provided the name
        if isinstance(key, basestring) and key in self.names:
            return self.coords[key]
            
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

    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property()
    def shape(self):
        return (self.N, self.N)
            
    @cached_property('_data')
    def diag(self):
        """
        The diagonal elements of the matrix
        """
        inds = [self._get_contracted_index(i, i) for i in range(self.N)]
        self._diag = self._data[inds]
        return self._diag

    @cached_property('_data', 'inverse_rescaling')
    def inverse(self):
        """
        The inverse of the covariance matrix, returned as a 2D ndarray
        """
        return np.linalg.inv(self.full()) * self.inverse_rescaling
    
    def full(self):
        """
        Return the full 2D covariance matrix array
        """
        toret = np.empty((self.N, self.N))
        
        # set the upper triangle and the diagonal
        upper_inds = np.triu_indices(self.N)
        toret[upper_inds] = self._data[:]
        
        # and now set the lower triangle
        i, j = np.tril_indices(self.N)
        toret[i, j] = toret[j, i]
        return toret
    
    def normalized(self):
        """
        Return the normalized covariance matrix, i.e., the correlation matrix
        """
        ii, jj = np.triu_indices(self.N)
        toret = self.copy()
        
        # normalize by the diagonals
        diag = self.diag
        norm = np.array([np.sqrt(diag[i]*diag[j]) for (i, j) in zip(ii, jj)])
        toret._data /= norm
        return toret
        
    def plot(self, filename=None):
        """
        Plot the correlation matrix (normalized covariance matrix), optionally
        saving if `filename != None`
        """
        import plotify as pfy
        pfy.clf()
        corr = self.normalized().full()
        colormesh = pfy.pcolormesh(corr, vmin=-1, vmax=1)
        pfy.colorbar(colormesh)
        if filename is not None:
            pfy.savefig(filename)
    
        return colormesh

#
#     #---------------------------------------------------------------------------
#     # k, x handling
#     #---------------------------------------------------------------------------
#     @property
#     def _k_level(self):
#      """
#      Internal variable defining which level the `k` column is defined in
#      the `MultiIndex`
#      """
#      try:
#          return self.__k_level
#      except AttributeError:
#          if 'k' not in self.index.names:
#              raise ValueError("Index for `PowerCovarianceMatrix` does not contain `k`")
#          self.__k_level = np.where(np.asarray(self.index.names) == 'k')[0][0]
#          return self.__k_level
#
#     #---------------------------------------------------------------------------
#     @property
#     def _x_level(self):
#         """
#         Internal variable defining which level the `self._outer_index_name`
#         column is defined in the `MultiIndex`
#         """
#         try:
#             return self.__x_level
#         except AttributeError:
#             x = self._outer_index_name
#             if x not in self.index.names:
#                 raise ValueError("Index for `PowerCovarianceMatrix` does not contain '%s'" %x)
#             self.__x_level = np.where(np.asarray(self.index.names) == x)[0][0]
#             return self.__x_level
#
#     #---------------------------------------------------------------------------
#     def _convert_x_integer(self, x_int):
#         """
#         Convert the integer value specifying a given outer index value
#         to the actual value
#         """
#         vals = getattr(self, self._outer_index_name + "s")
#         if x_int in vals:
#             return x_int
#
#         msg = "[0, %d) or [-%d, -1]" %(len(vals), len(vals))
#         if x_int < 0: x_int += len(vals)
#         if not 0 <= x_int < len(vals):
#             raise KeyError("Integer that identifies %s-band must be between %s"
#                             %(self._outer_index_name, msg))
#         return vals[x_int]
#
#     #---------------------------------------------------------------------------
#     def _convert_k_integer(self, k_int, x):
#         """
#         Convert the integer value specifying a given `k` (associated with
#         a specific `x`) to the actual value
#         """
#         # get the ks for this mu
#         ks = self.ks(**{self._outer_index_name:x})
#
#         msg = "[0, %d) or [-%d, -1]" %(len(ks), len(ks))
#         if k_int < 0: k_int += len(ks)
#         if not 0 <= k_int < len(ks):
#             raise KeyError("Integer that identifies k-band for %s = %s must be between %s"
#                             %(self._outer_index_name, x, msg))
#         return ks[k_int]
#
#     #---------------------------------------------------------------------------
#     @property
#     def _xs(self):
#         """
#         The `x` values associated with the matrix; only unique `x` values
#         are returned
#         """
#         return self.index.get_level_values(self._outer_index_name).unique()
#
#     #---------------------------------------------------------------------------
#
#     #---------------------------------------------------------------------------
#     # slicing functions
#     #---------------------------------------------------------------------------
#     def _x_slice(self, x1, x2=None):
#         """
#         Return the sub matrix associated with the specified `x1` and `x2` values
#
#         Note: this returns a `DataFrame`
#         """
#         if isinstance(x1, int):
#             x1 = self._convert_x_integer(x1)
#         if x2 is not None:
#             if isinstance(x2, int):
#                 x2 = self._convert_x_integer(x2)
#         else:
#             x2 = x1
#
#         # slice both the rows and columns
#         full = self.full()
#         slice1 = full.xs(x1, level=self._outer_index_name, axis=0)
#         slice2 = slice1.xs(x2, level=self._outer_index_name, axis=1)
#         return slice2
#
#     #---------------------------------------------------------------------------
#     def slice(self, k, x):
#         """
#         Return a slice of the covariance matrix at the specified (k, mu)
#         or (k, ell) values
#         """
#         # get the correct k, mu from integer input
#         if isinstance(x, int):
#             x = self._convert_x_integer(x)
#         if isinstance(k, int):
#             k = self._convert_k_integer(k, x)
#
#         return self.full()[x, k]
#
#     #---------------------------------------------------------------------------
#     # trimming functions
#     #---------------------------------------------------------------------------
#     def _rebin(self, frame, N, level, fixed=None):
#         """
#         Internal method to rebin the input `DataFrame`
#
#         Parameters
#         ----------
#         frame : DataFrame
#             the DataFrame that we are rebinning
#         N : int
#             the number of bins to rebin into to
#         level : int
#             the index of the level that we are rebinning, i.e., either the
#             `k` index or the `mu` index
#         fixed : float, optional
#             If not `None`, we are constrained by this value in the other
#             index, i.e., if we are rebinning `k` and fixed = 0.1, then we only
#             rebin `k` where mu = 0.1
#         """
#         # some levels management
#         level_name = frame.index.names[level]
#         other_level = (level + 1) % 2
#
#         # get the index
#         if fixed is not None:
#             if fixed not in frame.index.levels[other_level]:
#                 raise ValueError("`Fixed` value not in index")
#             sliced = frame.xs(fixed, level=other_level)
#             index = np.asarray(sliced.index, dtype=float)
#         else:
#             index = np.asarray(frame.index.levels[level], dtype=float)
#         if N >= len(index):
#             raise ValueError("Can only re-bin into fewer than %d bins" %len(index))
#
#         # new bins
#         diff = np.diff(index)
#         bins = np.linspace(np.amin(index)-0.5*diff[0], np.amax(index)+0.5*diff[-1], N+1)
#
#         mi_values = frame.index.values
#         if fixed is not None:
#             mi_values = [t for t in mi_values if t[other_level] == fixed]
#         values = np.array([val[level] for val in mi_values])
#         bin_numbers = np.digitize(values, bins) - 1
#
#         # make it a series and add it to the data frame
#         names = [self._outer_index_name, 'k']
#         bin_numbers = Series(bin_numbers, name='bin_number', index=Index(mi_values, names=names))
#         frame['bin_number'] = bin_numbers
#
#         # group by bin number and compute mean values in each bin
#         bin_groups = frame.reset_index(level=level).groupby(['bin_number'])
#
#         # replace "bin_number", conditioned on bin_number == new_bins
#         new_bins = bin_groups[level_name].mean()
#         frame[level_name] = new_bins.loc[frame.bin_number].values
#         del frame['bin_number']
#
#         # replace the "missing" data
#         torep = frame.loc[(frame[level_name]).isnull(), level_name].index.get_level_values(level)
#         frame.loc[(frame[level_name]).isnull(), level_name] = torep
#
#         # add mode information, if we have it
#         has_modes = False
#         if self.extra_info is not None and hasattr(self.extra_info, 'modes'):
#             if not hasattr(frame, 'modes'):
#                 has_modes = True
#                 frame['modes'] = self.extra_info.modes
#
#         # group by the new k bins
#         groups = frame.reset_index(level=other_level).groupby(names)
#
#         # apply the average function, either do weighted or unweighted
#         new_frame = groups.apply(tools.groupby_average)
#
#         # delete unneeded columns
#         del new_frame['k'], new_frame[self._outer_index_name]
#         if has_modes: del new_frame['modes']
#
#         # reset the columns, if it's a MultiIndex
#         if isinstance(new_frame.columns, MultiIndex):
#             new_frame.columns = MultiIndex.from_tuples(new_frame.columns.tolist(), names=new_frame.columns.names)
#
#         return new_frame
#
#     #--------------------------------------------------------------------------
#     def rebin_k(self, N, **kwargs):
#         """
#         Rebin the matrix to have `N` `k` bins, possibly only at a fixed
#         `x` value
#         """
#         x = kwargs.get(self._outer_index_name, None)
#         if isinstance(x, int):
#             x = self._convert_x_integer(x)
#
#         # rebin the index
#         frame0 = self._rebin(self.full(), N, level=self._k_level, fixed=x)
#
#         # now rebin the columns, using the transpose
#         frame1 = self._rebin(frame0.transpose(), N, level=self._k_level, fixed=x)
#
#         # get a copy
#         toret = self.copy()
#         super(PowerCovarianceMatrix, toret).__init__(np.asarray(frame1), index=frame1.index)
#
#         # rebin the extra dataframe
#         if self.extra_info is not None:
#             try:
#                 toret.extra_info = self._rebin(self.extra_info.copy(), N, level=self._k_level, fixed=x)
#             except:
#                 pass
#
#         # delete store quantities
#         del toret.inverse, toret.diag
#
#         return toret
#
#     #---------------------------------------------------------------------------
#     # Trimming functions
#     #---------------------------------------------------------------------------
#     def _trim(self, frame, lower, upper, level, fixed=None, trim_transpose=True):
#         """
#         Internal method to trim the covariance matrix
#         """
#         # some levels management
#         level_name = frame.index.names[level]
#         other_level = (level + 1) % 2
#
#         # if not provided, set to +/- infinity
#         if lower is None: lower = -np.inf
#         if upper is None: upper = np.inf
#
#         # get the x and k index values
#         x_vals = frame.index.get_level_values(self._outer_index_name)
#         k_vals = self.ks()
#
#         if level_name == 'k':
#             range_restrict = (k_vals >= lower)&(k_vals <= upper)
#             if fixed is not None:
#                 fixed_restrict = (x_vals == fixed)
#             else:
#                 fixed_restrict = np.ones(len(range_restrict), dtype=bool)
#         else:
#             range_restrict = (x_vals >= lower)&(x_vals <= upper)
#             if fixed is not None:
#                 fixed_restrict = (x_vals == fixed)
#             else:
#                 fixed_restrict = np.ones(len(range_restrict), dtype=bool)
#
#         # now get the indices and return the sliced frame
#         inds = np.logical_or((range_restrict & fixed_restrict), np.logical_not(fixed_restrict))
#         slice1 = frame[inds]
#
#         return slice1.transpose()[inds] if trim_transpose else slice1
#
#     #----------------------------------------------------------------------------
#     def trim_k(self, lower=None, upper=None, **kwargs):
#         """
#         Trim the covariance matrix to specified minimum/maximum k values,
#         possibly only for a specific x value
#         """
#         x = kwargs.get(self._outer_index_name, None)
#         if isinstance(x, int):
#             x = self._convert_x_integer(x)
#
#         if lower is None and upper is None:
#             raise ValueError("Specify at least one of `lower`, `upper` to trim")
#
#         # do the trimming
#         frame = self._trim(self.full(), lower, upper, self._k_level, fixed=x)
#
#         # make a copy
#         toret = self.copy()
#         super(PowerCovarianceMatrix, toret).__init__(np.asarray(frame), index=frame.index)
#
#         # rebin the extra dataframe
#         if self.extra_info is not None:
#             try:
#                 toret.extra_info = self._trim(self.extra_info.copy(), lower, upper,
#                                                 self._k_level, fixed=x, trim_transpose=False)
#             except:
#                 pass
#
#         # delete store quantities
#         del toret.inverse, toret.diag
#
#         return toret
#
#     #---------------------------------------------------------------------------
#     def _trim_x(self, lower=None, upper=None):
#         """
#         Trim the covariance matrix to specified minimum/maximum x values
#         """
#         if lower is None and upper is None:
#             raise ValueError("Specify at least one of `lower`, `upper` to trim")
#
#         # do the trimming
#         frame = self._trim(self.full(), lower, upper, self._x_level)
#
#         # make a copy
#         toret = self.copy()
#         super(PowerCovarianceMatrix, toret).__init__(np.asarray(frame), index=frame.index)
#
#         # rebin the extra dataframe
#         if self.extra_info is not None:
#             try:
#                 toret.extra_info = self._trim(self.extra_info.copy(), lower, upper,
#                                                 self._x_level, trim_transpose=False)
#             except:
#                 pass
#
#         # delete store quantities
#         del toret.inverse, toret.diag
#
#         return toret
#     #---------------------------------------------------------------------------
# #endclass PowerCovarianceMatrix
#
class PkmuCovarianceMatrix(CovarianceMatrix):
    """
    Class to hold a covariance matrix for P(k, mu).
    """
    def __init__(self, data, k_center, mu_center, **kwargs):
        """
        Parameters
        ----------
        data : array_like
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        k_center : array_like, (Nk, Nmu)
            The wavenumbers where the center of the measurement k-bins are defined. This
            must be 2-dimensional with shape (`Nk`, `Nmu`), where `Nk` is the number
            of k bins and `Nmu` is the number of mu bins. Also, if input data has
            shape `N`, then :math: `N = Nk*Nmu` must hold
        mu_center : array_like, (Nk, Nmu)
            The mu values where the center of the measurement mu-bins are defined. 
            Must have 2D shape
        """
        # check the input
        if k_center.ndim != 2 or mu_center.ndim != 2:
            raise ValueError("input k and mu index arrays must both have 2 dimensions")
        if np.shape(k_center) != np.shape(mu_center):
            raise ValueError("size mismatch between supplied k and mu index arrays")
        N = np.shape(data)[0]
        
        # k index
        for name, index in zip(['k_cen', 'mu_cen'], [k_center, mu_center]):
            setattr(self, name, index)
            flat = index.ravel(order='F')
            flat = flat[np.isfinite(flat)]
            setattr(self, name+"_flat", flat)

        if len(self.k_cen_flat) != N or len(self.mu_cen_flat) != N:
            raise ValueError("size mismatch between flattened k/mu index arrays and covariance matrix shape")
        
        # initialize the base class
        names = ['k', 'mu']
        coords = [self.k_cen_flat, self.mu_cen_flat]
        super(PkmuCovarianceMatrix, self).__init__(data, names=names, coords=coords, **kwargs)

    @classmethod 
    def from_spectra_set(cls, power_set, kmin=None, kmax=None, **kwargs):
        """
        Return a PkmuCovarianceMatrix from a SpectraSet of P(k,mu) measurements
        
        Parameters
        ----------
        power_set : lsskit.specksis.SpectraSet
            a set of P(k,mu) power spectrum classes
        kmin : float or array_like, optional
            minimum wavenumber to include in the covariance matrix; if an array is provided, 
            it is intepreted as the minimum value for each mu bin
        kmax : float or array_like, optional
            maximum wavenumber to include in the covariance matrix; if an array is provided, 
            it is intepreted as the minimum value for each mu bin 
        """
        from lsskit.data import tools
        kw = {'kmin':kmin, 'kmax':kmax, 'return_extras':True}
        C, (k_cen, mu_cen), extra = tools.compute_pkmu_covariance(power_set, **kw)
        
        toret = cls(C, k_cen, mu_cen, **kwargs)
        if 'mean_power' in extra:
            toret.mean_power = extra['mean_power']
        if 'modes' in extra:
            toret.modes = extra['modes']
        
        return toret

    #--------------------------------------------------------------------------
    # internal utility functions
    #--------------------------------------------------------------------------
    def _mu_slice(self, mu):
        """
        Return the indices matching the specified slice `mu`
        """
        index = self.mu_cen
        flat_index = self.mu_cen_flat
                
        if isinstance(mu, int):
            inds = np.zeros(index.shape, dtype=bool)
            inds[:,mu] = True
            inds = inds.ravel(order='F')
            inds = inds[np.isfinite(index.ravel(order='F'))]
        else:
            inds = np.isclose(flat_index, mu)
            if not inds.sum():
                raise ValueError("mu = %s is not a valid index value" %str(mu))
        return inds
        
    def _kmu_slice(self, k, mu):
        """
        Return the indices matching the specified slice `k`, `mu`
        """
        mu_inds = self._mu_slice(mu)
        ks = self.ks(mu)
        if isinstance(k, int):
            k_inds = np.isclose(ks, ks[k])
        else:
            k_inds = np.isclose(ks, k)
            if not k_inds.sum():
                raise ValueError("k = %s is not a valid index value" %str(k))
        
        good_mu = mu_inds==True
        mu_inds[good_mu] = mu_inds[good_mu] & k_inds
        return mu_inds
        
    #--------------------------------------------------------------------------
    # properties for extra info
    #--------------------------------------------------------------------------
    @property
    def mean_power(self):
        """
        The mean power, defined with the same shape as `k_cen` and `mu_cen`
        """
        try:
            return self._mean_power
        except AttributeError:
            raise AttributeError("`mean_power` attribute is not set")
    
    @mean_power.setter
    def mean_power(self, val):
        if np.shape(val) != self.k_cen.shape:
            raise ValueError("`mean_power` shape must be %s" %str(self.k_cen.shape))
            
        self._mean_power = val
        flat = val.ravel(order='F')
        self._mean_power_flat = flat[np.isfinite(flat)]
        
    @property
    def mean_power_flat(self):
        """
        The flattened mean power, which is set when `mean_power` is set
        """
        try:
            return self._mean_power_flat
        except AttributeError:
            raise AttributeError("first, set `mean_power` attribute to access `mean_power_flat`")
        
    @property
    def modes(self):
        """
        The number of modes, defined with the same shape as `k_cen` and `mu_cen`
        """
        try:
            return self._modes
        except AttributeError:
            raise AttributeError("`modes` attribute is not set")
    
    @modes.setter
    def modes(self, val):
        if np.shape(val) != self.k_cen.shape:
            raise ValueError("`modes` shape must be %s" %str(self.k_cen.shape))
            
        self._modes = val
        flat = val.ravel(order='F')
        self._modes_flat = flat[np.isfinite(flat)]
        
    @property
    def modes_flat(self):
        """
        The flattened number of modes, which is set when `modes` is set
        """
        try:
            return self._modes_flat
        except AttributeError:
            raise AttributeError("first, set `modes` attribute to access `modes_flat`")
    
    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def trim_k(self, kmin=None, kmax=None):
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
        if kmin is None and kmax is None:
            raise ValueError("both kmin and kmax are `None`, so no trimming is needed")
        if kmin is None: kmin = -np.inf
        if kmax is None: kmax = np.inf
        
        # format the kmin and kmax
        Nmu = len(self.mus())
        kmin_ = np.empty(Nmu)
        kmax_ = np.empty(Nmu)
        kmin_[:] = kmin
        kmax_[:] = kmax
        
        # now format k_cen and mu_cen
        idx = (self.k_cen <= kmax)&(self.k_cen >= kmin)
        flat_idx = idx.ravel(order='F')
        flat_idx = flat_idx[np.isfinite(flat_idx)]
        max_size = idx.sum(axis=0).max()
        
        new_kcen = np.ones((max_size, Nmu))*np.nan
        new_mucen = np.ones((max_size, Nmu))*np.nan
        for i in range(Nmu):
            this_idx = idx[:,i]
            size = this_idx.sum()
            new_kcen[:size, i] = self.k_cen[this_idx, i]
            new_mucen[:size, i] = self.mu_cen[this_idx, i]
            
        return PkmuCovarianceMatrix(self[flat_idx], new_kcen, new_mucen)
        
    def mus(self):
        """
        The unique center values of the `mu` bins. 
        
        Notes
        -----
        This assumes that each column of `mu_center` has the same mu value, 
        such that there number of columns in `mu_center` equals the number of 
        unique values
        """
        return np.unique(self.mu_cen_flat)
            
    def ks(self, mu=None):
        """
        The center values of the `k` bins, where each element of the covariance 
        matrix is defined. If `mu` is `None`, then the `k` values are returned 
        for all `mu` values, else for just a specific value of `mu`
        
        Parameters
        ----------
        mu : int or float, optional
            the mu value (or integer index) specifying which submatrix to 
            return the center k values for
        """
        if mu is None:
            return self.k_cen_flat
        else:
            inds = self._mu_slice(mu)
            return self.k_cen_flat[inds]
    
    def sel_kmu(self, k, mu):
        """
        Return the sub matrix associated with the specified `mu1` and
        `mu2` values

        Parameters
        ----------
        k : {int, float}
            The value of `k` specifying which rows to select of the full matrix
        mu : {int, float}, optional
            The value of `mu` specifying which columns to select of the full
            matrix. If `None`, the value is set to `mu1`

        Returns
        -------
        slice : array_like
            an array holding the sliced covariance matrix
        """
        inds = self._kmu_slice(k, mu)
        return self[inds,inds]
        
    def sel_mu(self, mu1, mu2=None):
        """
        Return the sub matrix associated with the specified `mu1` and
        `mu2` values

        Parameters
        ----------
        mu1 : {int, float}
            The value of `mu` specifying which rows to select of the full matrix
        mu2 : {int, float}, optional
            The value of `mu` specifying which columns to select of the full
            matrix. If `None`, the value is set to `mu1`

        Returns
        -------
        slice : array_like
            an array holding the sliced covariance matrix
        """
        inds1 = self._mu_slice(mu1)
        if mu2 is None:
            inds2 = inds1
        else:
            inds2 = self._mu_slice(mu2)
            
        return self[inds1, inds2]

    def get_modes(self, k=None, mu=None):
        """
        Return the number of modes as a function of `k` and `mu`, or if
        both are set to `None`, at all (k,mu) values defining the
        matrix index
        """
        if k is not None and mu is None:
            raise ValueError("`mu` must also be specified to get modes at specific `k`")
            
        if k is None and mu is None:
            return self.modes_flat
        else:
            if k is not None:
                inds = self._kmu_slice(k, mu)
            else:
                inds = self._mu_slice(mu)
            return self.modes_flat[inds]
    
    def get_mean_power(self, k=None, mu=None):
        """
        Return the mean power as a function of `k` and `mu`, or if
        both are set to `None`, at all (k,mu) values defining the
        matrix index

        NOTE: this has the shot noise included in it
        """
        if k is not None and mu is None:
            raise ValueError("`mu` must also be specified to get mean power at specific `k`")

        if k is None and mu is None:
            return self.mean_power_flat
        else:
            if k is not None:
                inds = self._kmu_slice(k, mu)
            else:
                inds = self._mu_slice(mu)
            return self.mean_power_flat[inds]
    
    def gaussian_covariance(self, k=None, mu=None):
        """
        Return the Gaussian prediction for the variance of the diagonal
        elements, given by:

        :math: 2 / N_{modes}(k,\mu)) * [ P(k,\mu) + P_{shot}(k,\mu) ]**2
        """
        # check the prequisites
        if not hasattr(self, 'mean_power'):
            raise AttributeError("`mean_power` attribute must be set to compute gaussian covariance")
        if not hasattr(self, 'modes'):
            raise AttributeError("`modes` attribute must be set to compute gaussian covariance")

        modes = self.get_modes(k=k, mu=mu)
        mean_power = self.get_mean_power(k=k, mu=mu)

        return 2./modes * mean_power**2

    #---------------------------------------------------------------------------
    # plotting
    #---------------------------------------------------------------------------
    def plot_diagonals(self, ax=None, mu=None, **options):
        """
        Plot the square-root of the diagonal elements for a specfic mu value

        Parameters
        ---------------
        ax : plotify.Axes, optional
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
        import plotify as pfy
        # get the current axes
        if ax is None: ax = pfy.gca()

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
            ax.color_cycle = "Paired"

        # loop over each mu
        for mu in mus:

            # get sigma for this mu sub matrix
            this_mu = self.sel_mu(mu)
            sigma = np.diag(this_mu)**0.5

            norm = 1.
            if options['subtract_gaussian']:
                norm = sigma.copy()
                sigma -= self.gaussian_covariance(mu=mu)**0.5
            elif options['norm_by_gaussian']:
                norm = self.gaussian_covariance(mu=mu)**0.5
            elif options['norm_by_power']:
                norm = self.get_mean_power(mu=mu)

            # do the plotting
            label = r"$\mu = {}$".format(mu)
            pfy.plot(self.ks(mu=mu), sigma/norm, label=label)
            if options['show_gaussian']:
                pfy.plot(self.ks(mu=mu), self.gaussian_covariance(mu=mu)**0.5/norm, ls='--')

            if np.isscalar(norm):
                ax.x_log_scale()
                ax.y_log_scale()

        # add the legend and axis labels
        ax.legend(loc=0, ncol=2)
        ax.xlabel.update(r'$k$ ($h$/Mpc)', fontsize=16)
        if options['norm_by_power']:
            ax.ylabel.update(r"$\sigma_P / P(k, \mu)$", fontsize=16)

        elif options['norm_by_gaussian']:
            ax.ylabel.update(r"$\sigma_P / \sigma_P^\mathrm{Gaussian}$", fontsize=16)
        else:
            if not options['subtract_gaussian']:
                ax.ylabel.update(r"$\sigma_P$ $(\mathrm{Mpc}/h)^3$", fontsize=16)
            else:
                ax.ylabel.update(r"$(\sigma_P - \sigma_P^\mathrm{Gaussian})/\sigma_P$", fontsize=16)

        return ax

    def plot_off_diagonals(self, kbar, ax=None, mu=None, mubar=None, **options):
        """
        Plot the off diagonal elements, specifically:

        :math: \sqrt(Cov(k, kbar) / P(k, mu) / P(kbar, mu))

        Parameters
        ---------------
        kbar : {float, int}
            The specific value of k to plot the covariances at
        ax : plotify.Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        mu : {float, int},
            If not `None`, only plot the diagonals for a single `mu` value, else
            plot for all `mu` values
        options : dict
            Dictionary of options with the following keywords:
                show_diagonal : bool, (`True`)
                    If `True`, plot the diagonal value as a stem plot
                show_binned : bool, (`False`)
                    If `True`, plot a smoothed spline interpolation
                show_zero_line : bool, (`False`)
                    If `True`, plot a y=0 horizontal line
        """
        import plotify as pfy
        import scipy.stats

        # get the current axes
        if ax is None: ax = pfy.gca()

        # setup the default options
        options.setdefault('show_diagonal', True)
        options.setdefault('show_binned', False)
        options.setdefault('show_zero_line', False)

        # determine the mus to plot
        if mu is None:
            mus = self.mus()
            mubars = mus
        else:
            if isinstance(mu, int):
                mu = self.mus()[mu]
            mus = [mu]

            if mubar is not None:
                if isinstance(mubar, int):
                    mubar = self.mus()[mubar]
                mubars = [mubar]
            else:
                mubars = mus

        # loop over each mu
        for mu, mubar in zip(mus, mubars):

            # ks for this mu
            ks = self.ks(mu=mu)
            kbars = self.ks(mu=mubar)

            # get the right kbar
            if isinstance(kbar, int):
                kbar = self.ks(mubar)[kbar]

            # get the closest value in the matrix
            if kbar not in kbars:
                kbar = kbars[abs(kbars - kbar).argmin()]
            kbar_index = abs(kbars - kbar).argmin()
            
            # get sigma for this mu sub matrix
            this_slice = self.sel_mu(mu, mubar)
            cov = this_slice[:,kbar_index]

            # the normalization
            P_k = self.get_mean_power(mu=mu)
            P_kbar = self.get_mean_power(k=kbar, mu=mubar)
            norm = P_k*P_kbar

            # remove the diagonal element
            toplot = cov/norm
            diag_element = toplot[kbar_index]
            toplot = np.delete(toplot, kbar_index)
            ks = ks[~np.isclose(ks, kbar)]

            # plot the fractional covariance
            if mu == mubar:
                args = (mu, kbar)
                label = r"$\mu = \bar{{\mu}} = {0}, \ \bar{{k}} = {1:.3f}$ $h$/Mpc".format(*args)
            else:
                args = (mu, mubar, kbar)
                label = r"$\mu = {0}, \ \bar{{\mu}} = {1}, \ \bar{{k}} = {2:.3f}$ $h$/Mpc".format(*args)
            pfy.plot(ks, toplot, label=label)

            # get the color to use
            this_color = ax.last_color

            # plot the diagonal element
            if options['show_diagonal']:
                markerline, stemlines, baseline = ax.stem([kbar], [diag_element], linefmt=this_color)
                pfy.plt.setp(markerline, 'markerfacecolor', this_color)
                pfy.plt.setp(markerline, 'markeredgecolor', this_color)

            # plot a smoothed spline
            if options['show_binned']:
                nbins = tools.histogram_bins(toplot)
                y, binedges, w = scipy.stats.binned_statistic(ks, toplot, bins=nbins)
                pfy.plot(0.5*(binedges[1:]+binedges[:-1]), y, color=this_color)

        if options['show_zero_line']:
            ax.axhline(y=0, c='k', ls='--', alpha=0.6)

        # add the legend and axis labels
        ax.legend(loc=0, ncol=2)
        ax.xlabel.update(r'$k$ $h$/Mpc', fontsize=16)
        ax.ylabel.update(r"$\mathrm{Cov}(k, \bar{k}) / (P(k, \mu) P(\bar{k}, \bar{\mu}))$", fontsize=16)

        return ax

