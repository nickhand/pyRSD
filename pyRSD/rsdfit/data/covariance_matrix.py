try:
    import plotify as pfy
except:
    msg = "Covariance matrix plotting requires `plotify` plotting package"
    raise ImportError(msg)
    
from ... import numpy as np
from . import tools
import copy
from pandas import DataFrame, Index, MultiIndex, Series
import pickle
import itertools

#-------------------------------------------------------------------------------
def load_covariance(filename):
    """
    Load the covariance matrix
    """
    return pickle.load(open(filename, 'r'))

#-------------------------------------------------------------------------------
class CovarianceMatrix(object):
    """
    Class to represent a covariance matrix. The class has an associated 
    `index`, which can be used to associated values with each element of the 
    matrix. This is useful for trimming the matrix to a given `index` range.
    """
    def __init__(self, data, index=None, verify=True):
        """
        Parameters
        ----------
        data : array_like
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        index : array_like, Index, MultiIndex optional
            An array of values associated with each axis of the matrix
        verify : bool, optional
            If `True`, verify that the matrix is positive-semidefinite and
            symmetric
        """
        # make local copies
        data = np.asarray(data).copy()
        
        # make a diagonal 2D matrix from input
        if data.ndim == 1:
            data = np.diag(data)
            
        # check basic properties
        if verify:
            self.check_properties(data)
        
        # initialize the backend        
        self._initialize_backend(data, index)
    
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        name = str(self.__class__).split('.')[-1].split("'")[0]
        return "<{name}: size {N}x{N}>".format(name=name, N=self.N)
    
    def __str__(self):
        """
        Builtin string method
        """
        return str(self.full())
        
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        """
        Access individual elements through array-like interface:

            `element = self[i, j]`

        where `i` and `j` are either integers in `[0, self.N]` or floats 
        specifying a key in `self.index`
        """
        # if only one key provided, return the diagonal element
        if not isinstance(key, tuple):
            key = (key, key)
            
        # if two ints, specified return like a usual matrix
        if len(key) == 2 and all(isinstance(k, int) for k in key):
            index = self._get_contracted_index(key[0], key[1])
            return self.data.iloc[index]
        
        # flatten any tuple of tuples
        key = tuple(np.ravel(key))
        
        # check if we have enough keys for this index
        if len(key) != 2*self._index_levels:
            key += key
                           
        # check to make sure the key is correct
        if not isinstance(key, tuple) or len(key) != 2*self._index_levels:
            raise KeyError("Do not understand key: {}".format(key))
          
        # check for ints in the right instance
        for k in key:
            if not isinstance(k, int):
                if self.index is None:
                    raise ValueError("Key must be an integer if no index is provided")
                if self._duplicate_index and self._index_levels == 1:
                    msg = "Key must be an integer if index contains duplicate keys with no MultiIndex"
                    raise ValueError(msg)
        
        # all integers
        if all(isinstance(k, int) for k in key):
            vals = tuple(self.data.index.levels[i][k] for i, k in enumerate(key))
            return self.data.loc[vals]
                
        # now do the the index case
        if key not in self.data.index:
            midpt = len(key)/2
            key = key[midpt:] + key[:midpt]
            if key not in self.data.index:
                raise KeyError("Sorry, element with key `{}` does not exist".format(key))

        return self.data.loc[key]
        
    #---------------------------------------------------------------------------
    def __add__(self, other):
        toret = self.copy()
        toret.data += other
        return toret

    def __radd__(self, other):
        return self.__add__(other)
        
    #---------------------------------------------------------------------------
    def __sub__(self, other):
        toret = self.copy()
        toret.data -= other
        return toret

    def __rsub__(self, other):
        return self.__sub__(other)
        
    #---------------------------------------------------------------------------
    def __mul__(self, other):
        toret = self.copy()
        toret.data *= other
        return toret

    def __rmul__(self, other):
        return self.__mul__(other)
        
    #---------------------------------------------------------------------------
    def __div__(self, other):
        toret = self.copy()
        toret.data /= other
        return toret

    def __rdiv__(self, other):
        return self.__div__(other)
        
    #---------------------------------------------------------------------------
    def __pow__(self, other):
        toret = self.copy()
        toret.data = toret.data**other
        return toret
                
    #---------------------------------------------------------------------------
    def _get_contracted_index(self, i, j):
        """
        Given the matrix element (i, j) return the index of the corresponding
        element in `self.data`
        """
        if i < 0: i += self.N
        if j < 0: j += self.N
        
        if not 0 <= i < self.N: raise IndexError("Index %d out of range" %i)
        if not 0 <= j < self.N: raise IndexError("Index %d out of range" %j)
        
        if i > j:
            i, j = j, i
            
        return i*(self.N-1) - sum(range(i)) + j
        
    #---------------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy of `self`
        """
        return copy.deepcopy(self)
        
    #---------------------------------------------------------------------------
    def _initialize_backend(self, data, index):
        """
        Initialize the `Pandas` backend for storing the covariance matrix
        """
        # store the size
        self.N = np.shape(data)[0]
        
        # check the index
        self._duplicate_index = False
        if index is not None:            
            # check the shape, etc
            if len(index) != self.N:
                raise ValueError("Shape mismatch between data and index for covariance matrix")

            # if input is just array, make it an Index
            if not isinstance(index, (Index, MultiIndex)):
                index = Index(index)
                
            # check for duplicate index
            self._duplicate_index = len(index.unique()) == len(index)

        else:
            index = Index(range(self.N))
            
        # store the index
        self.index = index
        
        # setup names
        if not isinstance(index, MultiIndex):
            index_i = index.copy()
            index_j = index.copy()
            index_i.name = 'i'
            index_j.name = 'j'
            self._index_levels = 1
        else:
            index_i = index_j = index
            self._index_levels = len(index.levels)
        
        # now make the frame
        frame = DataFrame(data, index=index_j, columns=index_i)
    
        # and make it into a series
        frame.values[np.triu_indices_from(frame, k=1)] = np.nan
        if self._index_levels > 1:
            self.data = frame.unstack(level=range(self._index_levels)).dropna()
        else:
            self.data = frame.unstack().dropna()
        
    #---------------------------------------------------------------------------
    def check_properties(self, data):
        
        if not np.allclose(data, data.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        if not np.all(np.linalg.eigvals(data) >= 0.):
            raise ValueError("Covariance matrix is not positive-semidefinite")
    
    #---------------------------------------------------------------------------
    @property
    def shape(self):
        return (self.N, self.N)
    
    #---------------------------------------------------------------------------
    @property
    def diag(self):
        """
        Return the diagonal elements
        """
        try:
            return self._diag
        except AttributeError:
            inds = [self._get_contracted_index(i, i) for i in range(self.N)]
            self._diag = np.array(self.data.iloc[inds])
            
            return self._diag
    
    @diag.deleter
    def diag(self):
        if hasattr(self, '_diag'): delattr(self, '_diag')

    #---------------------------------------------------------------------------
    def full(self):
        """
        Return the full 2D covariance matrix array
        """
        frame = self.data.unstack(level=range(self._index_levels))
        i, j = np.triu_indices_from(frame, k=1)
        frame.values[i, j] = frame.values[j, i]
        return frame
        
    #---------------------------------------------------------------------------
    def asarray(self):    
        """
        Return as a 2D numpy.array
        """
        return np.array(self.full())
    
    #---------------------------------------------------------------------------
    def normalized(self):
        """
        Return the normalized covariance matrix, i.e., the correlation matrix
        """
        ii, jj = np.triu_indices(self.N)
        toret = self.copy()
        
        # normalize by the diagonals
        diag = self.diag
        norm = np.array([np.sqrt(diag[i]*diag[j]) for (i, j) in zip(ii, jj)])
        toret.data /= norm
        return toret
        
    #---------------------------------------------------------------------------
    def plot(self, filename=None):
        """
        Plot the correlation matrix (normalized covariance matrix), optionally
        saving if `filename != None`
        """

        
        pfy.clf()
        corr = self.normalized().asarray()
        colormesh = pfy.pcolormesh(corr, vmin=-1, vmax=1)
        pfy.colorbar(colormesh)
        if filename is not None:
            pfy.savefig(filename)
    
        return colormesh
        
    #---------------------------------------------------------------------------
    @property
    def inverse(self):
        """
        The inverse of the CovarianceMatrix, returned as a 2D numpy.array
        """
        try:
            return self._inverse
        except AttributeError:
            self._inverse = np.linalg.inv(self.full())
            return self._inverse
    
    @inverse.deleter
    def inverse(self):
        if hasattr(self, '_inverse'): delattr(self, '_inverse')   
            
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Pickle the CovarianceMatrix
        """
        pickle.dump(self, open(filename, 'w'))
    
    #---------------------------------------------------------------------------
#endclass CovarianceMatrix

#-------------------------------------------------------------------------------
class PowerCovarianceMatrix(CovarianceMatrix):
    """
    Class to hold a covariance matrix for a power measurement, either P(k, mu)
    or a multipole moment
    """
    _allowed_extra_info = []
    
    def __init__(self, outer_index_name, data, ks, xs, units, h, **extra_info):

        # check the name of the outer index level
        if outer_index_name not in ['mu', 'ell']:
            raise ValueError("string identifier for outer index must be either `mu` or `ell`")
        self._outer_index_name = outer_index_name
        
        # if ks is not right length, assume we want to repeat
        if len(ks) != len(data):
            N = len(data) / len(ks)
            ks = np.concatenate((ks,)*N)
        
        # if outer index is not right length, assume we want to repeat
        if len(xs) != len(data):
            N = len(data) / len(xs)
            xs = np.concatenate([(x,)*N for x in xs])
            
        # tuples of (x, k)
        xks = zip(xs, ks)
        
        # do a few checks
        if len(xks) != len(data):
            msg = "Size mismatch been data and (k, {}) values".format(self._outer_index_name)
            raise ValueError(msg)
              
        # make the index now
        index = MultiIndex.from_tuples(xks, names=[self._outer_index_name, 'k'])
        
        # initialize the base class
        super(PowerCovarianceMatrix, self).__init__(data, index=index)
        
        # keep track of units
        if units not in ['absolute', 'relative']:
            raise ValueError("`units` must be one of [`absolute`, `relative`]")
        self._units = units
        
        # store h
        self._h = h
        
        # add the extra information
        self._store_extra_info(**extra_info)
        
    #---------------------------------------------------------------------------
    # unit conversions
    #---------------------------------------------------------------------------
    def _update_units(self, units, h):
        """
        Update the values in `self.data` to reflect the latest changes in 
        the units conversions
        """
        # update 'k' levels in self.data
        new_levels = []
        for i, name in enumerate(self.data.index.names):
            factor = 1.
            if name == 'k':
                factor = self._k_conversion_factor(units, h)
            new_levels.append(self.data.index.levels[i]*factor)
        self.data.index = self.data.index.set_levels(new_levels)
        
        # multiply by the power values
        self.data *= self._power_conversion_factor(units, h)**2
        
        # update the extra info too
        self._update_extra_info_units(units, h)
        
    #---------------------------------------------------------------------------
    @property
    def h(self):
        """
        The dimensionless Hubble factor to which the current output is scaled
        """
        return self._h
    
    @h.setter
    def h(self, val):
        
        # handle dependencies first        
        self._update_units(self.units, val)
        del self.inverse, self.diag
        
        # update h
        self._h = val
    
    #---------------------------------------------------------------------------
    @property
    def units(self):
        """
        The type of units to use for output quantities, given either by 
        `relative` or `absolute`. 
        """
        return self._units
        
    @units.setter
    def units(self, val):
        
        if val not in ['absolute', 'relative', None]:
            raise AttributeError("`units` must be one of [`absolute`, `relative`]")
            
        # handle dependencies first
        self._update_units(val, self.h)
        del self.inverse, self.diag
        
        # update the units
        self._units = val
        
    #---------------------------------------------------------------------------
    def _h_conversion_factor(self, data_type, from_units, to_units, h):
        """
        Internal method to compute units conversions
        """
        factor = 1.
        if from_units is None or to_units is None:
            return factor
        
        if to_units != from_units:
            factor = tools.h_conversion_factor(data_type, from_units, to_units, h)
            
        return factor
    
    #---------------------------------------------------------------------------
    def _k_conversion_factor(self, new_units, new_h):
        """
        Conversion factor for wavenumber
        """        
        # first convert to absolute units with original h
        factor1 = self._h_conversion_factor('wavenumber', self.units, 'absolute', self.h)
        
        # then to new units with desired h
        factor2 = self._h_conversion_factor('wavenumber', 'absolute', new_units, new_h)
        
        return factor1*factor2

    #---------------------------------------------------------------------------
    def _power_conversion_factor(self, new_units, new_h):
        """
        Conversion factor for power
        """
        # first convert to absolute units with original h
        factor1 = self._h_conversion_factor('power', self.units, 'absolute', self.h)

        # then to output_units with desired h
        factor2 = self._h_conversion_factor('power', 'absolute', new_units, new_h)

        return factor1*factor2    
        
    #---------------------------------------------------------------------------
    # k, x handling
    #---------------------------------------------------------------------------
    @property
    def _k_level(self):
     """
     Internal variable defining which level the `k` column is defined in 
     the `MultiIndex`
     """
     try:
         return self.__k_level
     except AttributeError:
         if 'k' not in self.index.names:
             raise ValueError("Index for `PowerCovarianceMatrix` does not contain `k`")
         self.__k_level = np.where(np.asarray(self.index.names) == 'k')[0][0]
         return self.__k_level
    
    #---------------------------------------------------------------------------
    @property
    def _x_level(self):
        """
        Internal variable defining which level the `self._outer_index_name` 
        column is defined in the `MultiIndex`
        """
        try:
            return self.__x_level
        except AttributeError:
            x = self._outer_index_name
            if x not in self.index.names:
                raise ValueError("Index for `PowerCovarianceMatrix` does not contain '%s'" %x)
            self.__x_level = np.where(np.asarray(self.index.names) == x)[0][0]
            return self.__x_level
            
    #---------------------------------------------------------------------------
    def _convert_x_integer(self, x_int):
        """
        Convert the integer value specifying a given outer index value 
        to the actual value
        """
        vals = getattr(self, self._outer_index_name + "s")
        if x_int in vals:
            return x_int
            
        msg = "[0, %d) or [-%d, -1]" %(len(vals), len(vals))
        if x_int < 0: x_int += len(vals)
        if not 0 <= x_int < len(vals):
            raise KeyError("Integer that identifies %s-band must be between %s" 
                            %(self._outer_index_name, msg))
        return vals[x_int]
        
    #---------------------------------------------------------------------------
    def _convert_k_integer(self, k_int, x):
        """
        Convert the integer value specifying a given `k` (associated with 
        a specific `x`) to the actual value
        """
        # get the ks for this mu
        ks = self.ks(**{self._outer_index_name:x})
        
        msg = "[0, %d) or [-%d, -1]" %(len(ks), len(ks))
        if k_int < 0: k_int += len(ks)
        if not 0 <= k_int < len(ks):
            raise KeyError("Integer that identifies k-band for %s = %s must be between %s" 
                            %(self._outer_index_name, x, msg))
        return ks[k_int]
    
    #---------------------------------------------------------------------------
    @property
    def _xs(self):
        """
        The `x` values associated with the matrix; only unique `x` values
        are returned
        """
        return self.index.get_level_values(self._outer_index_name).unique()
        
    #---------------------------------------------------------------------------
    def ks(self, **kwargs):
        """
        The `k` values where each element of the covariance matrix is defined.
        If `x` is `None`, then the `k` values are returned for all `x` values,
        else for just a specific value
        
        """
        x = kwargs.get(self._outer_index_name, None)
        
        # possibly convert the integer to the right x
        if isinstance(x, int):
            x = self._convert_x_integer(x)
           
        # get the values
        vals = self.index.get_level_values('k')
        if x is not None:
            inds = self.index.get_level_values(self._outer_index_name) == x
            vals = vals[inds]
        
        return np.asarray(vals)
        
    #---------------------------------------------------------------------------
    # slicing functions
    #---------------------------------------------------------------------------
    def _x_slice(self, x1, x2=None):
        """
        Return the sub matrix associated with the specified `x1` and `x2` values
        
        Note: this returns a `DataFrame`
        """
        if isinstance(x1, int):
            x1 = self._convert_x_integer(x1)
        if x2 is not None:
            if isinstance(x2, int):
                x2 = self._convert_x_integer(x2)
        else:
            x2 = x1

        # slice both the rows and columns
        full = self.full()
        slice1 = full.xs(x1, level=self._outer_index_name, axis=0)
        slice2 = slice1.xs(x2, level=self._outer_index_name, axis=1)
        return slice2
        
    #---------------------------------------------------------------------------
    def slice(self, k, x):
        """
        Return a slice of the covariance matrix at the specified (k, mu) 
        or (k, ell) values
        """
        # get the correct k, mu from integer input
        if isinstance(x, int):
            x = self._convert_x_integer(x)
        if isinstance(k, int):
            k = self._convert_k_integer(k, x)
                
        return self.full()[x, k]
        
    #---------------------------------------------------------------------------
    # trimming functions
    #---------------------------------------------------------------------------
    def _rebin(self, frame, N, level, fixed=None):
        """
        Internal method to rebin the input `DataFrame`
        
        Parameters
        ----------
        frame : DataFrame
            the DataFrame that we are rebinning
        N : int
            the number of bins to rebin into to
        level : int
            the index of the level that we are rebinning, i.e., either the 
            `k` index or the `mu` index
        fixed : float, optional
            If not `None`, we are constrained by this value in the other 
            index, i.e., if we are rebinning `k` and fixed = 0.1, then we only
            rebin `k` where mu = 0.1
        """  
        # some levels management
        level_name = frame.index.names[level]
        other_level = (level + 1) % 2
                  
        # get the index
        if fixed is not None:
            if fixed not in frame.index.levels[other_level]:
                raise ValueError("`Fixed` value not in index")
            sliced = frame.xs(fixed, level=other_level)
            index = np.asarray(sliced.index, dtype=float) 
        else:
            index = np.asarray(frame.index.levels[level], dtype=float)      
        if N >= len(index):
            raise ValueError("Can only re-bin into fewer than %d bins" %len(index))
        
        # new bins
        diff = np.diff(index)
        bins = np.linspace(np.amin(index)-0.5*diff[0], np.amax(index)+0.5*diff[-1], N+1)
    
        mi_values = frame.index.values
        if fixed is not None:
            mi_values = [t for t in mi_values if t[other_level] == fixed]
        values = np.array([val[level] for val in mi_values])
        bin_numbers = np.digitize(values, bins) - 1
        
        # make it a series and add it to the data frame
        names = [self._outer_index_name, 'k']
        bin_numbers = Series(bin_numbers, name='bin_number', index=Index(mi_values, names=names))
        frame['bin_number'] = bin_numbers
        
        # group by bin number and compute mean values in each bin
        bin_groups = frame.reset_index(level=level).groupby(['bin_number'])
           
        # replace "bin_number", conditioned on bin_number == new_bins
        new_bins = bin_groups[level_name].mean() 
        frame[level_name] = new_bins.loc[frame.bin_number].values
        del frame['bin_number']
        
        # replace the "missing" data
        torep = frame.loc[(frame[level_name]).isnull(), level_name].index.get_level_values(level)
        frame.loc[(frame[level_name]).isnull(), level_name] = torep
        
        # add mode information, if we have it
        has_modes = False
        if self.extra_info is not None and hasattr(self.extra_info, 'modes'):
            if not hasattr(frame, 'modes'):
                has_modes = True
                frame['modes'] = self.extra_info.modes
        
        # group by the new k bins
        groups = frame.reset_index(level=other_level).groupby(names)

        # apply the average function, either do weighted or unweighted
        new_frame = groups.apply(tools.groupby_average)
        
        # delete unneeded columns
        del new_frame['k'], new_frame[self._outer_index_name]
        if has_modes: del new_frame['modes']
        
        # reset the columns, if it's a MultiIndex
        if isinstance(new_frame.columns, MultiIndex):
            new_frame.columns = MultiIndex.from_tuples(new_frame.columns.tolist(), names=new_frame.columns.names)
        
        return new_frame
    
    #--------------------------------------------------------------------------
    def rebin_k(self, N, **kwargs):
        """
        Rebin the matrix to have `N` `k` bins, possibly only at a fixed 
        `x` value
        """
        x = kwargs.get(self._outer_index_name, None)
        if isinstance(x, int):
            x = self._convert_x_integer(x)
            
        # rebin the index
        frame0 = self._rebin(self.full(), N, level=self._k_level, fixed=x)
        
        # now rebin the columns, using the transpose
        frame1 = self._rebin(frame0.transpose(), N, level=self._k_level, fixed=x)
        
        # get a copy 
        toret = self.copy()
        super(PowerCovarianceMatrix, toret).__init__(np.asarray(frame1), index=frame1.index)
        
        # rebin the extra dataframe
        if self.extra_info is not None:
            try:
                toret.extra_info = self._rebin(self.extra_info.copy(), N, level=self._k_level, fixed=x)
            except:
                pass
        
        # delete store quantities
        del toret.inverse, toret.diag
        
        return toret
        
    #---------------------------------------------------------------------------
    # Trimming functions
    #---------------------------------------------------------------------------
    def _trim(self, frame, lower, upper, level, fixed=None, trim_transpose=True):
        """
        Internal method to trim the covariance matrix
        """
        # some levels management
        level_name = frame.index.names[level]
        other_level = (level + 1) % 2
 
        # if not provided, set to +/- infinity
        if lower is None: lower = -np.inf
        if upper is None: upper = np.inf
               
        # get the x and k index values
        x_vals = frame.index.get_level_values(self._outer_index_name)
        k_vals = self.ks()
        
        if level_name == 'k':
            range_restrict = (k_vals >= lower)&(k_vals <= upper)
            if fixed is not None:
                fixed_restrict = (x_vals == fixed)
            else:
                fixed_restrict = np.ones(len(range_restrict), dtype=bool)
        else:
            range_restrict = (x_vals >= lower)&(x_vals <= upper)
            if fixed is not None:
                fixed_restrict = (x_vals == fixed)
            else:
                fixed_restrict = np.ones(len(range_restrict), dtype=bool)       
                
        # now get the indices and return the sliced frame
        inds = np.logical_or((range_restrict & fixed_restrict), np.logical_not(fixed_restrict))
        slice1 = frame[inds]
        
        return slice1.transpose()[inds] if trim_transpose else slice1
           
    #----------------------------------------------------------------------------
    def trim_k(self, lower=None, upper=None, **kwargs):
        """
        Trim the covariance matrix to specified minimum/maximum k values, 
        possibly only for a specific x value
        """
        x = kwargs.get(self._outer_index_name, None)
        if isinstance(x, int):
            x = self._convert_x_integer(x)
            
        if lower is None and upper is None:
            raise ValueError("Specify at least one of `lower`, `upper` to trim")
            
        # do the trimming
        frame = self._trim(self.full(), lower, upper, self._k_level, fixed=x)

        # make a copy
        toret = self.copy()
        super(PowerCovarianceMatrix, toret).__init__(np.asarray(frame), index=frame.index)
        
        # rebin the extra dataframe
        if self.extra_info is not None:
            try:
                toret.extra_info = self._trim(self.extra_info.copy(), lower, upper, 
                                                self._k_level, fixed=x, trim_transpose=False)
            except:
                pass
                
        # delete store quantities
        del toret.inverse, toret.diag
        
        return toret
        
    #---------------------------------------------------------------------------
    def _trim_x(self, lower=None, upper=None):
        """
        Trim the covariance matrix to specified minimum/maximum x values
        """
        if lower is None and upper is None:
            raise ValueError("Specify at least one of `lower`, `upper` to trim")

        # do the trimming
        frame = self._trim(self.full(), lower, upper, self._x_level)

        # make a copy
        toret = self.copy()
        super(PowerCovarianceMatrix, toret).__init__(np.asarray(frame), index=frame.index)
        
        # rebin the extra dataframe
        if self.extra_info is not None:
            try:
                toret.extra_info = self._trim(self.extra_info.copy(), lower, upper, 
                                                self._x_level, trim_transpose=False)
            except:
                pass
                
        # delete store quantities
        del toret.inverse, toret.diag   
          
        return toret
    #---------------------------------------------------------------------------
#endclass PowerCovarianceMatrix
  
#-------------------------------------------------------------------------------
class PkmuCovarianceMatrix(PowerCovarianceMatrix):
    """
    Class to hold a covariance matrix for P(k, mu)
    """
    _allowed_extra_info = ['modes', 'mean_power']
    
    def __init__(self, data, ks, mus, units, h, **extra_info):
        """
        Parameters
        ----------
        data : array_like
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        ks : array_like
            The wavenumbers where the measurements are defined. For every `mu`
            measurement, it is assumed that the measurement is defined at
            the same k values. Units are given by the `units` keyword
        mus : array_like
            The mu values where the measurements are defined
        """
        # set the defaults for keywords
        for k in self._allowed_extra_info:
            extra_info.setdefault(k, None)
        
        # initialize the base class
        super(PkmuCovarianceMatrix, self).__init__('mu', data, ks, mus, units, h, **extra_info)

    #---------------------------------------------------------------------------
    def _store_extra_info(self, **kwargs):
        """
        Store the extra information needed to compute analytic variances
        """
        info = {}
        for k in self._allowed_extra_info:
            if kwargs.get(k, None) is not None:
                if len(kwargs[k]) != len(self.index):
                    msg = "size mismatch in PkmuCovarianceMatrix between index and %s info" %k
                    raise ValueError(msg)
                info[k] = kwargs[k]
        
        if len(info) > 0:    
            self.extra_info = DataFrame(data=info, index=self.index)
        else:
            self.extra_info = None
        
    #---------------------------------------------------------------------------
    def _update_extra_info_units(self, units, h):
        """
        Update the units store in the `extra_info` DataFrame
        """
        if self.extra_info is not None:
            
            # update 'k' levels 
            new_levels = []
            for i, name in enumerate(self.extra_info.index.names):
                factor = 1.
                if name == 'k':
                    factor = self._k_conversion_factor(units, h)
                new_levels.append(self.extra_info.index.levels[i]*factor)
            self.extra_info.index = self.extra_info.index.set_levels(new_levels)

            # multiply by the power values
            if hasattr(self.extra_info, 'mean_power'):
                self.extra_info['mean_power'] *= self._power_conversion_factor(units, h)
        
    #---------------------------------------------------------------------------
    @property
    def mus(self):
        """
        The `mu` values associated with the matrix; only unique `mu` values
        are returned
        """
        return self._xs
        
    #---------------------------------------------------------------------------
    def mu_slice(self, mu1, mu2=None):
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
        slice : {PkmuCovarianceMatrix, DataFrame}
            If the sliced submatrix is not symmetric, a `DataFrame` is returned,
            else a `PkmuCovarianceMatrix` is returned
        """
        return self._x_slice(mu1, mu2)
                
    #---------------------------------------------------------------------------
    def trim_mu(self, lower=None, upper=None):
        """
        Trim the covariance matrix to the specified minimum/maximum `mu` values
        
        Parameters
        ----------
        lower : float
            Minimum `mu` value to include in the matrix
        upper : float
            Maximum `mu` value to include in the matrix
        """
        return self._trim_x(lower=lower, upper=upper)
        
    #---------------------------------------------------------------------------
    # extra info
    #---------------------------------------------------------------------------
    def modes(self, k=None, mu=None):
        """
        The number of modes as a function of `k` and `mu`
        """
        if hasattr(self.extra_info, 'modes'):
            if k is None and mu is None:
                return self.extra_info.modes
            else:
                # slice at a certain mu
                if mu is not None:
                    if isinstance(mu, int): 
                        mu = self._convert_x_integer(mu)
                    modes = self.extra_info.modes.xs(mu, level='mu')

                # slice at specific k
                if k is not None:
                    if isinstance(k, int):
                        k = self._convert_k_integer(k, mu)
                    modes = modes.loc[k]
                return modes
        else:
            return None

    #---------------------------------------------------------------------------
    def mean_power(self, k=None, mu=None):
        """
        The mean power as a function of `k` and `mu`, in the units specified
        by `self.h` and `self.units`

        NOTE: this has the shot noise included in it
        """
        if hasattr(self.extra_info, 'mean_power'):
            if k is None and mu is None:
                return self.extra_info.mean_power

            else:    
                # slice at a certain mu
                if mu is not None:
                    if isinstance(mu, int): 
                        mu = self._convert_x_integer(mu)
                    P = self.extra_info.mean_power.xs(mu, level='mu')

                # slice at specific k
                if k is not None:
                    if isinstance(k, int):
                        k = self._convert_k_integer(k, mu)
                    P = P.loc[k]
                return P
        else:
            return None

    #---------------------------------------------------------------------------
    def gaussian_covariance(self, k=None, mu=None):
        """
        Return the Gaussian prediction for the variance of the diagonal 
        elements, given by: 

        :math: 2 / N_{modes}(k,\mu)) * [ P(k,\mu) + P_{shot}(k,\mu) ]**2
        """
        if k is not None and mu is None:
            msg = "`mu` must also be specified to compute Gaussian error at specific `k`"
            raise ValueError(msg)

        modes = self.modes(k=k, mu=mu)
        if modes is None:
            raise ValueError("Cannot compute Gaussian sigma without `modes`")

        mean_power = self.mean_power(k=k, mu=mu)
        if mean_power is None:
            raise ValueError("Cannot compute Gaussian sigma without `mean_power`")
 
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
                show_gaussian : bool
                    Overplot the Gaussian prediction, i.e., 
                    :math: 2/N_{modes} * [P(k) + Pshot(k)]**2
                norm_by_gaussian : bool
                    Normalize the diagonal elements by the Gaussian prediction
                norm_by_power : bool
                    Normalize the diagonal elements by the mean power, 
                    P(k) + Pshot(k)
                subtract_gaussian : bool
                    Subtract out the Gaussian prediction
        """     
        # get the current axes
        if ax is None: ax = pfy.gca()
        
        # setup the default options
        options.setdefault('show_gaussian', False)
        options.setdefault('norm_by_gaussian', False)
        options.setdefault('norm_by_power', False)
        options.setdefault('subtract_gaussian', False)
        
        # get the mus to plot
        if mu is None:
            mus = self.mus
        else:
            if isinstance(mu, int): 
                mu = self._convert_x_integer(mu)
            mus = [mu]
            
        if options['show_gaussian']:
            ax.color_cycle = "Paired"
            
        # loop over each mu
        for mu in mus:
                        
            # get sigma for this mu sub matrix
            this_mu = self.mu_slice(mu)
            sigma = np.diag(this_mu)**0.5
            
            norm = 1.
            if options['subtract_gaussian']:
                norm = sigma.copy()
                sigma -= self.gaussian_covariance(mu=mu)**0.5
            elif options['norm_by_gaussian']:
                norm = self.gaussian_covariance(mu=mu)**0.5
            elif options['norm_by_power']:
                norm = self.mean_power(mu=mu)
                
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
        if self.units == 'relative':
            k_units = r"($h$/Mpc)"
            P_units = r"$(\mathrm{Mpc}/h)^3$"
        else:
            k_units = "(1/Mpc)"
            P_units = r"$(\mathrm{Mpc})^3$"
            
        ax.xlabel.update(r'$k$ %s' %k_units, fontsize=16)
        if options['norm_by_power']:
            ax.ylabel.update(r"$\sigma_P / P(k, \mu)$", fontsize=16)

        elif options['norm_by_gaussian']:
            ax.ylabel.update(r"$\sigma_P / \sigma_P^\mathrm{Gaussian}$", fontsize=16)
        else:
            if not options['subtract_gaussian']:
                ax.ylabel.update(r"$\sigma_P$ %s" %P_units, fontsize=16)
            else:
                ax.ylabel.update(r"$(\sigma_P - \sigma_P^\mathrm{Gaussian})/\sigma_P$", fontsize=16)

        return ax
        
    #---------------------------------------------------------------------------
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
                show_diagonal : bool
                    If `True`, plot the diagonal value as a stem plot
                show_binned : bool
                    If `True`, plot a smoothed spline interpolation
                show_zero_line : bool
                    If `True`, plot a y=0 horizontal line
        """
        import scipy.stats
                
        # get the current axes
        if ax is None: ax = pfy.gca()
        
        # setup the default options
        options.setdefault('show_diagonal', True)
        options.setdefault('show_binned', False)
        options.setdefault('show_zero_line', False)
        
        # determine the mus to plot
        if mu is None:
            mus = self.mus
            mubars = mus
        else:
            if isinstance(mu, int): 
                mu = self._convert_x_integer(mu)
            mus = [mu]
            
            if mubar is not None:
                if isinstance(mubar, int): 
                    mubar = self._convert_x_integer(mubar)
                mubars = [mubar]
            else:
                mubars = mus
        
        # get the k units
        if self.units == 'relative':
            k_units = r"$h$/Mpc"
        else:
            k_units = "1/Mpc"
                
        # loop over each mu
        for mu, mubar in zip(mus, mubars):

            # ks for this mu
            ks = self.ks(mu=mu)
            kbars = self.ks(mu=mubar)
            
            # get the right kbar
            if isinstance(kbar, int):
                kbar = self._convert_k_integer(kbar, mubar)
            
            # get the closest value in the matrix
            if kbar not in kbars:
                kbar = kbars[abs(kbars - kbar).argmin()]
                    
            # get sigma for this mu sub matrix
            this_slice = self.mu_slice(mu, mu2=mubar)
            cov = this_slice.xs(kbar, axis=1)
            
            # the normalization
            P_k = self.mean_power(mu=mu)
            P_kbar = self.mean_power(k=kbar, mu=mubar)
            norm = P_k*P_kbar
            
            # remove the diagonal element
            toplot = cov/norm
            diag_element = toplot[kbar]
            toplot = toplot.drop(kbar)
            ks = ks[ks != kbar]
            
            # plot the fractional covariance
            if mu == mubar:
                args = (mu, kbar, k_units)
                label = r"$\mu = \bar{{\mu}} = {0}, \ \bar{{k}} = {1:.3f}$ {2}".format(*args)
            else:
                args = (mu, mubar, kbar, k_units)
                label = r"$\mu = {0}, \ \bar{{\mu}} = {1}, \ \bar{{k}} = {2:.3f}$ {3}".format(*args)
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
        ax.xlabel.update(r'$k$ %s' %k_units, fontsize=16)
        ax.ylabel.update(r"$\mathrm{Cov}(k, \bar{k}) / (P(k, \mu) P(\bar{k}, \bar{\mu}))$", fontsize=16)
        
        return ax
    #---------------------------------------------------------------------------
#endclass PkmuCovarianceMatrix


#-------------------------------------------------------------------------------
class PoleCovarianceMatrix(PowerCovarianceMatrix):
    """
    Class to hold a covariance matrix for multipole power moments
    """
    _allowed_extra_info = ['mus', 'modes', 'pkmu_power', 'pkmu_variance']
    
    def __init__(self, data, ks, ells, units, h, **extra_info):
        """
        Parameters
        ----------
        data : array_like
            The data representing the covariance matrix. If 1D, the input
            is interpreted as the diagonal elements, with all other elements
            zero
        ks : array_like
            The wavenumbers where the measurements are defined. For every `mu`
            measurement, it is assumed that the measurement is defined at
            the same k values. Units are given by the `units` keyword
        ells : array_like
            The ell values where the measurements are defined
        """
        # set the defaults for keywords
        for k in self._allowed_extra_info:
            extra_info.setdefault(k, None)
            
        super(PoleCovarianceMatrix, self).__init__('ell', data, ks, ells, units, h, **extra_info)
        
        # check that all ell submatrices have same k length
        ks = self.index.get_level_values('k')
        if len(ks) != len(ks.unique())*len(ells):
            raise ValueError("All `k` arrays must be the same for `PoleCovarianceMatrix`")

    #---------------------------------------------------------------------------
    def _fill_missing_data(self, frame):
        """
        Fill missing data by interpolating a spline
        """
        from scipy.interpolate import UnivariateSpline as spline

        # loop over each k bin
        ks = frame.index.get_level_values('k').unique()
        for i, k in enumerate(ks):

            x = frame.xs(k, level='k', drop_level=False)
            inds = x.mean_power.isnull()
            y = x.dropna()

            # now fill with spline values
            if inds.sum() > 0:
                s = spline(y.index.get_level_values('mu'), y.mean_power, w=1./y.variance**0.5)
                mus = x.index.get_level_values('mu')[inds]
                frame.loc[x.index[inds], 'mean_power'] = s(mus)

        return frame
        
    #---------------------------------------------------------------------------
    def _store_extra_info(self, **kwargs):
        """
        Store the extra information needed to compute analytic variances
        """
        from scipy.integrate import quad
        from scipy.special import legendre
        import itertools
        
        if not all(k in kwargs for k in ['mus', 'pkmu_power', 'modes', 'pkmu_variance']):
            self.extra_info = None
            return
            
        # the k, mu info
        ks = self.index.get_level_values('k').unique()
        mus = kwargs['mus']
        Nmu = len(mus)
        dmu = np.diff(mus)[0]
        
        # make the pkmu index and dataframe
        pkmu_index = MultiIndex.from_tuples(list(itertools.product(mus, ks)), names=['mu', 'k'])
        d = {'mean_power':kwargs['pkmu_power'], 'variance':kwargs['pkmu_variance'], 'modes':kwargs['modes']}
        frame = DataFrame(data=d, index=pkmu_index)
        
        # fill missing data with spline
        frame = self._fill_missing_data(frame)
        
        index = []
        total_covariance = []
        total_modes = []
        
        # sum the number of modes for each k bin 
        modes = frame.groupby(level='k').modes.sum()
        
        # do ell, ell' combinations
        for ell in self.ells:
            for ell_prime in self.ells:
                
                if ell_prime > ell: 
                    continue
                
                covariance = ks*0.
                for i, mu in enumerate(mus):   
    
                    # the integral over the legendre polynomial at this mu
                    mu_integral = quad(lambda x: legendre(ell)(x)*legendre(ell_prime)(x), mu-0.5*dmu, mu+0.5*dmu)[0]
                    
                    # slice at this mu
                    this_mu = frame.xs(mu, level='mu')
                    power = np.nan_to_num(this_mu.mean_power)**2
                    covariance += mu_integral * power
                    
                # normalize properly
                covariance *= (2*ell + 1)*(2*ell_prime + 1)
                
                # store the total
                total_covariance += list(covariance)
                total_modes += list(modes)
            
                # keep track of the index                 
                index += [(ell, ell_prime, ki) for ki in ks]
        
        index = MultiIndex.from_tuples(index, names=['ell', 'ell_bar', 'k'])
        d = {'mean_power_sq':total_covariance, 'modes':total_modes}
        self.extra_info = DataFrame(data=d, index=index)
    
    #---------------------------------------------------------------------------
    def _update_extra_info_units(self, units, h):
        """
        Update the units store in the `extra_info` DataFrame
        """
        if self.extra_info is not None:

            # update 'k' levels 
            new_levels = []
            for i, name in enumerate(self.extra_info.index.names):
                factor = 1.
                if name == 'k':
                    factor = self._k_conversion_factor(units, h)
                new_levels.append(self.extra_info.index.levels[i]*factor)
            self.extra_info.index = self.extra_info.index.set_levels(new_levels)

            # multiply by the power values
            if hasattr(self.extra_info, 'mean_power_sq'):
                self.extra_info['mean_power_sq'] *= self._power_conversion_factor(units, h)**2
        
    #---------------------------------------------------------------------------
    @property
    def ells(self):
        """
        The `ell` values associated with the matrix; only unique `ell` values
        are returned
        """
        return self._xs
        
    #---------------------------------------------------------------------------
    def ell_slice(self, ell, ell2=None):
        """
        Return the sub matrix associated with the specified `ell` value
        """
        return self._x_slice(ell, ell2)
    
    #---------------------------------------------------------------------------
    def trim_ell(self, lower=None, upper=None):
        """
        Trim the covariance matrix to specified minimum/maximum ell values
        """
        return self._trim_x(lower=lower, upper=upper)
        
    #---------------------------------------------------------------------------
    # extra info
    #---------------------------------------------------------------------------
    def modes(self, k=None, ell=None, ell_bar=None):
        """
        The number of modes as a function of `ell`, 'ell_bar` and `k`
        """
        if hasattr(self.extra_info, 'modes'):
            if k is None and ell is None:
                return self.extra_info.modes
            else:
                # slice at a certain ell
                if ell is None:
                    raise ValueError('Please specify `ell` value to get number of modes at given k')                    
                if isinstance(ell, int): 
                    ell = self._convert_x_integer(ell)
                
                # slice at ell_bar
                if ell_bar is None:
                    ell_bar = ell 
                if isinstance(ell_bar, int): 
                    ell_bar = self._convert_x_integer(ell_bar)
                
                # reverse the order if ell_bar > ell
                if ell_bar > ell:
                    ell, ell_bar = ell_bar, ell
                        
                # do the slicing
                modes = self.extra_info.modes.xs(ell, level='ell')
                modes = modes.xs(ell_bar, level='ell_bar')

                # slice at specific k
                if k is not None:
                    if isinstance(k, int):
                        k = self._convert_k_integer(k, ell)
                    modes = modes.loc[k]
                return modes
        else:
            return None

    #---------------------------------------------------------------------------
    def mean_power_sq(self, k=None, ell=None, ell_bar=None):
        """
        The mean power squared (that enters into the covariance) as a function 
        of `ell`, 'ell_bar` and `k`, in the units specified by `self.h` 
        and `self.units`

        NOTE: this has the shot noise included in it
        """
        if hasattr(self.extra_info, 'mean_power_sq'):
            if k is None and ell is None:
                return self.extra_info.mean_power_sq

            else:    
                # slice at a certain ell
                if ell is None:
                    raise ValueError('Please specify `ell` value to get `mean_power_sq` at given k')                    
                if isinstance(ell, int): 
                    ell = self._convert_x_integer(ell)
            
                # slice at ell_bar
                if ell_bar is None:
                    ell_bar = ell 
                if isinstance(ell_bar, int): 
                    ell_bar = self._convert_x_integer(ell_bar)
        
                # reverse the order if ell_bar > ell
                if ell_bar > ell:
                    ell, ell_bar = ell_bar, ell
        
                # do the slicing
                Psq = self.extra_info.mean_power_sq.xs(ell, level='ell')
                Psq = Psq.xs(ell_bar, level='ell_bar')

                # slice at specific k
                if k is not None:
                    if isinstance(k, int):
                        k = self._convert_k_integer(k, ell)
                    Psq = Psq.loc[k]
                return Psq
        else:
            return None

    #---------------------------------------------------------------------------
    def gaussian_covariance(self, k=None, ell=None, ell_bar=None):
        """
        Return the Gaussian prediction for the variance of the diagonal 
        elements, given by: 

        :math: 2 / N_{modes} * (2*\ell+1)*(2\bar{\ell}+1) 
                \int d\mu L_\ell(\mu) L_\bar{ell} [ P(k,\mu) + P_{shot}(k,\mu) ]**2
        """
        if k is not None and ell is None:
            msg = "`ell` must also be specified to compute Gaussian error at specific `k`"
            raise ValueError(msg)

        modes = self.modes(k=k, ell=ell, ell_bar=ell_bar)
        if modes is None:
            raise ValueError("Cannot compute Gaussian covariance without `modes`")

        mean_power_sq = self.mean_power_sq(k=k, ell=ell, ell_bar=ell_bar)
        if mean_power_sq is None:
            raise ValueError("Cannot compute Gaussian covariance without `mean_power_sq`")
 
        return 2./modes * mean_power_sq
    
    #---------------------------------------------------------------------------
    # plotting
    #---------------------------------------------------------------------------
    def plot_diagonals(self, ell, ell_bar=None, ax=None, **options):
        """
        Plot the square-root of the diagonal elements for a specfic ell value
        
        Parameters
        ---------------
        ell : int 
            If not `None`, only plot the diagonals for a single `ell` value, 
            else plot for all `ell` values
        ell_bar : int, optional
            If not `None`, plot the diagonal elements of the `ell` x `ell_bar`
            sub matrix, else plot those for the `ell` x `ell` sub matrix
        ax : plotify.Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        options : dict
            Dictionary of options with the following keywords:            
                show_gaussian : bool
                    Overplot the Gaussian prediction, i.e., 
                    :math: 2/N_{modes} * [P(k) + Pshot(k)]**2
                norm_by_gaussian : bool
                    Normalize the diagonal elements by the Gaussian prediction
                norm_by_power : bool
                    Normalize the diagonal elements by the mean power, 
                    P(k) + Pshot(k)
                subtract_gaussian : bool
                    Subtract out the Gaussian prediction
                absmag : bool
                    Plot the absolute magnitude, relevant for cross multipoles
        """     
        # get the current axes
        if ax is None: ax = pfy.gca()
        
        # setup the default options
        options.setdefault('show_gaussian', False)
        options.setdefault('norm_by_gaussian', False)
        options.setdefault('norm_by_power', False)
        options.setdefault('subtract_gaussian', False)
        options.setdefault('absmag', False)
        
        # determine the ells to plot
        if isinstance(ell, int): 
            ell = self._convert_x_integer(ell)
        if ell_bar is not None:
            ell_bar = self._convert_x_integer(ell_bar)
        else:
            ell_bar = ell    
        
        if options['show_gaussian']:
            ax.color_cycle = "Paired"
 
        # get sigma for this ell sub matrix
        this_slice = self.ell_slice(ell, ell_bar)
        sigma_sq = np.diag(this_slice)
        
        norm = 1.
        if options['subtract_gaussian']:
            norm = sigma.copy()
            sigma_sq -= self.gaussian_covariance(ell=ell, ell_bar=ell_bar)
        elif options['norm_by_gaussian']:
            norm = self.gaussian_covariance(ell=ell, ell_bar=ell_bar)
        elif options['norm_by_power']:
            norm = self.mean_power_sq(ell=ell, ell_bar=ell_bar)
            
        # do the plotting
        if ell == ell_bar:
            label = r"$\ell = \bar{{\ell}} = {}$".format(ell)
        else:
            label = r"$\ell = {}, \ \bar{{\ell}} = {}$".format(ell, ell_bar)
        
        toplot = sigma_sq/norm
        if options['absmag']: toplot = abs(toplot)
        pfy.plot(self.ks(ell=ell), toplot, label=label)
        if options['show_gaussian']:
            
            g = self.gaussian_covariance(ell=ell, ell_bar=ell_bar)/norm
            if options['absmag']: g = abs(g)
            pfy.plot(self.ks(ell=ell), g, ls='--')
            
        if np.isscalar(norm):
            ax.x_log_scale()
            ax.y_log_scale()
            
        # add the legend and axis labels    
        ax.legend(loc=0, ncol=2)
        if self.units == 'relative':
            k_units = r"($h$/Mpc)"
            P_units = r"$(\mathrm{Mpc}/h)^6$"
        else:
            k_units = "(1/Mpc)"
            P_units = r"$(\mathrm{Mpc})^6$"
            
        ax.xlabel.update(r'$k$ %s' %k_units, fontsize=16)
        if options['norm_by_power']:
            ax.ylabel.update(r"$\sigma_P^2 / (P_\ell(k) P_\bar{\ell}(k))$", fontsize=16)

        elif options['norm_by_gaussian']:
            ax.ylabel.update(r"$\sigma_P^2 / \sigma_{P, \mathrm{Gaussian}}^2$", fontsize=16)
        else:
            if not options['subtract_gaussian']:
                if options['absmag']:
                    ax.ylabel.update(r"$|\sigma_P^2|$ %s" %P_units, fontsize=16)
            else:
                ax.ylabel.update(r"$(\sigma^2_P - \sigma^2_{P, \mathrm{Gaussian}}) / \sigma_P^2$", fontsize=16)

        return ax
        
    #---------------------------------------------------------------------------
    def plot_off_diagonals(self, kbar, ell, ell_bar=None, ax=None, **options):
        """
        Plot the off diagonal elements, specifically:
        
        :math: \sqrt(Cov(k, kbar) / P(k, mu) / P(kbar, mu))
        
        Parameters
        ---------------
        kbar : {float, int}
            The specific value of k to plot the covariances at
        ell : int 
            If not `None`, only plot the diagonals for a single `ell` value, 
            else plot for all `ell` values
        ell_bar : int, optional
            If not `None`, plot the diagonal elements of the `ell` x `ell_bar`
            sub matrix, else plot those for the `ell` x `ell` sub matrix
        ax : plotify.Axes, optional
            Axes instance to plot to. If `None`, plot to the current axes
        options : dict
            Dictionary of options with the following keywords:            
                show_diagonal : bool
                    If `True`, plot the diagonal value as a stem plot
                show_binned : bool
                    If `True`, plot a smoothed spline interpolation
                show_zero_line : bool
                    If `True`, plot a y=0 horizontal line
        """
        import scipy.stats
                
        # get the current axes
        if ax is None: ax = pfy.gca()
        
        # setup the default options
        options.setdefault('show_diagonal', True)
        options.setdefault('show_binned', False)
        options.setdefault('show_zero_line', False)
        
        # determine the ells to plot
        if isinstance(ell, int): 
            ell = self._convert_x_integer(ell)
        if ell_bar is not None:
            ell_bar = self._convert_x_integer(ell_bar)
        else:
            ell_bar = ell
        
        # get the k units
        if self.units == 'relative':
            k_units = r"$h$/Mpc"
        else:
            k_units = "1/Mpc"
                
        # ks for this ell
        ks = self.ks(ell=ell)
        kbars = self.ks(ell=ell_bar)
        
        # get the right kbar
        if isinstance(kbar, int):
            kbar = self._convert_k_integer(kbar, ell_bar)
        
        # get the closest value in the matrix
        if kbar not in kbars:
            kbar = kbars[abs(kbars - kbar).argmin()]
                
        # get sigma for this mu sub matrix
        this_slice = self.ell_slice(ell, ell_bar)
        cov = this_slice.xs(kbar, axis=1)
        
        # the normalization
        P_k = self.mean_power_sq(ell=ell, ell_bar=ell_bar)**0.5
        P_kbar = self.mean_power_sq(ell=ell, ell_bar=ell_bar, k=kbar)**0.5
        norm = P_k*P_kbar
        
        # remove the diagonal element
        toplot = cov/norm
        diag_element = toplot[kbar]
        toplot = toplot.drop(kbar)
        ks = ks[ks != kbar]
        
        # plot the fractional covariance
        if ell == ell_bar:
            args = (ell, kbar, k_units)
            label = r"$\ell = \bar{{\ell}} = {0}, \ \bar{{k}} = {1:.3f}$ {2}".format(*args)
        else:
            args = (ell, ell_bar, kbar, k_units)
            label = r"$\ell = {0}, \ \bar{{\ell}} = {1}, \ \bar{{k}} = {2:.3f}$ {3}".format(*args)
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
        if self.units == 'relative':
            ax.xlabel.update(r'$k$ ($h$/Mpc)', fontsize=16)
        else:
            ax.xlabel.update(r'$k$ (1/Mpc)', fontsize=16)
        ax.ylabel.update(r"$\mathrm{Cov}(k, \bar{k}) / (P_\ell(k) P_\bar{\ell}(\bar{k}))$", fontsize=16)
        
        return ax
    #---------------------------------------------------------------------------
#endclass PoleCovarianceMatrix

#-------------------------------------------------------------------------------