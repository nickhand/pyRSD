from ... import numpy as np
import copy
from pandas import DataFrame, Index

#-------------------------------------------------------------------------------
class CovarianceMatrix(object):
    """
    Class to represent a covariance matrix
    """
    def __init__(self, data, index=None):
        
        # make local copies
        data = np.asarray(data).copy()
        if index is not None:
            index = np.asarray(index).copy()
        
        # make a diagonal 2D matrix from input
        if data.ndim == 1:
            data = np.diag(data)
            
        # check basic properties
        self.check_properties(data)
        
        # store the size
        self.N = np.shape(data)[0]
        
        # check the index
        self._duplicate_index = False
        if index is not None:
            if np.shape(index)[0] != self.N:
                raise ValueError("Shape mismatch between data and index for covariance matrix")
            if not np.all(index[:-1] <= index[1:]):
                self._duplicate_index = True
                unique = np.unique(index)
                N = len(index) / len(unique)
                if not np.array_equal(np.concatenate((unique,)*N), index):
                    raise ValueError("Matrix index not understood; proceeding is probably a bad idea")
        
        self._initialize_backend(data, index)
    
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        return repr(self.full())
    
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
            
        # check to make sure the key is correct
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Must specify both an `i` and `j` index")

        i, j = key  
        
        # check for ints in the right instance
        for x in [i, j]:
            if not isinstance(x, int):
                if self.index is None:
                    raise ValueError("Key must be an integer if no index is provided")
                if self._duplicate_index:
                    raise ValueError("Key must be an integer if index contains duplicate keys")
        
        # if index with duplicate keys, return based on 2D frame iloc   
        if self._duplicate_index:
            return self.full().iloc[i,j]
        
        # if not duplicate index, just use the keys
        if self.index is not None:      
            if isinstance(i, int): i = self.index[i]
            if isinstance(j, int): j = self.index[j]

        if not (i, j) in self.data.index:
            if not (j, i) in self.data.index:
                raise KeyError("Sorry, element (%s, %s) does not exist" %(i, j))
            else:
                i, j = j, i

        return self.data.loc[i, j]
        
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
        # setup the indices we need
        self.index = index
        if index is None:
            index = range(self.N)
        index_i = Index(index, name='i')
        index_j = Index(index, name='j')
        
        # now make the frame
        frame = DataFrame(data, index=index_j, columns=index_i)
        
        # and make it into a series
        frame.values[np.triu_indices_from(frame, k=1)] = np.nan
        self.data = frame.unstack().dropna()
        
    #---------------------------------------------------------------------------
    def check_properties(self, data):
        
        if not np.array_equal(data, data.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        if not np.all(np.linalg.eigvals(data) >= 0.):
            raise ValueError("Covariance matrix is not positive-semidefinite")
    
    #---------------------------------------------------------------------------
    @property
    def shape(self):
        return (self.N, self.N)
    
    #---------------------------------------------------------------------------
    def diag(self):
        """
        Return the diagonal elements
        """
        try:
            return self._diag
        except AttributeError:
            self._diag = np.array([self[i] for i in range(self.N)])
            return self._diag
    
    #---------------------------------------------------------------------------
    def full(self):
        """
        Return the full 2D covariance matrix array
        """
        frame = self.data.unstack()
        i, j = np.triu_indices_from(frame, k=1)
        frame.values[j, i] = frame.values[i, j]
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
        diag = self.diag()
        norm = np.array([np.sqrt(diag[i]*diag[j]) for (i, j) in zip(ii, jj)])
        toret.data /= norm
        return toret
        
    #---------------------------------------------------------------------------
    def plot(self, filename=None):
        """
        Plot the correlation matrix (normalized covariance matrix), optionally
        saving if `filename != None`
        """
        import plotify as pfy
        
        pfy.clf()
        corr = self.normalized().asarray()
        colormesh = pfy.pcolormesh(corr, vmin=-1, vmax=1)
        pfy.colorbar(colormesh)
        if filename is not None:
            pfy.savefig(filename)
    
        return colormesh
    #---------------------------------------------------------------------------
    def trim_by_index(self, val):
        """
        Trim the covariance matrix to specified maximum value using the index 
        """
        # get the index values that are less than the max value
        inds = [(ki <= val and kj <= val) for ki, kj in self.data.index.values]
        sliced = self.data[inds].unstack()
        i_hi, j_hi = np.triu_indices_from(sliced, k=1)
        sliced.values[j_hi, i_hi] = sliced.values[i_hi, j_hi]
        
        new_index = self.index[np.where(self.index <= val)]
        return CovarianceMatrix(np.array(sliced), index=new_index)
        
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
            
    #---------------------------------------------------------------------------
#endclass CovarianceMatrix

#-------------------------------------------------------------------------------
        
        