import pyRSD.extern.xarray as xr
import numpy as np
import pandas as pd
from six import string_types

def slice_data(data, indexers, indexes, return_indices=False):
    """
    Slice the input data according to the integer
    indexing specified by the dictionary ``indexers``

    Parameters
    ----------
    data : np.ndarray
        the data that will be sliced
    indexers : dict
        a dictionary specifiying the slicing, where keys
        are dimension names and values are the desired indexing
    indexes : list
        list of GridIndex objects, which define the coordinates
        and dimension names
    return_indices : list of array_like
        If `True`, also return the indices used to slice each dimension

    """
    ndim = np.ndim(data)
    shape = np.shape(data)
    key = [None] * ndim

    # loop over each axis and corresponding index
    for axis, index in enumerate(indexes):

        # get the set of indexers for this index
        inds = []
        for dim in index:
            if dim in indexers:
                v = indexers[dim]
                # expand out slice into element list
                if isinstance(v, slice):
                    v = range(*v.indices(shape[axis]))
                # if boolean index array, convert to element array
                elif not np.isscalar(v):
                    if np.array(v).dtype == bool:
                        v = np.where(v)[0]
                inds.append(v)

        # skip this axis, if no indexing provided
        if not len(inds): continue

        # take the intersection of all dimensions per index
        if np.isscalar(inds[0]):
            valid = inds[0]
        else:
            valid = sorted(set(inds[0]).intersection(*inds))
            if len(inds) > 1 and not len(valid):
                raise ValueError("non-overlapping indices requested for index #%d with dims %s" %(i, index.dims))
        key[axis] = valid

    # now use np.take to slice each axis
    for axis, indices in reversed(list(enumerate(key))):
        if indices is None: continue
        data = np.take(data, indices, axis=axis)
    if not return_indices:
        return data
    else:
        return data, key

def remap_label_indexers(indexes, indexers):
    """
    Map the specified index values from label-based to integer-based
    """
    toret = {}
    for i, index in enumerate(indexes):

        if index.ndim > 1:
            label = [slice(None)]*index.ndim
            for idim, dim in enumerate(index.dims):
                if dim not in indexers: continue
                v = indexers[dim]
                if not isinstance(v, slice):
                    _, v = index.get_nearest(dim, v)
                label[idim] = v

            mi = index.to_pandas()
            if mi.lexsort_depth != index.ndim:
                raise ValueError("lexsort depth too small for MultiIndex -- likely there are no repeating index values")
            toret[index.dims[0]] = mi.get_locs(tuple(label))
        else:
            dim = index.dims[0]
            if dim in indexers:
                label = indexers[dim]
                if not isinstance(label, slice):
                    _, label = index.get_nearest(dim, label)
                toret[dim] = xr.indexing.convert_label_indexer(index.to_pandas(dim), label, dim, None)

    return toret

class GridIndex(object):
    """
    Class to store the coordinates for a given dimension of a data array,
    optionally associating more than one dimension/coordinate per
    data array axis
    """
    def __init__(self, dims, coords):
        """
        Parameters
        ----------
        dims : str, list of str
            list of the names of each dimension
        coords  : array_like, list of array_like
            list of the arrays specifiying the coordinates for each axis
        """
        self.dims = dims
        if np.isscalar(self.dims):
            self.dims = [self.dims]
            coords = [coords]
        self.dims = list(self.dims)
        self.coords = dict(zip(self.dims, coords))

        if not all(len(self.coords[d]) == self.size for d in self.dims):
            raise ValueError("not all index coordinates have the same dimension")

    @property
    def size(self):
        """
        Length of coordinate grid for this index (if multiple coords, all
        should be the same)
        """
        return len(self.coords[self.dims[0]])

    @property
    def ndim(self):
        """
        The number of coordinate dimensions associated with this index
        """
        return len(self.dims)

    def __repr__(self):
        name = str(self.__class__).split('.')[-1].split("'")[0]
        return "<{name}: dimensions {dims}>".format(name=name, dims=self.dims)

    def __str__(self):
        return str(self.__repr__())

    def __contains__(self, key):
        """
        Can use `in` keyword to test the dimension names for the index
        """
        return key in self.dims

    def __iter__(self):
        """
        Iterate over the dimension names
        """
        for key in self.dims:
            yield key

    def is_unique(self, key=None):
        """
        Do all dimensions associated with this index have unique coordinates
        """
        if key is not None:
            return self.to_pandas(key).is_unique
        else:
            return all(self.to_pandas(d).is_unique for d in self.dims)

    def to_pandas(self, key=None, unique=False):
        """
        If `key` is specified, convert the coordinate values to a `pandas.Index`.
        If `ndim` is greater than one, return a `pandas.MultiIndex` for all
        dimensions of the index
        """
        if unique:
            return self.to_pandas(key=key, unique=False).unique()

        if key is None and self.ndim > 1:
            indexes = [self.to_pandas(d) for d in self.dims]
            names = [i.name for i in indexes]
            return pd.MultiIndex.from_tuples(list(zip(*indexes)), names=names)
        if key is None: key = self.dims[0]

        return pd.Index(self.coords[key], name=key)

    def get_nearest(self, dim, value):
        """
        Return the nearest value in the index for dimension `dim`
        to the desired `value`

        Parameters
        ----------
        dim : str
            string specifying the dimension name
        value : float
            value to find the nearest match to

        Returns
        -------
        idx : int
            the integer index specifying the nearest element
        nearest : float
            the nearest element value
        """
        index = pd.Index(self.to_pandas(dim).unique())

        # the value should be a list
        if np.isscalar(value):
            value = [value]

        ii = index.get_indexer(value, method='nearest')
        return ii, index[ii]


class GridIndexer(object):
    """
    Class to facilitate indexing and slicing of data defined on a grid, i.e.,
    a (k, mu) or (k, ell) grid.

    The main functions are the ``sel`` and ``isel`` functions which return
    the speicifed indices for slicing a data array

    Notes
    -----
    *   setup is similar to ``xarray`` package, with dimension names defined
        for each axis of the data, as well as the corresponding coordinate
        grid
    *   we can also define multiple coordinates per axis element, which
        becomes useful for i.e., covariance matrix manipulations, where
        each element is defined at a (k1, mu1) and (k2, mu2)
    *   ``isel`` selects based on integer indexing and ``sel`` selects
        on coordinate value (always selecting the nearest value on the grid)
    """
    def __init__(self, dims, coords):
        """
        Parameters
        ----------
        dims : list
            list of the names of each dimension
        coords  : list
            list of the arrays specifiying the coordinates for each axis
        """
        self.dims = dims
        if len(set(self.dims_flat)) != len(self.dims_flat):
            raise ValueError("please use unique dimension names")

        # index is a list of GridIndex objects for each axis
        self.index = []
        for i, dim in enumerate(dims):
            if not isinstance(dim, string_types + (tuple, list)):
                raise TypeError("each dimension must be specified by str or tuple/list of str")
            self.index.append(GridIndex(dim, coords[i]))

        # coords is a list of dictionaries holding (dim, coord) pairs for each axis
        self.coords = [index.coords for index in self.index]

    @property
    def dims_flat(self):
        """
        A flattened list of dimensions, for convenience
        """
        return np.ravel(self.dims)

    @property
    def ndim(self):
        """
        The number of dimensions
        """
        return len(self.dims)

    def isel(self, data, return_indices=False, **indexers):
        """
        Return a new numpy.ndarray that has been sliced by the integer indexing
        along the specified dimension(s).
        """
        invalid = [k for k in indexers if not k in self.dims_flat]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        return slice_data(data, indexers, self.index, return_indices=return_indices)

    def sel(self, data, return_indices=False, **indexers):
        """
        Return a new numpy.ndarray that has been sliced by the label indexing
        along the specified dimension(s).
        """
        invalid = [k for k in indexers if not k in self.dims_flat]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        return self.isel(data, return_indices=return_indices, **remap_label_indexers(self.index, indexers))

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
        for index in self.index:
            if dim in index:
                return index.get_nearest(dim, coord)

        # if we get here, we have failed
        raise ValueError("dimension name `%s` should be one of %s" %(dim, self.dims))

    def is_unique(self, axis, key=None):
        """
        Is the specified axis have unique coordinate values?
        """
        index = self.index[axis]
        if key is not None:
            return index.to_pandas(key).is_unique
        else:
            return all(index.to_pandas(d).is_unique for d in index.dims)
