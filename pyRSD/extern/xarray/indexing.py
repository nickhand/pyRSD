import numpy as np
import pandas as pd

def to_0d_object_array(value):
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((1,), dtype=object)
    result[:] = [value]
    result.shape = ()
    return result

def is_dict_like(value):
    return hasattr(value, '__getitem__') and hasattr(value, 'keys')
    
def _try_get_item(x):
    try:
        return x.item()
    except AttributeError:
        return x

def _asarray_tuplesafe(values):
    """
    Convert values into a numpy array of at most 1-dimension, while preserving
    tuples.

    Adapted from pandas.core.common._asarray_tuplesafe
    """
    if isinstance(values, tuple):
        result = to_0d_object_array(values)
    else:
        result = np.asarray(values)
        if result.ndim == 2:
            result = np.empty(len(values), dtype=object)
            result[:] = values

    return result


def _is_nested_tuple(possible_tuple):
    return (isinstance(possible_tuple, tuple)
            and any(isinstance(value, (tuple, list, slice))
                    for value in possible_tuple))


def convert_label_indexer(index, label, index_name='', method=None,
                          tolerance=None):
    """Given a pandas.Index and labels (e.g., from __getitem__) for one
    dimension, return an indexer suitable for indexing an ndarray along that
    dimension. If `index` is a pandas.MultiIndex and depending on `label`,
    return a new pandas.Index or pandas.MultiIndex (otherwise return None).
    """
    # backwards compatibility for pandas<0.16 (method) or pandas<0.17
    # (tolerance)
    kwargs = {}
    if method is not None:
        kwargs['method'] = method
    if tolerance is not None:
        if pd.__version__ < '0.17':
            raise NotImplementedError(
                'the tolerance argument requires pandas v0.17 or newer')
        kwargs['tolerance'] = tolerance

    new_index = None

    if isinstance(label, slice):
        if method is not None or tolerance is not None:
            raise NotImplementedError(
                'cannot use ``method`` argument if any indexers are '
                'slice objects')
        indexer = index.slice_indexer(_try_get_item(label.start),
                                      _try_get_item(label.stop),
                                      _try_get_item(label.step))
        if not isinstance(indexer, slice):
            # unlike pandas, in xarray we never want to silently convert a slice
            # indexer into an array indexer
            raise KeyError('cannot represent labeled-based slice indexer for '
                           'dimension %r with a slice over integer positions; '
                           'the index is unsorted or non-unique')

    elif is_dict_like(label):
        is_nested_vals = _is_nested_tuple(tuple(label.values()))
        if not isinstance(index, pd.MultiIndex):
            raise ValueError('cannot use a dict-like object for selection on a '
                             'dimension that does not have a MultiIndex')
        elif len(label) == index.nlevels and not is_nested_vals:
            indexer = index.get_loc(tuple((label[k] for k in index.names)))
        else:
            indexer, new_index = index.get_loc_level(tuple(label.values()),
                                                     level=tuple(label.keys()))

    elif isinstance(label, tuple) and isinstance(index, pd.MultiIndex):
        if _is_nested_tuple(label):
            indexer = index.get_locs(label)
        elif len(label) == index.nlevels:
            indexer = index.get_loc(label)
        else:
            indexer, new_index = index.get_loc_level(
                label, level=list(range(len(label)))
            )

    else:
        label = _asarray_tuplesafe(label)
        if label.ndim == 0:
            if isinstance(index, pd.MultiIndex):
                indexer, new_index = index.get_loc_level(label.item(), level=0)
            else:
                indexer = index.get_loc(label.item(), **kwargs)
        elif label.dtype.kind == 'b':
            indexer, = np.nonzero(label)
        else:
            indexer = index.get_indexer(label, **kwargs)
            if np.any(indexer < 0):
                raise KeyError('not all values found in index %r'
                               % index_name)
    return indexer, new_index