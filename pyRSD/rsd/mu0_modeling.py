from .. import numpy as np, data as sim_data
import cPickle
import itertools
import bisect

def align_domain(*args, **kwargs):
    """
    Align the input domain values
    """
    ndims = kwargs.get('ndims', 1)
    args = list(args)
    if len(args) == 1:
        if np.shape(args[0])[1] == ndims:
            return args[0]
    elif len(args) != ndims:
        raise ValueError("mismatch between number of input arrays and number of dimensions")
        
    nonscalar_idx = []
    for i, val in enumerate(args):
        if not np.isscalar(val):
            nonscalar_idx.append(i)
    
    if len(nonscalar_idx) != len(args):
        scalar_idx = range(ndims)
        for i in reversed(nonscalar_idx): 
            del scalar_idx[i]
        for i in scalar_idx:
            args[i] = [args[i]]  
    pts = np.asarray(list(itertools.product(*args)))
    return pts, map(len, args)
    
def unpack(a):
    if isinstance(a, tuple):
        return map(unpack, a)
    else:
        try:
            return a[0] if not len(a)-1 else a
        except:
            return a

def reshape(a, dims):
    if isinstance(a, tuple):
        return [reshape(x, dims) for x in a]
    else:
        return np.squeeze(a.reshape(dims))
    
class GPModelParams(object):
    
    def __init__(self, path):
        self.data = cPickle.load(open(path, 'r'))
        self.param_names = self.data.keys()
        self.ndims = self.data[self.param_names[0]].X.shape[-1]
        
    def __call__(self, *args, **kwargs):
        col = kwargs.get('col', None)
        return_err = kwargs.get('return_err', False)

        pts, dims = align_domain(*args, ndims=self.ndims)
        if col is not None:
            return unpack(reshape(self.data[col].predict(pts, eval_MSE=return_err), dims))
        else:
            toret = {}
            for p in self.param_names:
                ans = unpack(reshape(self.data[p].predict(pts, eval_MSE=return_err), dims))
                toret[p] = ans
            return toret
            
    def to_dict(self, *args, **kwargs):
        data = self.__call__(*args, **kwargs)
        keys = data.keys()
        data = np.concatenate([data[k][...,None] for k in keys],axis=-1)
        shape = data.shape[:-1]
        if not len(shape):
            return dict(zip(keys, data))
        toret = np.empty(shape, dtype=object)
        for sl in np.ndindex(*shape):
            toret[sl] = dict(zip(self.param_names, data[sl]))
        return toret
        
class SplineTableModelParams(object):
    
    def __init__(self, path):
        self.data = cPickle.load(open(path, 'r'))
        self.index = np.array(sorted(self.data.keys()))
        self.index_min = np.amin(self.index)
        self.index_max = np.amax(self.index)
        
        self.param_names = self.data[self.index[0]].keys()
        self.ndims = 2 # by default
        
    def __call__(self, *args, **kwargs):
        col = kwargs.get('col', None)
        x, y = args
        
        # return NaNs if we are out of bounds
        if x < self.index_min or x > self.index_max:
            args = (x, self.index_min, self.index_max)
            raise ValueError("index value {} is out of interpolation range [{}, {}]".format(*args))

        if x in self.index:
            i = abs(self.index - x).argmin()
            x_lo = x_hi = self.index[i]
            w = 0.
        else:
            ihi = bisect.bisect(self.index, x)
            ilo = ihi - 1

            x_lo = self.index[ilo]
            x_hi = self.index[ihi]
            w = (x - x_lo) / (x_hi - x_lo) 
        
        if col is None:
            toret = {}
            for col in self.param_names:
                val_lo = (1 - w)*self.data[x_lo][col](y) 
                val_hi = w*self.data[x_hi][col](y)
                toret[col] = (val_lo + val_hi)
            return toret
        else:
            val_lo = (1 - w)*self.data[x_lo][col](y) 
            val_hi = w*self.data[x_hi][col](y)
            return val_lo + val_hi
            
    def to_dict(self, *args, **kwargs):
        data = self.__call__(*args, **kwargs)
        keys = data.keys()
        data = np.concatenate([data[k][...,None] for k in keys],axis=-1)
        shape = data.shape[:-1]
        if not len(shape):
            return dict(zip(keys, data))
        toret = np.empty(shape, dtype=object)
        for sl in np.ndindex(*shape):
            toret[sl] = dict(zip(self.param_names, data[sl]))
        return toret
        
#------------------------------------------------------------------------------
# Models
#------------------------------------------------------------------------------
class StochasticityPadeModelParams(GPModelParams):
    """
    The bestfit params for the (type B) stochasticity, modeled using a Pade expansion,
    and interpolated using a Gaussian process as a function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.stochB_pade_params()
        super(StochasticityPadeModelParams, self).__init__(path)

class StochasticityLogModelParams(SplineTableModelParams):
    """
    The bestfit params for the (type B) stochasticity, modeled using a Pade expansion,
    and interpolated using a spline table as a function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.stochB_log_params()
        super(StochasticityLogModelParams, self).__init__(path)

class CrossStochasticityLogModelParams(GPModelParams):
    """
    The bestfit params for the (type B) cross bin stochasticity, modeled using 
    a Pade expansion, and interpolated using a Gaussian process as a function 
    of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.stochB_cross_log_params()
        super(CrossStochasticityLogModelParams, self).__init__(path)
    
class PhmResidualModelParams(GPModelParams):
    """
    The model for the Phm residual, Phm - b1*Pzel, modeled using a Pade expansion,
    and interpolated using a Gaussian process as a function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.Phm_residual_params()
        super(PhmResidualModelParams, self).__init__(path)
        
class PhmCorrectedPTModelParams(GPModelParams):
    """
    The model for the corrected PT model for Phm, modeled using a linear
    slope at high k, and interpolated using a Gaussian process as a function 
    of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.Phm_correctedPT_params()
        super(PhmCorrectedPTModelParams, self).__init__(path)
        
class PhhModelParams(GPModelParams):
    """
    The model for Phh, interpolated using a Gaussian process as a 
    function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.Phh_params()
        super(PhhModelParams, self).__init__(path)
        

        
        