from .. import numpy as np, data as sim_data
import cPickle
import itertools

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
        data = np.concatenate([data[k][...,None] for k in self.param_names],axis=-1)
        shape = data.shape[:-1]
        if not len(shape):
            return dict(zip(self.param_names, data))
        toret = np.empty(shape, dtype=object)
        for sl in np.ndindex(*shape):
            toret[sl] = dict(zip(self.param_names, data[sl]))
        return toret
        
#------------------------------------------------------------------------------
# Models
#------------------------------------------------------------------------------
class StochasticityModelParams(GPModelParams):
    """
    The bestfit params for the (type B) stochasticity, modeled using a Pade expansion,
    and interpolated using a Gaussian process as a function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.stochB_gp_params()
        super(StochasticityModelParams, self).__init__(path)
        
class PhmResidualModelParams(GPModelParams):
    """
    The model for the Phm residual, Phm - b1*Pzel, modeled using a Pade expansion,
    and interpolated using a Gaussian process as a function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.Phm_residual_gp_params()
        super(PhmResidualModelParams, self).__init__(path)
        
class PhmCorrectedPTModelParams(GPModelParams):
    """
    The model for the corrected PT model for Phm, modeled using a linear
    slope at high k, and interpolated using a Gaussian process as a function 
    of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.Phm_correctedPT_gp_params()
        super(PhmCorrectedPTModelParams, self).__init__(path)
        
class PhhModelParams(GPModelParams):
    """
    The model for Phh, interpolated using a Gaussian process as a 
    function of sigma8(z) and b1 
    """
    def __init__(self):
        path = sim_data.Phh_gp_params()
        super(PhhModelParams, self).__init__(path)
        

        
        