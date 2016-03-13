from pyRSD import data_dir
import os

import gcl
from gcl import ClassCosmology
from gcl import ClassParams
from gcl import Cosmology
from gcl import Engine

from gcl import Constants

from gcl import CubicSpline
from gcl import LinearSpline
from gcl import Spline

from gcl import xi_to_pk, pk_to_xi
from gcl import ComputeXiLM, compute_xilm_fftlog as ComputeXiLM_fftlog
from gcl import IntegrationMethods

class DocFixer(type):
    def __init__(cls, name, bases, attrs):
        base = bases[0]
        if base.__doc__ is not None:
            cls.__doc__ = base.__doc__
        if base.__init__.__doc__ is not None:
            cls.__init__.__func__.__doc__ = base.__init__.__doc__

class PickalableSWIG:
 
    def __setstate__(self, state):
        self.__init__(*state['args'])
 
    def __getstate__(self):
        return {'args': self.args}
    

def FindFilename(fname):
        
    if not os.path.exists(fname):
        fname_ = os.path.join(data_dir, 'params', fname)
        if not os.path.exists(fname_):
            raise ValueError("input file does not exists; tried %s and %s" %(fname, fname_))
        fname = fname_
    return fname

#-------------------------------------------------------------------------------
# Cosmology
class Cosmology(gcl.Cosmology, PickalableSWIG):
    __metaclass__ = DocFixer 
    
    def __init__(self, *args):
        
        # find the correct filenames, searching the data directory too
        newargs = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                newargs[i] = FindFilename(arg)
        gcl.Cosmology.__init__(self, *newargs)
    
    def __getstate__(self):
        args = [self.GetParamFile(), self.GetTransferFit(), self.sigma8(), self.GetTransferFile(), 
                self.GetDiscreteK(), self.GetDiscreteTk()]
        
        for i in [0, 3]:
            s = args[i].split(data_dir+'/params/')
            if len(s) == 2: args[i] = s[-1]
        return {'args': args} 
    
    @classmethod
    def from_power(cls, param_file, pkfile):
        # class the underlying c++ method
        toret = cls.FromPower(FindFilename(param_file), pkfile)
        # promote the class to pygcl.Cosmology
        toret.__class__ = Cosmology
        return toret
        
    @classmethod
    def from_file(cls, param_file, tkfile):
        return cls(FindFilename(param_file), FindFilename(tkfile))
            
    def __getitem__(self, key):
        if hasattr(self, key):
            f = getattr(self, key)
            if callable(f): return f()
        raise KeyError("Sorry, cannot return parameter '%s' in dict-like fashion" %key)
        
    def LoadTransferFunction(self, tkfile, kcol=1, tcol=2):
        # find the correct filename, searching the data directory too
        tkfile = FindFilename(tkfile)
        return gcl.Cosmology.LoadTransferFunction(self, tkfile, kcol, tcol)
            
#-------------------------------------------------------------------------------
# CorrelationFunction
class CorrelationFunction(gcl.CorrelationFunction, PickalableSWIG):
    __metaclass__ = DocFixer 
       
    def __init__(self, *args):
        self.args = args
        gcl.CorrelationFunction.__init__(self, *args)    
        
    def __getstate__(self):
        args = self.args
        if len(args) < 3: args = (args[0], self.GetKmin(), self.GetKmax())
        return {'args': args}    

#-------------------------------------------------------------------------------
# Kaiser
class Kaiser(gcl.Kaiser, PickalableSWIG):
    __metaclass__ = DocFixer
    
    def __init__(self, *args):
        self.args = args
        gcl.Kaiser.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        if len(args) == 2: args += (self.GetLinearBias(),)
        return {'args': args}
        
#-------------------------------------------------------------------------------
# NonlinearPS
class NonlinearPS(gcl.NonlinearPS, PickalableSWIG):
    __metaclass__ = DocFixer 
 
    def __init__(self, *args):
        args = list(args)
        args[0] = FindFilename(args[0])
        self.args = args
        gcl.NonlinearPS.__init__(self, *args)
        
#-------------------------------------------------------------------------------
# LinearPS
class LinearPS(gcl.LinearPS, PickalableSWIG):
    __metaclass__ = DocFixer 
    
    def __init__(self, *args):
        self.args = args
        gcl.LinearPS.__init__(self, *args)
        
    def __setstate__(self, state):
        self.__init__(*state['args'][:2])
        self.SetSigma8AtZ(state['args'][2])
 
    def __getstate__(self):
        args = self.args + (self.GetSigma8AtZ(),)
        return {'args': args}

#-------------------------------------------------------------------------------

# Imn
class Imn(gcl.Imn, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.Imn.__init__(self, *args)

#-------------------------------------------------------------------------------

# Jmn
class Jmn(gcl.Jmn, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.Jmn.__init__(self, *args)

#-------------------------------------------------------------------------------

# Kmn
class Kmn(gcl.Kmn, PickalableSWIG):
    __metaclass__ = DocFixer  
    def __init__(self, *args):
        self.args = args
        gcl.Kmn.__init__(self, *args)

#-------------------------------------------------------------------------------

# ImnOneLoop
class ImnOneLoop(gcl.ImnOneLoop, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.ImnOneLoop.__init__(self, *args)

#-------------------------------------------------------------------------------

# OneLoopPdd
class OneLoopPdd(gcl.OneLoopPdd, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopPdd.__init__(self, *args)
        
    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}

#-------------------------------------------------------------------------------

# OneLoopPdv
class OneLoopPdv(gcl.OneLoopPdv, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopPdv.__init__(self, *args)
        
    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}
#-------------------------------------------------------------------------------

# OneLoopPvv
class OneLoopPvv(gcl.OneLoopPvv, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopPvv.__init__(self, *args)
        
    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}
#-------------------------------------------------------------------------------

# OneLoopP22Bar
class OneLoopP22Bar(gcl.OneLoopP22Bar, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopP22Bar.__init__(self, *args)
    
    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}
#-------------------------------------------------------------------------------

# ZeldovichPS
class ZeldovichPS(gcl.ZeldovichPS, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = list(args)
        self.args.pop(1) # remove the redshift
        gcl.ZeldovichPS.__init__(self, *args)
        
    def __getstate__(self):
        args = self.args
        if len(args) == 1:
            args += (self.GetApproxLowKFlag(), )
        args += (self.GetSigma8AtZ(), self.GetK0Low(), self.GetSigmaSq(), self.GetX0Zel(), self.GetXZel(), self.GetYZel())
        return {'args': args}

#-------------------------------------------------------------------------------

# ZeldovichP00
class ZeldovichP00(gcl.ZeldovichP00, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP00.__init__(self, *args)
        
#-------------------------------------------------------------------------------

# ZeldovichP01
class ZeldovichP01(gcl.ZeldovichP01, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP01.__init__(self, *args)

#-------------------------------------------------------------------------------

# ZeldovichP11
class ZeldovichP11(gcl.ZeldovichP11, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP11.__init__(self, *args)

#-------------------------------------------------------------------------------    
# ZeldovichCF
class ZeldovichCF(gcl.ZeldovichCF, PickalableSWIG):
    __metaclass__ = DocFixer 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichCF.__init__(self, *args)    
