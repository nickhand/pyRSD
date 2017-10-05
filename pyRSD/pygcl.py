from pyRSD import data_dir
import os
from six import add_metaclass, string_types

from pyRSD import gcl
from .gcl import ClassEngine
from .gcl import ClassParams
from .gcl import Cosmology
from .gcl import transfers

from .gcl import Constants
from .gcl import CubicSpline
from .gcl import LinearSpline
from .gcl import Spline

from .gcl import xi_to_pk, pk_to_xi
from .gcl import ComputeXiLM, compute_xilm_fftlog as ComputeXiLM_fftlog
from .gcl import IntegrationMethods
from .gcl import SimpsIntegrate, TrapzIntegrate

class DocFixer(type):

    def __new__(cls, name, bases, dct):
        dct['__doc__'] = bases[0].__doc__
        dct['__init__'].__doc__ = bases[0].__init__.__doc__
        return type.__new__(cls, name, bases, dct)

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
@add_metaclass(DocFixer)
class Cosmology(gcl.Cosmology, PickalableSWIG):

    def __init__(self, *args):

        # find the correct filenames, searching the data directory too
        newargs = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                newargs[i] = FindFilename(arg)
        gcl.Cosmology.__init__(self, *newargs)

    def __getstate__(self):
        args = [dict(self.GetParams()), self.GetTransferFit(), self.sigma8(),
                self.GetDiscreteK(), self.GetDiscreteTk()]
        return {'args': args}

    def __setstate__(self, state):
        args = state['args']
        args[0] = ClassParams.from_dict(args[0])
        self.__init__(*state['args'])

    def clone(self, tf=None):
        """
        Copy the Cosmology object, optionally changing the Transfer Function
        """
        args = self.__getstate__()['args']
        args[0] = ClassParams.from_dict(args[0])
        toret = Cosmology(*args)
        if tf is not None:
            toret.SetTransferFunction(tf)
        return toret

    @classmethod
    def from_power(cls, param_file, k, Pk):
        toret = cls.FromPower(FindFilename(param_file), k, Pk)
        toret.__class__ = Cosmology
        return toret

    @classmethod
    def from_file(cls, param_file, k, Tk):
        return cls(FindFilename(param_file), k, Tk)

    def __getitem__(self, key):
        if hasattr(self, key):
            f = getattr(self, key)
            if callable(f): return f()
        raise KeyError("Sorry, cannot return parameter '%s' in dict-like fashion" %key)

#-------------------------------------------------------------------------------
# CorrelationFunction
@add_metaclass(DocFixer)
class CorrelationFunction(gcl.CorrelationFunction, PickalableSWIG):

    def __init__(self, *args):
        self.args = args
        gcl.CorrelationFunction.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        if len(args) < 3: args = (args[0], self.GetKmin(), self.GetKmax())
        return {'args': args}

#-------------------------------------------------------------------------------
# Kaiser
@add_metaclass(DocFixer)
class Kaiser(gcl.Kaiser, PickalableSWIG):

    def __init__(self, *args):
        self.args = args
        gcl.Kaiser.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        if len(args) == 2: args += (self.GetLinearBias(),)
        return {'args': args}

#-------------------------------------------------------------------------------
# NonlinearPS
@add_metaclass(DocFixer)
class NonlinearPS(gcl.NonlinearPS, PickalableSWIG):

    def __init__(self, *args):
        args = list(args)
        args[0] = FindFilename(args[0])
        self.args = args
        gcl.NonlinearPS.__init__(self, *args)

#-------------------------------------------------------------------------------
# LinearPS
@add_metaclass(DocFixer)
class LinearPS(gcl.LinearPS, PickalableSWIG):

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
@add_metaclass(DocFixer)
class Imn(gcl.Imn, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.Imn.__init__(self, *args)

#-------------------------------------------------------------------------------

# Jmn
@add_metaclass(DocFixer)
class Jmn(gcl.Jmn, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.Jmn.__init__(self, *args)

#-------------------------------------------------------------------------------

# Kmn
@add_metaclass(DocFixer)
class Kmn(gcl.Kmn, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.Kmn.__init__(self, *args)

#-------------------------------------------------------------------------------

# ImnOneLoop
@add_metaclass(DocFixer)
class ImnOneLoop(gcl.ImnOneLoop, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.ImnOneLoop.__init__(self, *args)

#-------------------------------------------------------------------------------

# OneLoopPdd
@add_metaclass(DocFixer)
class OneLoopPdd(gcl.OneLoopPdd, PickalableSWIG):
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopPdd.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}

#-------------------------------------------------------------------------------

# OneLoopPdv
@add_metaclass(DocFixer)
class OneLoopPdv(gcl.OneLoopPdv, PickalableSWIG):
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopPdv.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}
#-------------------------------------------------------------------------------

# OneLoopPvv
@add_metaclass(DocFixer)
class OneLoopPvv(gcl.OneLoopPvv, PickalableSWIG):
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopPvv.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}
#-------------------------------------------------------------------------------

# OneLoopP22Bar
@add_metaclass(DocFixer)
class OneLoopP22Bar(gcl.OneLoopP22Bar, PickalableSWIG):
    def __init__(self, *args):
        self.args = (args[0],)
        gcl.OneLoopP22Bar.__init__(self, *args)

    def __getstate__(self):
        args = self.args
        args += (self.GetEpsrel(), self.GetOneLoopPower())
        return {'args': args}
#-------------------------------------------------------------------------------

# ZeldovichPS
@add_metaclass(DocFixer)
class ZeldovichPS(gcl.ZeldovichPS, PickalableSWIG):
    def __init__(self, *args):
        self.args = (args[0], )
        gcl.ZeldovichPS.__init__(self, *args)

    def __getstate__(self):
        args = (self.args[0], self.GetApproxLowKFlag(), self.GetSigma8AtZ(), self.GetK0Low(), self.GetSigmaSq(), self.GetX0Zel(), self.GetXZel(), self.GetYZel())
        return {'args': args}

#-------------------------------------------------------------------------------

# ZeldovichP00
@add_metaclass(DocFixer)
class ZeldovichP00(gcl.ZeldovichP00, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP00.__init__(self, *args)

#-------------------------------------------------------------------------------

# ZeldovichP01
@add_metaclass(DocFixer)
class ZeldovichP01(gcl.ZeldovichP01, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP01.__init__(self, *args)

#-------------------------------------------------------------------------------

# ZeldovichP11
@add_metaclass(DocFixer)
class ZeldovichP11(gcl.ZeldovichP11, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP11.__init__(self, *args)

#-------------------------------------------------------------------------------
# ZeldovichCF
@add_metaclass(DocFixer)
class ZeldovichCF(gcl.ZeldovichCF, PickalableSWIG,):
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichCF.__init__(self, *args)
