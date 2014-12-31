import gcl
from gcl import ClassCosmology
from gcl import ClassParams
from gcl import ComputeXiLM
from gcl import Constants
from gcl import CorrelationFunction
from gcl import Cosmology
from gcl import CubicSpline
from gcl import Engine
from gcl import CubicSpline
from gcl import LinearSpline
from gcl import ShiftedLinearSpline
from gcl import SmoothedXiMultipole
from gcl import Spline


class PickalableSWIG:
 
    def __setstate__(self, state):
        self.__init__(*state['args'])
 
    def __getstate__(self):
        return {'args': self.args}
#-------------------------------------------------------------------------------

# LinearPS
class LinearPS(gcl.LinearPS, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.LinearPS.__init__(self, *args)

#-------------------------------------------------------------------------------

# Imn
class Imn(gcl.Imn, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.Imn.__init__(self, *args)

#-------------------------------------------------------------------------------

# Jmn
class Jmn(gcl.Jmn, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.Jmn.__init__(self, *args)

#-------------------------------------------------------------------------------

# Kmn
class Kmn(gcl.Kmn, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.Kmn.__init__(self, *args)

#-------------------------------------------------------------------------------

# ImnOneLoop
class ImnOneLoop(gcl.ImnOneLoop, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.ImnOneLoop.__init__(self, *args)

#-------------------------------------------------------------------------------

# OneLoopPdd
class OneLoopPdd(gcl.OneLoopPdd, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.OneLoopPdd.__init__(self, *args)

#-------------------------------------------------------------------------------

# OneLoopPdv
class OneLoopPdv(gcl.OneLoopPdv, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.OneLoopPdv.__init__(self, *args)

#-------------------------------------------------------------------------------

# OneLoopPvv
class OneLoopPvv(gcl.OneLoopPvv, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.OneLoopPvv.__init__(self, *args)

#-------------------------------------------------------------------------------

# OneLoopP22Bar
class OneLoopP22Bar(gcl.OneLoopP22Bar, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.OneLoopP22Bar.__init__(self, *args)

#-------------------------------------------------------------------------------

# ZeldovichPS
class ZeldovichPS(gcl.ZeldovichPS, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichPS.__init__(self, *args)
        
    def __getstate__(self):
        args = self.args
        if len(self.args) == 1:
            args += (self.GetRedshift(), self.GetSigma8(), self.GetSigmaSq(), self.GetXZel(), self.GetYZel())
        return {'args': args}

#-------------------------------------------------------------------------------

# ZeldovichP00
class ZeldovichP00(gcl.ZeldovichP00, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP00.__init__(self, *args)
        
#-------------------------------------------------------------------------------

# ZeldovichP01
class ZeldovichP01(gcl.ZeldovichP01, PickalableSWIG):
 
    def __init__(self, *args):
        self.args = args
        gcl.ZeldovichP01.__init__(self, *args)

#-------------------------------------------------------------------------------
        
