import scipy.interpolate
import numpy as np

class trapolator:
    """
    The base class for the different extrapolating/interpolating classes.
    
    Parameters
    ----------
    min_x : float, optional
        The minimum x value that the function is defined over. Default is ``None``,
        which is interpreted as min_x = -np.inf.
    max_x : float, optional
        The maximum x value that the function is defined over. Default is ``None``,
        which is interpreted as max_x = np.inf.
    """
    def __init__(self, min_x=None, max_x=None):
        self.x_range = (min_x, max_x)
    
    def __call__(self, x):
        return self.get(x)
    
    def valid(self, x):
        """
        Test whether the x value is within the domain specified by ``self.x_range``
        """
        x = np.asarray(x)
        if x.ndim == 0:
            return self.valid(np.array([x]))[0]
        ok = np.zeros(x.shape, 'bool')
        min_x, max_x = self.x_range
        if min_x is None:
            if max_x is None:
                # (-inf, inf)
                return np.ones(x.shape, 'bool')
            else:
                # (-inf, max)
                return x < max_x
        else:
            if max_x is None:
                # [min, inf)
                return x >= min_x
            else:
                # [min, max)
                return (x>=min_x)*(x<max_x)
#endclass trapolator
#-------------------------------------------------------------------------------

class basicInterpolator(trapolator):
    """
    Class to handle basic interpolation, with default interpolation of x, y(x) 
    performed by scipy.interpolate.interp1d.
    
    Parameters
    ----------
    x : array_like
        A 1-D array of monotonically increasing real values.
    y : array_like
        A 1-D array of real values, of len=len(x).
    min_x : float, optional
        The minimum x value that the function is defined over. Default is ``None``,
        which is interpreted as min_x = -np.inf.
    max_x : float, optional
        The maximum x value that the function is defined over. Default is ``None``,
        which is interpreted as max_x = np.inf.
    """
    INTERP = scipy.interpolate.interp1d  # use scipy.interpolate.interp1d
    YLOG   = False                       # do x vs y(x)
    
    def __init__(self, x, y, min_x=None, max_x=None):
        if min_x is None:
            min_x = x.min()
        if max_x is None:
            max_x = x.max()
        trapolator.__init__(self, min_x, max_x)
        self.ylog = self.YLOG
        if self.ylog:
            y = np.log(y)
        self.interp = self.INTERP(x, y)
        
    def get(self, x):
        y = self.interp(x)
        if self.ylog:
            y = np.exp(y)
        return y
#endclass basicInterpolator

#-------------------------------------------------------------------------------

# basicInterpolator does the interpolation in linear space
linearInterpolator = basicInterpolator

#-------------------------------------------------------------------------------
class logInterpolator(basicInterpolator):
    """
    Sub-class of ``basicInterpolator`` that uses scipy.interpolate.interp1d to 
    interpolate x vs log(y(x))
    """
    YLOG = True
#endclass logInterpolator

#-------------------------------------------------------------------------------
class splineInterpolator(basicInterpolator):
    """
    Sub-class of ``basicInterpolator`` that uses 
    scipy.interpolate.InterpolatedUnivariateSpline to interpolate x vs y(x)
    """
    INTERP = scipy.interpolate.InterpolatedUnivariateSpline
#endclass splineInterpolator

#-------------------------------------------------------------------------------
class splineLogInterpolator(splineInterpolator):
    """
    Sub-class of ``splineInterpolator`` that uses 
    scipy.interpolate.InterpolatedUnivariateSpline to interpolate x vs log(y(x))
    """
    YLOG = True
#endclass splineLogInterpolator

#-------------------------------------------------------------------------------
class powerLawExtrapolator(trapolator):
    """
    A class to perform power law extrapolations of a function.
    
    Parameters
    ----------
    gamma : float, optional
        The power law index, such that ``y = A*x**gamma``. Default is 1.
    A : float, optional
        The amplitude of the power law fit, such that 
        ``y = A*x**gamma``. Default is 1.
    min_x : float, optional
        The minimum x value that the function is defined over. Default is ``None``,
        which is interpreted as min_x = -np.inf.
    max_x : float, optional
        The maximum x value that the function is defined over. Default is ``None``,
        which is interpreted as max_x = np.inf.
    """
    def __init__(self, gamma=1., A=1., min_x=None, max_x=None):
        trapolator.__init__(self, min_x=min_x, max_x=max_x)
        self.params = gamma, A
    
    def get(self, x):
        gamma, A = self.params
        return A*x**gamma
#endclass powerLawExtrapolator

#-------------------------------------------------------------------------------
class exponentialExtrapolator(trapolator):
    """
    A class to perform exponential extrapolations of a function.
    
    Parameters
    ----------
    gamma : float, optional
        The exponential factor, such that ``y = A*exp(gamma*x)``. Default is 1.
    A : float, optional
        The amplitude of the fit, such that ``y = A*exp(gamma*x)``. Default is 1.
    min_x : float, optional
        The minimum x value that the function is defined over. Default is ``None``,
        which is interpreted as min_x = -np.inf.
    max_x : float, optional
        The maximum x value that the function is defined over. Default is ``None``,
        which is interpreted as max_x = np.inf.
    """
    def __init__(self, gamma=-1., A=1., min_x=None, max_x=None):
        trapolator.__init__(self, min_x=min_x, max_x=max_x)
        self.params = gamma, A
    
    def get(self, x):
        gamma, A = self.params
        return A*np.exp(x*gamma)
#endclass exponentialExtrapolator

#-------------------------------------------------------------------------------
class functionator(trapolator):
    """
    A class to combine several different trapolators in order to model a 
    full function. This is where the magic happens.
    
    Parameters
    ----------
    min_x : float, optional
        The minimum x value that the function is defined over. Default is ``None``,
        which is interpreted as min_x = -np.inf.
    max_x : float, optional
        The maximum x value that the function is defined over. Default is ``None``,
        which is interpreted as max_x = np.inf.
    ops : array_like, optional
        List of the trapolator instances that will make up the function.
    """
    def __init__(self, min_x=None, max_x=None, ops=[]):
        trapolator.__init__(self, min_x, max_x)
        self.ops = []
        for op in ops:
            self.append(op)

    def append(self, op):
        self.ops.append(op)

    def get(self, x):
        x = np.asarray(x)
        if x.ndim == 0:
            return self.get(np.array([x]))[0]
        ok = np.zeros(x.shape, 'bool')
        y = np.zeros(x.shape, 'float')
        for op in self.ops:
            s = op.valid(x)
            if any(s):
                y[s] = op.get(x[s])
                ok[s] = True
        if not all(ok):
            print 'Warning: functionator called out-of bounds'
            print (~ok).nonzero()
        return y

    def valid(self, x):
        x = np.asarray(x)
        if x.ndim == 0:
            return self.valid(array([x]))[0]
        ok = np.zeros(x.shape, 'bool')
        for op in self.ops:
            ok += op.valid(x)
        return ok
#endclass functionator
#-------------------------------------------------------------------------------
