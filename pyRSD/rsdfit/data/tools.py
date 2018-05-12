import operator
from ... import numpy as np
from six import PY3

divop = operator.truediv if PY3 else operator.div

# Note: operator here gives the function needed to go from 
# `absolute` to `relative` units
variables = {"wavenumber" : {'operator': divop, 'power' : 1}, \
             "distance" : {'operator': operator.mul, 'power' : 1}, \
             "volume" : {'operator': operator.mul, 'power' : 3}, \
             "power" : {'operator': operator.mul, 'power' : 3} }


#-------------------------------------------------------------------------------
def histogram_bins(signal):
    """
    Return optimal number of bins.
    """
    select = -np.isnan(signal) & -np.isinf(signal)
    h = (3.5 * np.std(signal[select])) / (len(signal[select])**(1./3.))
    bins = int(np.ceil((max(signal[select])-min(signal[select])) / h))*2
    
    return bins
    
#-------------------------------------------------------------------------------
def h_conversion_factor(variable_type, input_units, output_units, h):
    """
    Return the factor needed to convert between the units, dealing with 
    the pesky dimensionless Hubble factor, `h`.
    
    Parameters
    ----------
    variable_type : str
        The name of the variable type, must be one of the keys defined
        in ``units.variables``.
    input_units : str, {`absolute`, `relative`}
        The type of the units for the input variable
    output_units : str, {`absolute`, `relative`}
        The type of the units for the output variable
    h : float
        The dimensionless Hubble factor to use to convert
    """
    units_types = ['relative', 'absolute']
    units_list = [input_units, output_units]
    if not all(t in units_types for t in units_list):
        raise ValueError("`input_units` and `output_units` must be one of %s, not %s" %(units_types, units_list))
        
    if variable_type not in variables.keys():
        raise ValueError("`variable_type` must be one of %s" %variables.keys())
    
    if input_units == output_units:
        return 1.
    
    exponent = variables[variable_type]['power']
    if input_units == "absolute":
        return variables[variable_type]['operator'](1., h)**(exponent)
        
    else:
        return 1./(variables[variable_type]['operator'](1., h)**(exponent))
        
#-------------------------------------------------------------------------------
def groupby_average(frame):
    """
    Compute the average of the columns of `frame`. This is designed to be 
    used with the `groupby` and `apply` functions, where the frame is 
    the `DataFrame of a given group. 
    """
    has_modes = False
    if hasattr(frame, 'modes'):
        has_modes = True
        total_modes = frame.modes.sum()
        
    if has_modes:
        weights = frame.modes.copy()
    else:
        weights = np.ones(len(frame))
        
    weights /= np.sum(weights)
    weighted = frame.multiply(weights, axis='index')
    
    toret = np.sum(weighted, axis=0)
    if has_modes: toret['modes'] = total_modes
    
    return toret
    
#-------------------------------------------------------------------------------