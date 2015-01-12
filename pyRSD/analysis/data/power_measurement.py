from ... import numpy as np
from ..parameters import ParameterSet, tools
from . import CovarianceMatrix

import pickle
import logging

logger = logging.getLogger('pyRSD.analysis.data')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
class PowerMeasurement(object):
    """
    Class representing a power spectrum measurement, either P(k, mu) or 
    multipole moments
    """
    def __init__(self, 
                 filename, 
                 k_col, 
                 power_col,
                 power_type, 
                 identifier, 
                 err_col=None, 
                 k_trim=None):
        """
        Load the parameters and initialize
        
        Parameters
        ----------
        filename : str
            The name of the ASCII file holding the measurement 
        k_col : int
            The integer giving the column number for the wavenumber data. 
            Wavenumbers should have units of `h/Mpc`
        power_col : int
            The integer giving the column number for the power data. 
            Power data should have units of `(Mpc/h)^3`
        power_type : {`pkmu`, `pole`}, str
            The type of power measurement
        identifier : float, int
            The value identifying the power spectrum, either a `mu` value, in
            which case a float should be passed, or `ell`, multipole number, in 
            which case an int should be passed
        err_col : int, optional
            The integer giving the column number for the error data. 
            Power data should have units of `(Mpc/h)^3`
        k_trim : float, optional
            Trim results to this value, in units of `h/Mpc`
        """
        # find the correct path
        filename = tools.find_file(filename)
                       
        # load the data
        data = np.loadtxt(filename)
        self._k_input = data[:,k_col]
        self._power_input = data[:,power_col]
        if err_col is not None:
            self._error_input = data[:,err_col]
        else:
            self._error_input = None
            
        if power_type not in ['pkmu', 'pole']:
            logger.error("PowerMeasurement must be of type 'pkmu' or 'pole', not '{0}'".format(power_type))
            raise ValueError("PowerMeasurement type must be either `pkmu` or `pole`")
            
        self.type = power_type
        self._identifier = identifier
        
        # set the k_trim
        self.k_trim = k_trim
        
    #---------------------------------------------------------------------------
    @property
    def k_trim(self):
        """
        Maximum k value to trim the results to in units of `h/Mpc`
        """
        return self._k_trim
        
    @k_trim.setter
    def k_trim(self, val):
        if val is None:
            self._k_trim = np.amax(self._k_input)
            self._k_trim_inds = ()
        else:
            self._k_trim = val
            self._k_trim_inds = np.where(self._k_input <= val)
            
    #---------------------------------------------------------------------------    
    @property
    def k(self):
        """
        The wavenumbers of the measurement in units of `h/Mpc`
        """
        return self._k_input[self._k_trim_inds]
        
    #---------------------------------------------------------------------------
    @property
    def power(self):
        """
        The power measurement in units of `(Mpc/h)^3`
        """
        return self._power_input[self._k_trim_inds]
        
    #---------------------------------------------------------------------------
    @property
    def error(self):
        """
        The error on the power measurement in units of `(Mpc/h)^3`
        """
        if self._error_input is not None:
            return self._error_input[self._k_trim_inds]
        else:
            return None 
        
    #---------------------------------------------------------------------------
    @property
    def mu(self):
        """
        If `type` == `pkmu`, then this returns the mu value associated with 
        the measurement
        """
        if self.type == 'pole':
            raise AttributeError("No `mu` attribute for `PowerMeasurement` of type `pole`")
        
        return self._identifier
            
    #---------------------------------------------------------------------------
    @property
    def ell(self):
        """
        If `type` == `pole`, then this returns the multipole number, ell, 
        associated with the measurement
        """
        if self.type == 'pkmu':
            raise AttributeError("No `ell` attribute for `PowerMeasurement` of type `pkmu`")
        
        return self._identifier
            
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        kwargs = {'mu' : self.mu, 'N' : len(self.k), 'k_max' : np.amax(self.k)}
        if self.type == 'pkmu':
            return "<PowerMeasurement P(k, mu={mu}), k_max = {k_max:.3} h/Mpc, {N} data points>".format(**kwargs)
        else:
            return "<PowerMeasurement P_{{ell={ell}}}(k), k_max = {k_max:.3} h/Mpc, {N} data points>".format(**kwargs)
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string representation
        """
        return self.__repr__()
    #---------------------------------------------------------------------------
#endclass PowerMeasurement

#-------------------------------------------------------------------------------
class PowerData(object):
    """
    Class to hold several `PowerMeasurement` objects and combine the 
    associated covariance matrices
    """
    def __init__(self, param_file):
        """
        Initialize and setup up the measurements. 
        """
        self.params = ParameterSet(param_file)
        
        # setup the measurements and covariances
        self._setup_measurements()
        self._setup_covariance()
        
    #---------------------------------------------------------------------------
    def __repr__(self):
        """
        Builtin representation method
        """
        toret = "Measurements\n" + "_"*12 + "\n"
        toret += "\n".join(map(str, self.measurements))
        
        if self.diagonal_covariance:
            toret += "\n\nusing diagonal covariance matrix"
        else:
            toret += "\n\nusing full covariance matrix"
        return toret
            
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Builtin string representation
        """
        return self.__repr__()
        
    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        """
        Integer access to the `measurements` attribute
        """
        if not isinstance(key, int):
            raise KeyError("`PowerMeasurement` index must be an integer")
        if key < 0: key += self.size
        if not 0 <= key < self.size:
            raise KeyError("`PowerMeasurement` index out of range")
            
        return self.measurements[key]
        
    #---------------------------------------------------------------------------
    def _setup_measurements(self):
        """
        Setup the measurements included in this `PowerData`
        """
        # measurement loading controlled by statistics parameter
        if 'statistics' not in self.params:
            raise ValueError("Parameter `statistics` must be passed")
        stats = self.params['statistics'].value
        
        # loop over each statistic
        self.measurements = []
        for stat_name in stats:
            
            # parse the name
            power_type, value = stat_name.lower().split('_')
            value = float(value)
            if power_type not in ['pkmu', 'pole']:
                logger.error("Measurement must be of type 'pkmu' or 'pole', not '{0}'".format(power_type))
                raise ValueError("Measurement type must be either `pkmu` or `pole`")
            
            # now make the PowerMeasurement object
            if stat_name not in self.params:
                raise ValueError("Statistic `%s` must have associated parameter for info" %stat_name)
            info = self.params[stat_name].value
            args = info['file'], info['x_col'], info['y_col'], power_type, value
            kwargs = {'err_col' : info.get('err_col', None), 'k_trim' : self.k_trim}
            self.measurements.append(PowerMeasurement(*args, **kwargs))
        
        logger.info("Read {N} measurements: {stats}".format(N=len(self.measurements), stats=stats))
            
    #---------------------------------------------------------------------------
    def _setup_covariance(self):
        """
        Setup the combined covariance matrix
        """
        # load the covariance from a pickle
        if self.params['covariance'] is not None:
            filename = tools.find_file(self.params['covariance'].value)
            C = pickle.load(open(filename, 'r'))
            index =  np.concatenate([d._k_input for d in self.measurements])
            self.covariance = CovarianceMatrix(C, index=index)
            logger.info("Read covariance matrix from pickle file '{f}'".format(f=filename))
            
            # possibly trim by a k_max
            if self.k_trim is not None:
                self.covariance = self.covariance.trim_by_index(self.k_trim)
                logger.info("Trimmed read covariance matrix to k_max = {k} h/Mpc".format(k=self.k_trim))
            
        # use the diagonals
        else:
            if any(isinstance(d.error, type(None)) for d in self.measurements):
                msg = "If no covariance matrix provided, all measurements must have errors"
                logger.error(msg)
                raise ValueError(msg)
                
            errors = np.concatenate([d.error for d in self.measurements])
            variances = errors**2
            self.covariance = CovarianceMatrix(variances, index=self.combined_k)
            logger.info('Initialized diagonal covariance matrix from error columns')
        
        # rescale the covariance matrix
        if self.params['covariance_rescaling'] is not None:
            rescale = self.params['covariance_rescaling'].value
            logger.info("Rescaled covariance matrix by value = {val}".format(val=str(rescale)))
            self.covariance *= rescale
        
        
        # verify the covariance matrix
        if len(self.combined_power) != self.covariance.N:
            args = (len(self.combined_power), self.covariance.N)
            logger.error("Combined power size {0}, covariance size {1}".format(*args))
            raise ValueError("Shape mismatch between covariance matrix and power data points")
            
    #---------------------------------------------------------------------------
    @property
    def size(self):
        """
        Return the number of measurements
        """
        return len(self.measurements)
    
    #---------------------------------------------------------------------------
    @property
    def combined_k(self):
        """
        The measurement k values, concatenated from each `PowerMeasurement`
        in `self.measurement`
        """
        try:
            return self._combined_k
        except AttributeError:
            self._combined_k = np.concatenate([d.k for d in self.measurements])
            return self._combined_k
            
    #---------------------------------------------------------------------------
    @property
    def combined_power(self):
        """
        The measurement power values, concatenated from each `PowerMeasurement`
        in `self.measurement`
        """
        try:
            return self._combined_power
        except AttributeError:
            self._combined_power = np.concatenate([d.power for d in self.measurements])
            return self._combined_power
            
    #---------------------------------------------------------------------------   
    @property
    def diagonal_covariance(self):
        """
        Return `True` if the covariance matrix is diagonal
        """
        try:
            return self._diagonal_covariance
        except AttributeError:
            C = self.covariance.asarray()
            self._diagonal_covariance = np.array_equal(np.nonzero(C), np.diag_indices_from(C))
            return self._diagonal_covariance
            
    #---------------------------------------------------------------------------
    @property
    def k_trim(self):
        """
        The wavenumber of the data that the results have been trimmed to
        """
        if 'k_trim' in self.params:
            return self.params['k_trim'].value
        else:
            return None
    
    #---------------------------------------------------------------------------
#endclass PowerData

#-------------------------------------------------------------------------------