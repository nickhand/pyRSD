from ... import numpy as np
from .. import logging
from ..parameters import ParameterSet
from . import CovarianceMatrix

logger = logging.getLogger('rsdfit.data')
logger.addHandler(logging.NullHandler())

#-------------------------------------------------------------------------------
class PowerMeasurement(object):
    """
    Class representing a power spectrum measurement, either P(k, mu) or 
    multipole moments
    """
    def __init__(self,  
                 k, 
                 power,
                 power_type, 
                 identifier, 
                 error=None,
                 width=None, 
                 k_min=None,
                 k_max=None):
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
        width : float, optional
            If `power_type` == `pkmu`, this provides the width of the
            mu bin
        k_min : float, optional
            The minimum wavenumber (inclusive) in units of `h/Mpc`
        k_max : float, optional
            The maximum wavenumber (inclusive) in units of `h/Mpc`
        """                       
        # save the data
        self._k_input = k
        self._power_input = power
        if error is not None:
            self._error_input = error
        else:
            self._error_input = None
            
        if power_type not in ['pkmu', 'pole']:
            logger.error("PowerMeasurement must be of type 'pkmu' or 'pole', not '{0}'".format(power_type))
            raise ValueError("PowerMeasurement type must be either `pkmu` or `pole`")
            
        self.type = power_type
        self._identifier = identifier
        self._width = width
        
        # set the bounds
        self.k_min = k_min
        self.k_max = k_max
        
    #---------------------------------------------------------------------------
    @property
    def label(self):
        """
        Return the label associated with this kind of measurement
        """
        if self.type == 'pkmu':
            return self.type + '_' + str(self.mu)
        elif self.type == 'pkmu_diff':
            return self.type + '_{}_{}'.format(*self.mu)
        else:
            if self.type == 'pole':
                if self.ell == 0:
                    return 'monopole'
                elif self.ell == 2:
                    return 'quadrupole'
        raise NotImplementedError("Confused about what label corresponds to this measurement")
        
    #---------------------------------------------------------------------------
    @property
    def k_max(self):
        """
        Maximum k value to trim the results to in units of `h/Mpc`
        """
        return self._k_max
        
    @k_max.setter
    def k_max(self, val):
        if val is None:
            self._k_max = np.amax(self._k_input)
            self._k_max_inds = np.ones(len(self._k_input), dtype=bool)
        else:
            self._k_max = val
            self._k_max_inds = self._k_input <= val
            
    #---------------------------------------------------------------------------
    @property
    def k_min(self):
        """
        Minimum k value to trim the results to in units of `h/Mpc`
        """
        return self._k_min
        
    @k_min.setter
    def k_min(self, val):
        if val is None:
            self._k_min = np.amin(self._k_input)
            self._k_min_inds = np.ones(len(self._k_input), dtype=bool)
        else:
            self._k_min = val
            self._k_min_inds = self._k_input >= val
            
    #---------------------------------------------------------------------------
    @property
    def _k_trim_inds(self):
        """
        The indices of the data points following between k_min and k_max
        """    
        return self._k_min_inds & self._k_max_inds
        
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
        
        return self._identifier[self._k_trim_inds]
        
    #---------------------------------------------------------------------------
    @property
    def dmu(self):
        """
        If `type` == `pkmu`, then this returns the width of the mu bin associated
        with this measurement
        """
        if self.type == 'pole':
            raise AttributeError("No `dmu` attribute for `PowerMeasurement` of type `pole`")
        
        try:
            return self._width
        except:
            return None
            
    #---------------------------------------------------------------------------
    @property
    def ell(self):
        """
        If `type` == `pole`, then this returns the multipole number, ell (as
        an integer), associated with the measurement
        """
        if self.type == 'pkmu':
            raise AttributeError("No `ell` attribute for `PowerMeasurement` of type `pkmu`")
        
        return int(self._identifier)
            
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
        self.params = ParameterSet.from_file(param_file, tags='data')
        self.kmin, self.kmax = self.params['fitting_range'].value
            
        # read the data file
        self.read_data()

        # setup the measurements and covariances
        self._set_measurements()
        self._set_covariance()
        self._apply_bounds()
        self._rescale_inverse_covar()
        
    #---------------------------------------------------------------------------
    def read_data(self):
        """
        Read the data file
        """
        with open(self.params['data_file'].value, 'r') as ff:
            shape = tuple(map(int, ff.readline().split()))
            columns = ff.readline().split()
            data = np.loadtxt(ff)
        dtype = [(col, 'f8') for col in columns]
        self.data = np.empty(shape, dtype=dtype)
        for i, col in enumerate(columns):
            self.data[col] = data[...,i].reshape(shape)
            
    def to_file(self, filename, mode='w'):
        """
        Save the parameters of this data class to a file
        """            
        # save the params
        self.params.to_file(filename, mode=mode, header_name='data params', 
                            footer=True, as_dict=False)
                                  
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
            
    def __str__(self):
        """
        Builtin string representation
        """
        return self.__repr__()
        
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
    # internal functions to setup
    #---------------------------------------------------------------------------
    def _set_measurements(self):
        """
        Setup the measurements included in this `PowerData`
        """
        # measurement loading controlled by statistics parameter
        if 'statistics' not in self.params:
            raise ValueError("Parameter `statistics` must be passed")
        stats = self.params['statistics'].value
        
        # loop over each statistic
        self.measurements = []
        for i, stat_name in enumerate(stats):
            
            # parse the name
            power_type, value = stat_name.lower().split('_')
            value = float(value)
            
            if power_type not in ['pkmu', 'pole']:
                logger.error("Measurement must be of type 'pkmu' or 'pole', not '{0}'".format(power_type))
                raise ValueError("Measurement type must be either `pkmu` or `pole`")
            if power_type == 'pole':
                raise NotImplementedError("`pole` type not implemented currently")
                
            # now make the PowerMeasurement object
            k = self.data['k'][:,i]
            mu = self.data['mu'][:,i]
            power = self.data['power'][:,i]
            error = self.data['error'][:,i]
            self.measurements.append(PowerMeasurement(k, power, 'pkmu', mu, error=error))
            
        # # make sure all the ks are the same
        # tmp = self.measurements
        # if not all(np.array_equal(tmp[i].k, tmp[i+1].k) for i in range(len(tmp)-1)):
        #     msg = "All measurements read do not have same wavenumber array"
        #     logger.error(msg)
        #     raise ValueError(msg)
        logger.info("Read {N} measurements: {stats}".format(N=len(self.measurements), stats=stats))
            
    def _apply_bounds(self):
        """
        Apply the k bounds, and trim both the measurements and the covariance matrix
        """
        # trim the measurements
        inds = []
        for d in self.measurements:
            d.k_max = self.kmax
            d.k_min = self.kmin
            inds += list(d._k_trim_inds)
        
        # now trim and make the new covariance
        if self.kmax is not None or self.kmin is not None:
            C = self.covariance[inds]
            self.covariance = CovarianceMatrix(C, verify=False)
            logger.info("Trimmed read covariance matrix to [{}, {}] h/Mpc".format(self.kmin, self.kmax))
        
        # verify the covariance matrix
        if len(self.combined_power) != self.covariance.N:
            args = (len(self.combined_power), self.covariance.N)
            logger.error("Combined power size {0}, covariance size {1}".format(*args))
            raise ValueError("shape mismatch between covariance matrix and power data points")
            
    def _rescale_inverse_covar(self):
        """
        Rescale the inverse of the covariance matrix in order to get an unbiased estimate
        """
        # rescale the inverse of the covariance matrix (if it's from mocks)
        rescale_inverse = self.params.get('rescale_inverse_covariance', False)
        if rescale_inverse:
            if 'covariance_Nmocks' not in self.params:
                raise ValueError("cannot rescale inverse covariance without `covariance_Nmocks`")
            
            Ns = self.params['covariance_Nmocks'].value
            nb = len(self.combined_k)
            rescaling = 1.*(Ns - nb - 2) / (Ns - 1)
            self.covariance.inverse_rescaling = rescaling
            logger.info("rescaling inverse of covariance matrix using Ns = %d, nb = %d" %(Ns, nb))
            logger.info("   rescaling factor = %.3f" %rescaling)
            
    def _set_covariance(self):
        """
        Setup the combined covariance matrix
        
        Note: at this point, no k bounds have been applied
        """
        import pickle
        pickle_load = lambda filename: pickle.load(open(filename, 'r'))
        
        loaded = False
        index_ks = []
        for d in self.measurements: 
            index_ks += list(d.k)
        
        # load the covariance from a pickle
        if self.params['covariance'].value is not None:
            
            filename = self.params['covariance'].value
            readers = [CovarianceMatrix.from_pickle, CovarianceMatrix.from_plaintext, pickle_load]
            for reader in readers:
                try:
                    C = reader(filename)
                    break
                except Exception as e:
                    continue
            else:
                raise IOError("failure to load covariance from `%s`; tried pickle and plain text readers: %s" %(filename, str(e)))
                      
            if isinstance(C, np.ndarray):
                self.covariance = CovarianceMatrix(C, verify=False)
            elif isinstance(C, CovarianceMatrix):
                self.covariance = C
            
            if C.N != len(index_ks):
                msg = "loaded %d data points; covariance size = %dx%d" %(len(index_ks), C.N, C.N)
                raise ValueError("size mismatch between data and covariance; " + msg)
            logger.info("Read covariance matrix successfully from file '{f}'".format(f=filename)) 
            loaded = True           
                        
        # use the diagonals
        else:
            if any(isinstance(d.error, type(None)) for d in self.measurements):
                msg = "If no covariance matrix provided, all measurements must have errors"
                logger.error(msg)
                raise ValueError(msg)
                
            errors = np.concatenate([d.error for d in self.measurements])
            variances = errors**2
            self.covariance = CovarianceMatrix(variances, verify=False)
            logger.info('Initialized diagonal covariance matrix from error columns')
        
        # rescale the covariance matrix
        rescaling = self.params.get('covariance_rescaling', 1.0)
        self.covariance *= rescaling
        if rescaling != 1.0:
            logger.info("rescaled covariance matrix by value = {:s}".format(str(rescaling)))
                    
        # set errors for each indiv measurement to match any loaded covariance
        if loaded: self._set_errs_from_cov()
                
    def _set_errs_from_cov(self):
        """
        Set the errors for each individual measurement to match the diagonals
        of the covariance matrix
        """
        # the variances
        variances = self.covariance.diag
        for i, m in enumerate(self.measurements):
            size = len(m._k_input)
            errs = (variances[size*i : size*(i+1)])**0.5
            m._error_input = errs
            
    #---------------------------------------------------------------------------
    # properties
    #---------------------------------------------------------------------------
    @property
    def ndim(self):
        """
        The number of data bins 
        """
        return len(self.combined_power)
        
    @property
    def size(self):
        """
        Return the number of measurements
        """
        return len(self.measurements)
    
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
            
    @property
    def diagonal_covariance(self):
        """
        Return `True` if the covariance matrix is diagonal
        """
        try:
            return self._diagonal_covariance
        except AttributeError:
            C = self.covariance.full()
            self._diagonal_covariance = np.array_equal(np.nonzero(C), np.diag_indices_from(C))
            return self._diagonal_covariance
    