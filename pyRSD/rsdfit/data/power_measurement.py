from ... import numpy as np
from ...rsd._cache import Cache, parameter, cached_property
from ...rsd import PkmuTransfer, PolesTransfer, PkmuGrid
from .. import logging, MPILoggerAdapter
from ..parameters import ParameterSet
from  . import PkmuCovarianceMatrix, PoleCovarianceMatrix

logger = MPILoggerAdapter(logging.getLogger('rsdfit.data'))

class PowerMeasurement(Cache):
    """
    Class representing a power spectrum measurement, either P(k, mu) or 
    P(k, ell)
    """
    def __init__(self, kind, data, kmin=None, kmax=None):
        """
        Load the parameters and initialize
        
        Parameters
        ----------
        kind : str, {`pkmu`, `pole`}
            the type of measurement, either P(k,mu) or P(k,ell)
        data : dict
            dictionary holding the relevant data fields
        kmin : float, optional
            The minimum wavenumber (inclusive) in units of `h/Mpc`
        kmax : float, optional
            The maximum wavenumber (inclusive) in units of `h/Mpc`
        """    
        # initialize the base class
        Cache.__init__(self)
         
        # the kind of measurement
        self.type = kind

        # set the data attributes
        for field in ['k', 'power']:
            setattr(self, '_%s_input' %field, data[field])
        if 'error' in data:
            setattr(self, '_error_input', data['error'])
        
        # mu/ell
        if self.type == 'pkmu': 
            self._mu_input = data['mu']
            self.ell = None
        else:
            self._mu_input = None
            self.ell = data['ell']
        
        # set the bounds
        self.kmin = kmin
        self.kmax = kmax
        
    #--------------------------------------------------------------------------
    # parameters
    #--------------------------------------------------------------------------
    @parameter
    def type(self, val):
        """
        The type of measurement, either `pkmu` or `pole`
        """
        if val not in ['pkmu', 'pole']:
            msg = "PowerMeasurement must be of type 'pkmu' or 'pole', not %s" %power_type
            logger.error(msg)
            raise ValueError(msg)
        return val
        
    @parameter
    def _k_input(self, val):
        """
        Internal variable to store input `k` values
        """
        return val
    
    @parameter
    def _power_input(self, val):
        """
        Internal variable to store input `power` values
        """
        return val
    
    @parameter
    def _error_input(self, val):
        """
        Internal variable to store input `error` values
        """
        return val
        
    @parameter
    def _mu_input(self, val):
        """
        Internal variable to store input `mu` values
        """
        return val
    
    @parameter
    def kmin(self, val):
        """
        Minimum k value to trim the results to in units of `h/Mpc`
        """
        if val is None:
            val = -np.inf
        return val
        
    @parameter
    def kmax(self, val):
        """
        Maximum k value to trim the results to in units of `h/Mpc`
        """
        if val is None:
            val = np.inf
        return val
                
    @parameter
    def ell(self, val):
        """
        Returns the multipole number for this P(k,ell) measurement
        """
        return val
    
    #--------------------------------------------------------------------------
    # cached properties
    #--------------------------------------------------------------------------
    @cached_property("_k_input", "kmin", "kmax")
    def _trim_idx(self):
        """
        The indices of the data points following between k_min and k_max
        """
        return (self._k_input >= self.kmin)&(self._k_input <= self.kmax)
        
    @cached_property("_k_input", "_trim_idx")
    def k(self):
        """
        The wavenumbers of the measurement in units of `h/Mpc`
        """
        return self._k_input[self._trim_idx]
        
    @cached_property("_mu_input", "_trim_idx")
    def mu(self):
        """
        Returns either a single or array of `mu` values associated with
        this P(k,mu) measurement
        """
        if self._mu_input is None or np.isscalar(self._mu_input):
            return self._mu_input
        else: 
            return self._mu_input[self._trim_idx]
        
    @cached_property("_power_input", "_trim_idx")
    def power(self):
        """
        The power measurement in units of `(Mpc/h)^3`
        """
        return self._power_input[self._trim_idx]
        
    @cached_property("_error_input", "_trim_idx")
    def error(self):
        """
        The error on the power measurement in units of `(Mpc/h)^3`
        """
        if np.isscalar(self._error_input):
            return self._error_input
        else:
            return self._error_input[self._trim_idx]
                
    @cached_property("k")
    def size(self):
        """
        The number of data points
        """
        return len(self.k)
        
    @cached_property("type", "ell", "mu")
    def identifier(self):
        """
        The `ell` or average `mu` value for this measurement
        """
        if self.type == 'pole':
            return int(self.ell)
        else:
            if not np.isscalar(self.mu):
                return np.round(np.mean(self.mu), 2)
            else:
                return np.round(self.mu, 2)
    
    @cached_property('type', 'identifier')
    def label(self):
        """
        Return the label associated with this kind of measurement
        """
        if self.type == 'pkmu':
            return self.type + '_' + str(self.identifier)
        else:
            if self.type == 'pole':
                if self.ell == 0:
                    return 'monopole'
                elif self.ell == 2:
                    return 'quadrupole'
                elif self.ell == 4:
                    return 'hexadecapole'
                
        raise NotImplementedError("confused about what label corresponds to this measurement")
        
    def __repr__(self):
        """
        Builtin representation method
        """
        if self.type == 'pkmu':
            meas = "P(k, mu=%s)" %str(self.identifier) 
        else:
            meas = "P(k, ell=%s)" %str(self.identifier)
        bounds = "({kmin:.3} - {kmax:.3})".format(kmin=np.amin(self.k), kmax=np.amax(self.k))
        return "<PowerMeasurement %s, %s h/Mpc, %d data points>" %(meas, bounds, self.size)

    def __str__(self):
        """
        Builtin string representation
        """
        return self.__repr__()
        

class PowerData(Cache):
    """
    Class to hold several `PowerMeasurement` objects and combine the 
    associated covariance matrices
    """
    def __init__(self, param_file):
        """
        Initialize and setup up the measurements 
        """
        Cache.__init__(self)
        
        # read the params from file
        self.params = ParameterSet.from_file(param_file, tags='data')
        self.mode = self.params.get('mode', 'pkmu') # either `pkmu` or `poles`
        
        # read the data file and (optional) grid for binning effects
        self.read_data()
        
        # mu_bounds / ells
        self._mu_bounds = self.params.get('mu_bounds', None)
        self._ells = self.params.get('ells', None)
        if self._ells is not None: 
            self._ells = np.asarray(self._ells, dtype=float)
        
        # create the measurements and covariances
        self.set_all_measurements()
        self.set_covariance()

        # slice the data
        self.slice_data()

        # set the k-limits
        self.set_k_limits()

        # rescale inverse covar?
        self.rescale_inverse_covar()

        # read a window function?
        self.window = None
        window_file = self.params.get('window_file', None)
        if window_file is not None: 
            self.window = np.loadtxt(window_file)
            
        # finally, read and setup an (optional) grid for binnning effects
        self.read_grid()

        # and verify
        self.verify()
        
    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def mode(self, val):
        """
        The measurement mode, either `pkmu` or `poles`
        """
        if val not in ['pkmu', 'poles']:
            raise ValueError("`PowerData` mode must be either `pkmu` or `poles`")
        return val
        
    @parameter
    def measurements(self, val):
        """
        List of `PowerMeasurement` objects
        """
        return val
        
    @parameter
    def covariance(self, val):
        """
        Either a `PkmuCovarianceMatrix` or `PoleCovarianceMatrix` object
        """
        return val
        
    @parameter
    def kmin(self, val):
        """
        The minimum allowed wavenumbers
        """
        toret = np.empty(self.size)
        toret[:] = val
        
        # trim the measurements and covariance
        for i, m in enumerate(self):
            m.kmin = toret[i]
        self.covariance = self.covariance.trim_k(kmin=toret)
        
        return toret
        
    @parameter
    def kmax(self, val):
        """
        The maximum allowed wavenumbers
        """
        toret = np.empty(self.size)
        toret[:] = val
        
        # trim the measurements and covariance
        for i, m in enumerate(self):
            m.kmax = toret[i]
        self.covariance = self.covariance.trim_k(kmax=toret)
        
        return toret
        
    @parameter
    def window(self, val):
        """
        The window function, default is `None`
        """
        return val
                            
    #---------------------------------------------------------------------------
    # initialization and setup functions
    #---------------------------------------------------------------------------
    def verify(self):
        """
        Verify the data and covariance
        """
        # log the kmin/kmax
        lims = ", ".join("(%.2f, %.2f)" %(x,y) for x,y in zip(self.kmin, self.kmax))
        logger.info("trimmed the read covariance matrix to: [%s] h/Mpc" %lims, on=0)
                           
        # verify the covariance matrix
        if self.ndim != self.covariance.N:
            args = (self.ndim, self.covariance.N)
            msg = "size mismatch: combined power size %d, covariance size %d" %args
            logger.error(msg)
            raise ValueError(msg)
        
        # verify the grid
        if self.transfer is not None:
            if self.transfer.size != self.covariance.N:
                msg = "size mismatch between grid transfer function and covariance: "
                args = (self.transfer.size, self.covariance.N)
                raise ValueError(msg + "grid size =  %d, cov size = %d" %args)
        
    def read_data(self):
        """
        Read the data file, storing either the measured 
        `P(k,mu)` or `P(k,ell)` data
        
        Notes
        -----
        *   sets `data` to be a structured array of shape
            `(Nk, Nmu)` or `(Nk, Nell)`
        """
        # read the data first
        data_file = self.params['data_file'].value
        with open(data_file, 'r') as ff:    
            shape = tuple(map(int, ff.readline().split()))
            columns = ff.readline().split()
            N = np.prod(shape)
            data = np.loadtxt(ff)
    
        # return a structured array
        dtype = [(col, 'f8') for col in columns]
        self.data = np.empty(shape, dtype=dtype)
        for i, col in enumerate(columns):
            self.data[col] = data[...,i].reshape(shape, order='F')
            
    def read_grid(self):
        """
        If `grid_file` is present, initialize `PkmuGrid`, which
        holds a finely-binned P(k,mu) grid to account for discrete
        binning effects
        
        Notes
        -----
        *   sets `grid` to be `None` or an instance of `PkmuGrid`
        *   sets `transfer` to be `None` or an instance of `PkmuTransfer`
            or `PolesTransfer`
        """
        self.grid = None; self.transfer = None
        
        grid_file = self.params.get('grid_file', None)
        if grid_file is not None:
            
            # initialize the grid
            self.grid = PkmuGrid.from_plaintext(grid_file)
            
            # set modes to zero outside k-ranges to avoid model failing
            self.grid.modes[self.grid.k < self.global_kmin] = 0.
            self.grid.modes[self.grid.k > self.global_kmax] = 0.

            # initialize the transfer
            if self.mode == 'pkmu':
                x = self._mu_bounds
                cls = PkmuTransfer; lab = 'mu_bounds'
            else:
                x = self._ells
                cls = PolesTransfer; lab = 'ells'
            
            # verify and initialize    
            if x is None:
                raise ValueError("`%s` parameter must be defined if `grid_file` is supplied" %lab)
            if len(x) != self.size:
                raise ValueError("size mismatch between `%s` and number of measurements" %lab)
            self.transfer = cls(self.grid, x, kmin=self.kmin, kmax=self.kmax)
            
            # set the window function
            if self.mode == 'poles' and self.window is not None:
                self.transfer.window = self.window
                    
    def set_all_measurements(self):
        """
        Initialize a list of `PowerMeasurement` objects from the 
        input data that has already been read
        
        Notes
        -----
        *   initializes `measurements`, which is a list of 
            a `PowerMeasurement` for each column in `data`, with
            no k limits applied
        """    
        # verify that the stats list has same length as data columns
        stats = self.params['statistics'].value
        if len(stats) != np.shape(self.data)[-1]:
            args = (len(stats), np.shape(self.data)[-1])
            raise ValueError("mismatch between number of data columns read and number of statistics")
        
        # loop over each statistic
        self.measurements = []
        for i, stat_name in enumerate(stats):
                        
            # parse the name
            power_type, value = stat_name.lower().split('_')
            value = float(value)
            
            if power_type not in ['pkmu', 'pole']:
                logger.error("measurement must be of type 'pkmu' or 'pole', not '%s'" %power_type)
                raise ValueError("measurement type must be either `pkmu` or `pole`")
                
            # get the relevant data for the PowerMeasurement
            fields = ['k', 'mu', 'power', 'error']
            data = {}
            for field in fields:
                if field in self.data.dtype.names:
                    data[field] = self.data[field][:,i]
            if power_type == 'pole': 
                data['ell'] = value
            
            # add the power measurement, with no k limits (yet)
            self.measurements.append(PowerMeasurement(power_type, data))
        
        # log the number of measurements read
        logger.info("read {N} measurements: {stats}".format(N=self.size, stats=stats), on=0)
    
    def set_covariance(self):
        """
        Read and set the combined covariance matrix
        
        Note: at this point, no k bounds have been applied
        """
        loaded = False
        N_data = sum(m.size for m in self)

        # try to load the covariance file
        cov_file = self.params.get('covariance', None)
        if cov_file is not None:
            
            # read either a PkmuCovariance or PoleCovarianceMatrix from
            # a plaintext file
            try:
                if self.mode == 'pkmu':
                    reader = 'PkmuCovarianceMatrix'
                    self.covariance = PkmuCovarianceMatrix.from_plaintext(cov_file)
                else:
                    reader = 'PoleCovarianceMatrix'
                    self.covariance = PoleCovarianceMatrix.from_plaintext(cov_file)
            except Exception as e:
                raise RuntimeError("failure to load %s from plaintext file: %s" %(reader, str(e)))
  
            # verify we have the right size, initially
            if self.covariance.N != N_data:
                N = self.covariance.N
                msg = "have %d data points, but covariance size is %dx%d" %(N_data, N, N)
                raise ValueError("size mismatch between read data and covariance -- " + msg)
            logger.info("read covariance matrix successfully from file '{f}'".format(f=cov_file), on=0) 
            loaded = True           
                        
        # use the diagonals
        else:
            if any(isinstance(m.error, type(None)) for m in self):
                msg = "if no covariance matrix provided, all measurements must have errors"
                logger.error(msg)
                raise ValueError(msg)
                
            # make the covariance matrix from diagonals
            x = np.concatenate([m.k for m in self])
            y = np.concatenate([np.repeat(m.identifier, m.size) for m in self])
            errors = np.concatenate([m.error for m in self])
            if self.mode == 'pkmu':
                self.covariance = PkmuCovarianceMatrix(errors**2, x, y, verify=False)
            else:
                self.covariance = PoleCovarianceMatrix(errors**2, x, y, verify=False)
            logger.info('initialized diagonal covariance matrix from error columns', on=0)
        
        # rescale the covariance matrix
        rescaling = self.params.get('covariance_rescaling', 1.0)
        self.covariance = self.covariance*rescaling
        if rescaling != 1.0:
            logger.info("rescaled covariance matrix by value = {:s}".format(str(rescaling)), on=0)
                    
        # set errors for each indiv measurement to match any loaded covariance
        if loaded: self.set_errors_from_cov()
    
    def set_errors_from_cov(self):
        """
        Set the errors for each individual measurement to match the 
        diagonals of the covariance matrix
        """
        # the variances
        variances = self.covariance.diag
        for i, m in enumerate(self):
            m._error_input = (variances[m.size*i :m. size*(i+1)])**0.5
            
    def slice_data(self):
        """
        Slice the data measurements + covariance to only include 
        certain statistics, corresponding to the column numbers in 
        `usedata`
        
        Notes
        -----
        *   the values in `usedata` are interpreted as the column
            numbers of `data` to use
        *   trims `measurements` and `covariance` accordingly
        """
        usedata = self.params.get('usedata', None)
        if usedata is None:
            return
        
        # compute the correct indexers for the covariance
        if self.mode == 'pkmu':
            mus = self.covariance.mus()
            trim = [mus[i] for i in usedata]
            indexer = {'mu1':trim, 'mu2':trim}
        else:
            ells = self.covariance.ells()
            trim = [ells[i] for i in usedata]
            indexer = {'ell1':trim, 'ell2':trim}
        self.covariance = self.covariance.sel(**indexer)

        # handle mu_bounds/ells slicing
        for key in ['_mu_bounds', '_ells']:
            val = getattr(self, key)
            if val is not None:
                if len(val) == self.size:
                    val = [val[i] for i in usedata]
                setattr(self, key, val)
                    
        
        # trim measurements
        self.measurements = [m for idx, m in enumerate(self) if idx in usedata]

    def set_k_limits(self):
        """
        Set the k-limits, reading from the parameter file
        
        Notes
        -----
        *   sets `N` to the number of statistics, i.e., either
            number of mu-bins or multipoles, that will be 
            included in the analysis 
        *   sets `kmin` and `kmax` to arrays of length `N`
            holding the minimum/maximum wavenumber values for
            each statistic
        *   sets `kmin` and `kmax` to the global kmin and kmax 
            values
        """        
        fit_range = self.params['fitting_range'].value
        if isinstance(fit_range, tuple) and len(fit_range) == 2:
            self.kmin = fit_range[0]
            self.kmax = fit_range[1]
        else:
            if len(fit_range) != self.size:
                raise ValueError("mismatch between supplied fitting ranges and data read")
            self.kmin, self.kmax = zip(*fit_range)
                                
    def rescale_inverse_covar(self):
        """
        Rescale the inverse of the covariance matrix in order to get an 
        unbiased estimate
        """
        # rescale the inverse of the covariance matrix (if it's from mocks)
        rescale_inverse = self.params.get('rescale_inverse_covariance', False)
        if rescale_inverse:
            if 'covariance_Nmocks' not in self.params:
                raise ValueError("cannot rescale inverse covariance without `covariance_Nmocks`")
            
            # set the inverse rescaling
            Ns = self.params['covariance_Nmocks'].value
            nb = self.ndim
            rescaling = 1.*(Ns - nb - 2) / (Ns - 1)
            self.covariance.inverse_rescaling = rescaling
            logger.info("rescaling inverse of covariance matrix using Ns = %d, nb = %d" %(Ns, nb), on=0)
            logger.info("   rescaling factor = %.3f" %rescaling, on=0)
            
            # update the `error` attribute of each measurement
            for m in self:
                m._error_input = m._error_input*rescaling**(-0.5)
            
    #---------------------------------------------------------------------------
    # some builtins
    #---------------------------------------------------------------------------
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
    
    def __iter__(self):
        """
        Iterate over the ``measurements`` list
        """
        return iter(self.measurements)
            
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property('window')
    def window_kmax_boost(self):
        """
        The boost factor for the `global_kmax` parameter
        """
        if self.window is None:
            return 1.
        else:
            return 1.5
            
    @cached_property('window')
    def window_kmin_boost(self):
        """
        The boost factor for the `global_kmin` parameter
        """
        if self.window is None:
            return 1.
        else:
            return 0.25
            
    @cached_property('kmin')
    def global_kmin(self):
        """
        The global mininum wavenumber
        """
        return self.window_kmin_boost * self.kmin.min()
        
    @cached_property('kmax', 'window_kmax_boost')
    def global_kmax(self):
        """
        The global maximum wavenumber
        """
        return self.window_kmax_boost * self.kmax.max()
            
    @cached_property('combined_power')
    def ndim(self):
        """
        The number of data bins 
        """
        return len(self.combined_power)
        
    @cached_property('measurements')
    def size(self):
        """
        Return the number of measurements
        """
        return len(self.measurements)
    
    @cached_property('measurements', 'kmin', 'kmax')
    def combined_k(self):
        """
        The measurement `k` values, concatenated from each 
        `PowerMeasurement` in `measurements`
        """
        return np.concatenate([m.k for m in self])
        
    @cached_property('measurements', 'kmin', 'kmax')
    def combined_mu(self):
        """
        The measurement `mu` values, concatenated from each 
        `PowerMeasurement` in `measurements`
        """
        tocat = []
        for m in self:
            if np.isscalar(m.mu):
                tocat.append(np.repeat(m.mu, m.size))
            else:
                tocat.append(m.mu)
        toret = np.concatenate(tocat)
        assert len(toret) == len(self.combined_k) 
        return toret
        
    @cached_property('measurements', 'kmin', 'kmax')
    def combined_ell(self):
        """
        The measurement `ell` values, concatenated from each 
        `PowerMeasurement` in `measurements`
        """
        return np.concatenate([np.repeat(m.ell, m.size) for m in self])
   
    @cached_property('measurements', 'kmin', 'kmax')
    def combined_power(self):
        """
        The measurement power values, concatenated from each `PowerMeasurement`
        in `self.measurement`
        """
        return np.concatenate([m.power for m in self])
        
    @cached_property('measurements', 'kmin', 'kmax')
    def combined_error(self):
        """
        The measurement `error` values, concatenated from each 
        `PowerMeasurement` in `measurements`
        """
        return np.concatenate([m.error for m in self])
            
    @cached_property('measurements', 'kmin', 'kmax')
    def diagonal_covariance(self):
        """
        Return `True` if the covariance matrix is diagonal
        """
        C = self.covariance.values
        return np.array_equal(np.nonzero(C), np.diag_indices_from(C))
    
    @property
    def flat_slices(self):
        """
        Return a list of length `size` that holds the slices needed
        to extract the `i-th` measurement from the flattened list
        of measurements
        """    
        idx = [0] + list(np.cumsum([m.size for m in self]))
        return [slice(idx[i], idx[i+1]) for i in range(self.size)]
    

    