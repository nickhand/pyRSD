from __future__ import print_function

from ... import numpy as np
from ...rsd._cache import Cache, parameter, cached_property
from ...rsd import PkmuTransfer, PolesTransfer, PkmuGrid, INTERP_KMAX
from ...rsd.window import WindowTransfer

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
                elif self.ell == 6:
                    return 'tetrahexadecapole'

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


class PowerDataSchema(Cache):
    """
    The schema for the :class:`PowerData` class, defining the allowed
    initialization parameters
    """
    @staticmethod
    def help():
        """
        Print out the help information for the necessary initialization parameters
        """
        print("Initialization Parameters for PowerData" + '\n' + '-'*50)
        for name in PowerDataSchema._param_names:
            par = getattr(PowerDataSchema, name)
            print(name+" :\n"+par.__doc__)

    @parameter
    def mode(self, val):
        """
        The type of data, either ``pkmu`` or ``poles``
        """
        if val not in ['pkmu', 'poles']:
            raise ValueError("the data 'mode' should be 'poles' or 'pkmu'")
        return val

    @parameter
    def statistics(self, val):
        """
        A list of the string names for each statistic that
        will be read from file; these strings should be of the form:

        >> ['pole_0', 'pole_2', ...]
        >> ['pkmu_0.1', 'pkmu_0.3', ...]
        """
        if not isinstance(val, list):
            raise TypeError("the data 'statistics' should be a list ")
        if not all(s.startswith('pole_') or s.startswith('pkmu_') for s in val):
            raise ValueError("all data statistics names should start with 'pole_' or 'pkmu_'")
        return val

    @parameter
    def fitting_range(self, val):
        """
        The fitting range for each statistics.

        This can either be a tuple of (kmin, kmax), which will be
        used for each statistic or a list of tuples of (kmin, kmax)
        """
        return val

    @parameter
    def data_file(self, val):
        """
        The string specifying the name of the file holding the data measurements
        """
        return val

    @parameter(default=None)
    def grid_file(self, val):
        """
        A string specifying the name of the file holding a
        :class:`pyRSD.rsd.transfer.PkmuGrid` to read
        """
        return val

    @parameter(default=None)
    def window_file(self, val):
        """
        A string specifying the name of the file holding a
        the multipoles of the window function in configuration space
        """
        return val

    @parameter
    def covariance(self, val):
        """
        The string specifying the name of the file holding the covariance matrix
        """
        return val

    @parameter(default=None)
    def usedata(self, val):
        """
        A list of statistic numbers that will be included in the final
        analysis. This allows one to exclude certain statistics read from file

        By default (``None``), all statistics are included
        """
        return val

    @parameter(default=1.0)
    def covariance_rescaling(self, val):
        """
        Rescale the covariance matrix read from file by this amount
        """
        return val

    @parameter(default=0.)
    def covariance_Nmocks(self, val):
        """
        The number of mocks that was used to measure the covariance matrix

        If this is non-zero, then the inverse covariance matrix will
        be rescaled to account for noise due to the finite number of mocks
        """
        return val

    @parameter(default=None)
    def ells(self, val):
        """
        A list of integers specifying the multipole numbers, corresponding
        to each statistic in the final analysis

        This must be supplied when the :attr:`mode` is ``poles``
        """
        return val

    @parameter(default=None)
    def mu_bounds(self, val):
        """
        A list of tuples specifying (mu_min, mu_max), corresponding
        to the edges of the mu bins for each statistic in the final analysis

        This must be supplied when the :attr:`mode` is ``pkmu``
        """
        return val

    @parameter(default=4)
    def max_ellprime(self, val):
        """
        When convolving a multipoles of order ``ell``, include contributions up
        to and including this multipole number
        """
        return val

class PowerData(PowerDataSchema):
    """
    Class to hold several `PowerMeasurement` objects and combine the
    associated covariance matrices
    """
    def __init__(self, param_file):
        """
        Initialize and setup up the measurements
        """
        # read the params from file
        self.params = ParameterSet.from_file(param_file, tags='data')

        # set all of the valid ones
        for name in PowerDataSchema._param_names:
            if name in self.params:
                setattr(self, name, self.params[name].value)
            try:
                has_default = getattr(self, name)
            except ValueError:
                raise ValueError("PowerData class is missing the '%s' initialization parameter" %name)

        # and initialize
        self.initialize()

    def initialize(self):
        """
        Do the initialization steps after reading the params
        """
        # read the data file and (optional) grid for binning effects
        self.read_data()

        # store ells as floats
        if self.ells is not None:
            self.ells = np.asarray(self.ells, dtype=float)

        # create the measurements and covariances
        self.set_all_measurements()
        self.set_covariance()

        # slice the data
        self.slice_data()

        # log the measurements
        labels = [m.label for m in self.measurements]
        logger.info("using measurements: %s" %str(labels), on=0)

        # set the k-limits
        self.set_k_limits()

        # rescale inverse covar?
        self.rescale_inverse_covar()

        # read a window function?
        self.window = None
        if self.window_file is not None:
            self.window = np.loadtxt(self.window_file)

        # finally, read and setup an (optional) grid for binnning effects
        self.read_grid()

        # and verify
        self.verify()

    #---------------------------------------------------------------------------
    # additional parameters
    #---------------------------------------------------------------------------
    @parameter
    def measurements(self, val):
        """
        List of `PowerMeasurement` objects
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
        self.covariance_matrix = self.covariance_matrix.trim_k(kmin=toret)

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
        self.covariance_matrix = self.covariance_matrix.trim_k(kmax=toret)

        return toret

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
        if self.ndim != self.covariance_matrix.N:
            args = (self.ndim, self.covariance_matrix.N)
            msg = "size mismatch: combined power size %d, covariance size %d" %args
            logger.error(msg)
            raise ValueError(msg)

        # verify the grid
        if self.transfer is not None:
            if self.binning_transfer.size != self.covariance_matrix.N:
                msg = "size mismatch between grid transfer function and covariance: "
                args = (t.size, self.covariance_matrix.N)
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
        with open(self.data_file, 'r') as ff:
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

        if self.grid_file is not None:

            # initialize the grid
            self.grid = PkmuGrid.from_plaintext(self.grid_file)

            # set modes to zero outside k-ranges to avoid model failing
            self.grid.modes[self.grid.k < self.global_kmin] = np.nan
            self.grid.modes[self.grid.k > self.global_kmax] = np.nan

            # initialize the transfer
            if self.mode == 'pkmu':
                x = self.mu_bounds
                cls = PkmuTransfer; lab = 'mu_bounds'
            else:
                x = self.ells
                cls = PolesTransfer; lab = 'ells'
            kws = {'kmin':self.kmin, 'kmax':self.kmax}

            # verify and initialize
            if x is None:
                raise ValueError("`%s` parameter must be defined if `grid_file` is supplied" %lab)
            if len(x) != self.size:
                raise ValueError("size mismatch between `%s` and number of measurements" %lab)

            # initialize the transfer function
            self.binning_transfer = cls(self.grid, x, **kws)
            if self.window is None:
                self.transfer = self.binning_transfer
            else:
                kws = {}

                # determine max ell prime and which poles we need
                if max(x) < self.max_ellprime:
                    kws['ells_mask'] = [True if ell in x else False for ell in range(0, self.max_ellprime+1, 2)]
                    x = list(range(0, self.max_ellprime+1, 2))
                kws['max_ellprime'] = self.max_ellprime

                # evaluate at k_out of the binning grid
                self.transfer = WindowTransfer(self.window, x, k_out=self.binning_transfer.coords[0], **kws)

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
        stats = self.statistics
        if len(stats) != np.shape(self.data)[-1]:
            args = (len(stats), np.shape(self.data)[-1])
            raise ValueError(("mismatch between number of data columns read and number of statistics:"
                            + " %d statistics and %d data columns read" %args))

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
        if self.covariance is not None:

            try:
                if self.mode == 'pkmu':
                    self.covariance_matrix = PkmuCovarianceMatrix.from_plaintext(self.covariance)
                else:
                    self.covariance_matrix = PoleCovarianceMatrix.from_plaintext(self.covariance)
                logger.info("read covariance matrix successfully from file '{f}'".format(f=self.covariance), on=0)
            except:
                raise RuntimeError("failure to load covariance from plaintext file: '%s'" %self.covariance)

            # verify we have the right size, initially
            if self.covariance_matrix.N != N_data:
                N = self.covariance_matrix.N
                msg = "have %d data points, but covariance size is %dx%d" %(N_data, N, N)
                raise ValueError("size mismatch between read data and covariance -- " + msg)
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
                self.covariance_matrix = PkmuCovarianceMatrix(errors**2, x, y, verify=False)
            else:
                self.covariance_matrix = PoleCovarianceMatrix(errors**2, x, y, verify=False)
            logger.info('initialized diagonal covariance matrix from error columns', on=0)

        # rescale the covariance matrix
        rescaling = self.covariance_rescaling # default is 1.0
        self.covariance_matrix *= rescaling
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
        variances = self.covariance_matrix.diag
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
        if self.usedata is None:
            return

        # compute the correct indexers for the covariance
        if self.mode == 'pkmu':
            mus = self.covariance_matrix.mus()
            trim = [mus[i] for i in self.usedata]
            indexer = {'mu1':trim, 'mu2':trim}
        else:
            ells = self.covariance_matrix.ells()
            trim = [ells[i] for i in self.usedata]
            indexer = {'ell1':trim, 'ell2':trim}
        self.covariance_matrix = self.covariance_matrix.sel(**indexer)

        # handle mu_bounds/ells slicing
        for key in ['mu_bounds', 'ells']:
            val = getattr(self, key)
            if val is not None:
                if len(val) == self.size:
                    val = [val[i] for i in self.usedata]
                setattr(self, key, val)

        # trim measurements
        self.measurements = [m for idx, m in enumerate(self) if idx in self.usedata]

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
        fit_range = self.fitting_range
        if isinstance(fit_range, tuple) and len(fit_range) == 2:
            self.kmin = fit_range[0]
            self.kmax = fit_range[1]
        else:
            if len(fit_range) != self.size:
                raise ValueError("mismatch between supplied fitting ranges and data read")
            self.kmin, self.kmax = list(zip(*fit_range))

    def rescale_inverse_covar(self):
        """
        Rescale the inverse of the covariance matrix in order to get an
        unbiased estimate
        """
        # rescale the inverse of the covariance matrix (if it's from mocks)
        rescale_inverse = self.covariance_Nmocks > 0
        if rescale_inverse:

            # set the inverse rescaling
            Ns = self.covariance_Nmocks
            nb = self.ndim
            rescaling = 1.*(Ns - nb - 2) / (Ns - 1)
            self.covariance_matrix.inverse_rescaling = rescaling
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
    @cached_property('kmin')
    def global_kmin(self):
        """
        The global mininum wavenumber
        """
        if self.window is None:
            return self.kmin.min()
        else:
            return min(1e-4, self.kmin.min())

    @cached_property('kmax')
    def global_kmax(self):
        """
        The global maximum wavenumber
        """
        if self.window is None:
            return self.kmax.max()
        else:
            return max(INTERP_KMAX, self.kmax.max())

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
        C = self.covariance_matrix.values
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
