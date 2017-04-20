from pyRSD import numpy as np, os
from pyRSD.rsdfit import FittingDriver, params_filename, model_filename, logging
from pyRSD.rsdfit import GlobalFittingDriver
from pyRSD.rsdfit.util import rsd_io, rsdfit_parser
from pyRSD.rsdfit.util import rsd_logging, mpi_manager

from mpi4py import MPI
from emcee.utils import MPIPool

def find_init_result(val):
    """
    Return the name of the file holding the maximum probability
    from a directory
    """
    import os
    if not os.path.exists(val):
        raise RuntimeError("cannot set `start_from` to `%s`: no such file" %val)
    
    if os.path.isdir(val):
        from glob import glob
        from pyRSD.rsdfit.results import EmceeResults, LBFGSResults
        import operator
        
        pattern = os.path.join(val, "*.npz")
        result_files = glob(pattern)
        if not len(result_files):
            raise RuntimeError("did not find any chain (`.npz`) files matching pattern `%s`" %pattern)
        
        # find the chain file which has the maximum log prob in it and use that
        max_lnprobs = []
        for f in result_files:
            
            try:
                r = EmceeResults.from_npz(f)
                max_lnprobs.append(r.max_lnprob)
            except:
                r = LBFGSResults.from_npz(f)
                max_lnprobs.append(-r.min_chi2)

        index, value = max(enumerate(max_lnprobs), key=operator.itemgetter(1))
        return result_files[index]
    else:
        return val
 
class RSDFitDriver(object):
    """
    The main driver class to run `rsdfit`
    """
    def __init__(self, comm, mode, **kwargs):
        """
        Parameters
        ----------
        comm : MPI communicator
            the global MPI communicator that will optionally
            be split to distribute work in parallel
        mode : str
            the subparser name
        kwargs: 
            the key/value pairs corresponding to the command-line
            parser from rsd_parser()
        """
        self.comm = comm
        self.mode = mode
        self.restart_file = None
        
        self._config = []
        for k in kwargs:
            self._config.append(k)
            setattr(self, k, kwargs[k])
            
        self.preprocess()
        self.initialize_algorithm()
    
    @classmethod
    def create(cls, comm=None):
        """
        Parse the command-line options and
        return an initialized `RSDFitDriver`
        """
        if comm is None: comm = MPI.COMM_WORLD
    
        # rank 0 parses the command-line options
        if comm.size > 1:
            args = None
            if comm.rank == 0:
                args = rsdfit_parser().parse_args()    
            args = comm.bcast(args, root=0)
        else:
            args = rsdfit_parser().parse_args()
        
        # initialize the class and return
        args = vars(args)
        mode = args.pop('subparser_name')
        return cls(comm, mode, **args)
        
    
    @classmethod
    def parse_args(cls):
        """
        Parse the command-line arguments from the 
        parser returned by `rsdfit_parser()`
        """
        return rsdfit_parser().parse_args()
        
    def initialize_algorithm(self):
        """
        Initialize the driver that runs the desired algorithm
        """            
        # ``analysis`` mode
        if self.mode == 'analyze':
            from pyRSD.rsdfit import analysis
            kws = {k:getattr(self, k) for k in self._config}
            driver = analysis.AnalysisDriver(**kws)
        
        # ``run`` mode
        elif self.mode == 'run':
            params_file = os.path.join(self.folder, params_filename)
            
            # initialize from an existing directory, with existing parameter file
            if os.path.isdir(self.folder) and self.params == params_file and os.path.exists(params_file):            
                driver = FittingDriver.from_directory(self.folder, model_file=self.model)
            
            # initalize a new object from scratch
            else:
                init_model = self.model is None
                driver = FittingDriver(self.params, extra_param_file=self.extra_params, init_model=init_model)
                
                # initialize and save a model, if we need to
                if not init_model:
                    driver.model = self.model
                else:
                    if self.comm.rank == 0:
                        model_dir = driver.params.get('model_dir', self.folder)
                        np.save(os.path.join(model_dir, model_filename), driver.theory.model)
            
            # only one rank needs to write out
            if self.comm.rank == 0:
                driver.to_file(os.path.join(self.folder, params_filename))
            
            # have everyone wait
            self.comm.barrier()
            
            # set driver values from command line
            solver = driver.params['solver_type'].value
            if solver == 'mcmc':
                if self.walkers is None:
                    raise rsd_io.ConfigurationError("please specify the number of walkers to use")
                if self.iterations is None:
                    raise rsd_io.ConfigurationError("please specify the number of steps to run")
            
            # store some command line arguments
            driver.params.add('walkers', value=self.walkers)
            driver.params.add('iterations', value=self.iterations)
            
            # set max iterations for LBFGS
            if solver == 'nlopt' and self.iterations is not None:
                options = driver.params['lbfgs_options'].value
                options['max_iter'] = self.iterations
                
            # check if we need to find previous result
            init_from = driver.params.get('init_from', None)
            if init_from == 'result':
                start_from = driver.params.get('start_from', None)
                if start_from is None:
                    start_from = find_init_result(self.folder)
                elif os.path.isdir(start_from):
                    start_from = find_init_result(start_from)
        
                if not os.path.exists(start_from):
                    raise rsd_io.ConfigurationError("`start_from` parameter `%s` is not a valid path" %start_from)
                driver.params.add('start_from', value=start_from)
                
        # ``restart`` mode
        elif self.mode == 'restart':
                    
            # load the driver from param file, optionally reading model from file
            driver = FittingDriver.from_directory(self.folder, model_file=self.model)
    
            # set driver values from command line
            if self.burnin is not None:
                driver.params.add('burnin', value=self.burnin)
        
        self.algorithm = driver
        
    def preprocess(self):
        """
        Do some preprocessing
        """
        # set some restart variables
        if self.mode == 'restart':
            if 'nchains' not in self._config:
                self._config.append('nchains')
            self.nchains = len(self.restart_files)
        elif self.mode == 'analyze':
            if 'nchains' not in self._config:
                self._config.append('nchains')
            self.nchains = 1
        
        # too many chains requested?
        if self.nchains > self.comm.size:
            raise ValueError("number of chains requested must be less than total processes")
            
        # add the console logger
        silent = getattr(self, 'silent', False)
        if not silent: rsd_logging.add_console_logger(self.comm.rank)
        
        # remove any stream handlers, if silent
        if silent:
            logger = logging.getLogger()
            logger.handlers = [
                h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
            logger.addHandler(logging.NullHandler())
            
    def output_name(self, results, chain_number):
        """
        Return the name of the output file
        """
        kwargs = {}
        kwargs['walkers'] = getattr(results, 'walkers', None)
        if self.restart_file is not None:
            kwargs['restart'] = self.restart_file
        
        fitter = self.algorithm.params['solver_type'].value
        iterations =  getattr(results, 'iterations', 0)
        return rsd_io.create_output_file(self.folder, fitter, chain_number, iterations, **kwargs)
    
    def run(self):
        """
        Run the full `rsdfit` pipeline
        
        This uses `MPIManager` to enforce the pool behavior when calling
        the `run` function of the desired algorithm
        """
        # analyze mode
        if self.mode == 'analyze':
            self.algorithm.run()
            return
    
        # set the global algorithm for each rank
        GlobalFittingDriver.set(self.algorithm)
    
        # manage the MPI ranks
        debug = getattr(self, 'debug', False)
        with mpi_manager.MPIManager(self.comm, self.nchains, debug=debug) as mpi_master:
            
            # log all the results to a file
            with rsd_logging.FileLogger(mpi_master.rank, debug=debug) as logger:
                
                # set the restart file for this rank
                if self.mode == 'restart':
                    logger.restart = self.restart_file = self.restart_files[mpi_master.rank]
                    self.algorithm.set_restart(self.restart_file, self.iterations)

                # run the algorithm
                kws = {'pool':mpi_master.pool, 'chains_comm':mpi_master.par_runs_comm}
                logger.exception = self.algorithm.run(**kws)
                result = self.algorithm.results
                
                # finalize
                if result is not None:
                    logger.output_name = self.output_name(result, mpi_master.rank)
                    self.algorithm.finalize_fit(logger.exception, logger.output_name)
                    
                # finally raise the exception
                if logger.exception:
                    raise logger.exception
def main():
    
    # add a console logger
    rsd_logging.add_console_logger(MPI.COMM_WORLD.rank)
    
    # create and run
    driver = RSDFitDriver.create()
    driver.run()
    
    # force the exit
    os._exit(0)
    
if __name__ == "__main__":
    main()

    
    
