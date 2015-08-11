from .util import rsd_io, parse_command_line
from . import FittingDriver, logging, params_filename, model_filename
from .. import os, sys

import tempfile

def split_ranks(N_ranks, N_chunks):
    seq = range(N_ranks)
    avg = N_ranks / float(N_chunks)
    last = 0.

    i = 0
    while last < N_ranks:
        yield i, seq[int(last):int(last + avg)]
        last += avg
        i += 1
  
#-------------------------------------------------------------------------------
def copy_log(temp_log_name, output_name, restart=None):
    """
    Copy the contents of the log from the temporary log file    
    """
    # setup the log directory
    output_dir, fname = output_name.rsplit(os.path.sep, 1)
    log_dir = os.path.join(output_dir, 'logs')
    if not os.path.exists(log_dir): 
        os.makedirs(log_dir)
        
    # check if we need to copy over an old log file
    if restart is not None:
        old_log_file = os.path.join(log_dir, os.path.splitext(os.path.basename(restart))[0] + '.log')
        old_log_lines = open(old_log_file, 'r').readlines()
        
    log_file = open(os.path.join(log_dir, os.path.splitext(fname)[0] + '.log'), 'w')
    if restart is not None:
        for line in old_log_lines:
            log_file.write(line)
       
    # write out the lines 
    for line in open(temp_log_name, 'r'):
        log_file.write(line)
    log_file.close()
            
#-------------------------------------------------------------------------------
def add_console_logger(rank):
    """
    Add a logger that logs to the console at level `INFO`.
    """ 
    # set up console logging
    logging.basicConfig(level=logging.INFO,
                        format='chain #%d: '%rank + '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
                                              
#-------------------------------------------------------------------------------
def add_file_logger(filename, rank):
    """
    Add a logger that logs everything to a file at level `DEBUG` and
    logs to the console at level `INFO`. If :code:`silent = True`, then do 
    not log anything to the console
    """    
    # define a Handler which writes DEBUG messages or higher file
    f = logging.FileHandler(filename, mode='w')
    f.setLevel(logging.DEBUG)
    formatter = logging.Formatter('chain #%d: '%rank + '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    f.setFormatter(formatter)
    logging.getLogger('').addHandler(f)
            
#-------------------------------------------------------------------------------
def run():
    """
    Run the analysis steps as specified by the command line arguments passed
    to the script `rsdfits`
    
    Parameters
    ----------
    args : Namespace
        A `Namespace` containing the arguments passed to the `pyRSDFitter`,
        script. These are the arguments returned from the parser initialized
        by `util.initialize_parser`
    """
    from mpi4py import MPI
    from emcee.utils import MPIPool
    
    # get the world MPI attributes
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()
    world_group = world_comm.Get_group()
    
    # parse the command line arguments
    args = parse_command_line()
    
    # split ranks
    chains_group, chains_comm, pool_comm, pool = [None]*4
    if world_size > 1:
        ranges = []
        for i, ranks in split_ranks(world_size, args.nchains):
            ranges.append(ranks[0])
            if world_rank in ranks: color = i
        
        pool_comm = world_comm.Split(color, 0)
        if args.nchains > 1:
            chains_group = world_group.Incl(ranges)
            chains_comm = world_comm.Create(chains_group)
    
    # initialize the emcee pool, if the comm has more than 1 process
    if pool_comm is not None and pool_comm.size > 1:
        pool = MPIPool(comm=pool_comm)

    # if using MPI and not the master, wait for instructions
    if pool is not None and not pool.is_master():
        pool.wait()
        sys.exit(0)
    mpi_kwargs = {'pool':pool, 'chains_comm':chains_comm}

    # console logger
    silent = args.silent if 'silent' in args else False
    chain_number = chains_comm.rank if chains_comm is not None else 0
    if not silent: add_console_logger(chain_number)
    copy_kwargs = {}

    # run the full fitting pipeline
    if args.subparser_name == 'run':

        # either load the driver if it exists already, or initialize it
        if os.path.isdir(args.folder):            
            driver = FittingDriver.from_directory(args.folder, **mpi_kwargs)
        else:
            driver = FittingDriver(args.params, extra_param_file=args.extra_params, **mpi_kwargs)
            driver.to_file(os.path.join(args.folder, params_filename))
            rsd_io.save_pickle(driver.theory.model, os.path.join(args.folder, model_filename))
                
        # set driver values from command line
        solver = driver.params['fitter'].value
        if solver == 'emcee':
            if args.walkers is None:
                raise rsd_io.ConfigurationError("please specify the number of walkers to use")
            if args.iterations is None:
                raise rsd_io.ConfigurationError("please specify the number of steps to run")
        
        # store some command line arguments
        driver.params.add('walkers', value=args.walkers)
        driver.params.add('iterations', value=args.iterations)
                        
    # restart from previous chain
    elif args.subparser_name == 'restart':
        
        # load the driver from param file, optionally reading model from file
        initialize_model = False if 'model' in args else True
        driver = FittingDriver.from_restart(args.folder, args.restart_file, args.iterations, **mpi_kwargs)
        
        # set driver values from command line
        if args.burnin is not None:
            driver.params.add('burnin', value=args.burnin)
        copy_kwargs['restart'] = args.restart_file
        
    # analyze an existing chain
    elif args.subparser_name == 'analyze':
        return
            
    try:
        # log to a temporary file (for now)
        temp_log = tempfile.NamedTemporaryFile(delete=False)
        temp_log_name = temp_log.name
        temp_log.close()
        add_file_logger(temp_log_name, chain_number)

        # run the fitting
        exception = driver.run()
        
    except Exception as e:
        raise Exception(e)
    finally:
        
        # handle the MPI stuff
        if pool is not None: pool.close()
        if chains_group is not None: chains_group.Free()
        if chains_comm is not None: chains_comm.Free()
        
        # get the output and finalize
        kwargs = {}
        if driver.results is not None:
            kwargs['walkers'] = driver.results.walkers
            kwargs['iterations'] =  driver.results.iterations
            output_name = rsd_io.create_output_file(args, driver.params['fitter'].value, chain_number, **kwargs)
            driver.finalize_fit(exception, output_name)
    
            # now save the log to the logs dir
            copy_log(temp_log_name, output_name, **copy_kwargs)

        # if we made it this far, it's safe to delete the old results
        if os.path.exists(temp_log_name):
            os.remove(temp_log_name)
        if args.subparser_name == 'restart' and os.path.exists(args.restart_file):
            os.remove(args.restart_file)
        
#-------------------------------------------------------------------------------

    
    
