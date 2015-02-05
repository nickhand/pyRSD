from .util import rsd_io
from . import FittingDriver
from .. import os

import logging
import tempfile

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
        
    # delete any old files
    if os.path.exists(temp_log_name):
        os.remove(temp_log_name)
    if restart is not None and os.path.exists(old_log_file):
        os.remove(old_log_file)
    
#-------------------------------------------------------------------------------
def add_console_logger():
    """
    Add a logger that logs to the console at level `INFO`.
    """ 
    # set up console logging
    logging.basicConfig(level=logging.INFO,
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
                                              
#-------------------------------------------------------------------------------
def add_file_logger(filename):
    """
    Add a logger that logs everything to a file at level `DEBUG` and
    logs to the console at level `INFO`. If :code:`silent = True`, then do 
    not log anything to the console
    """
    # define a Handler which writes DEBUG messages or higher file
    f = logging.FileHandler(filename, mode='w')
    f.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    f.setFormatter(formatter)
    logging.getLogger('').addHandler(f)
            
#-------------------------------------------------------------------------------
def run(args, pool):
    """
    Run the analysis steps as specified by the command line arguments passed
    to the script `pyRSDFitter`
    
    Parameters
    ----------
    args : Namespace
        A `Namespace` containing the arguments passed to the `pyRSDFitter`,
        script. These are the arguments returned from the parser initialized
        by `util.initialize_parser`
    """
    # console logger
    silent = args.silent if 'silent' in args else False
    if not silent: add_console_logger()
 
    # run the full fitting pipeline
    if args.subparser_name == 'run':

        # either load the driver if it exists already, or initialize it
        if 'driver.pickle' in args.params:
            driver = rsd_io.load_pickle(args.params)
        else:
            driver = FittingDriver(args.params, pool=pool)
        
        # store some command line arguments
        driver.params.set('walkers', args.walkers)
        driver.params.set('iterations', args.iterations)
        driver.params.set('threads', args.threads)
        
        # set driver values from command line
        solver = driver.params['fitter'].value
        if solver == 'emcee':
            if args.walkers is None:
                raise rsd_io.ConfigurationError("Please specify the number of walkers to use")
            if args.iterations is None:
                raise rsd_io.ConfigurationError("Please specify the number of steps to run")
                
        # log to a temporary file (for now)
        temp_log = tempfile.NamedTemporaryFile(delete=False)
        temp_log_name = temp_log.name
        temp_log.close()
        add_file_logger(temp_log_name)
        
        # run the fitting
        exception = driver.run()
                
        # get the output and finalize
        w, i = driver.results.walkers, driver.results.iterations
        output_name = rsd_io.create_output_file(args, solver, w, i)
        driver.finalize_fit(exception, output_name)
        
        # now let's save the new driver
        rsd_io.save_pickle(driver, os.path.join(args.folder, 'driver.pickle'))
        
        # now save the log to the logs dir
        copy_log(temp_log_name, output_name)
    
    # restart from previous chain
    elif args.subparser_name == 'restart':
        
        # load the driver 
        driver = rsd_io.load_pickle(args.params)
        
        # set driver values from command line
        driver.params.set('threads', args.threads)
        
        # log to a temporary file (for now)
        temp_log = tempfile.NamedTemporaryFile(delete=False)
        temp_log_name = temp_log.name
        temp_log.close()
        add_file_logger(temp_log_name)
       
        # run the restart with the iterations
        exception = driver.restart_chain(args.iterations, args.restart_file, pool=pool)
        
        # get the output and finalize
        w, i = driver.results.walkers, driver.results.iterations
        output_name = rsd_io.create_output_file(args, 'emcee', w, i)
        driver.finalize_fit(exception, output_name)
        
        # now save the log to the logs dir
        copy_log(temp_log_name, output_name, restart=args.restart_file)
        
        # if we made it this far, it's safe to delete the old results
        if os.path.exists(args.restart_file):
            os.remove(args.restart_file)
        
    # analyze an existing chain
    elif args.subparser_name == 'analyze':
        pass
        
#-------------------------------------------------------------------------------
        
        
    
    
    
