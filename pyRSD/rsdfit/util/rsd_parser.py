from . import rsd_io
from ... import os

import argparse as ap
import textwrap as tw
import logging


logger = logging.getLogger('rsdfit.parser')
logger.addHandler(logging.NullHandler())


#-------------------------------------------------------------------------------
# parser types
#-------------------------------------------------------------------------------
def existing_file(fname):
    """
    Check if the file exists. If not raise an error
    
    Parameters
    ----------
    fname: str
        the name of the file to read
        
    Returns
    -------
    fname : string
    """
    if os.path.isfile(fname):
        return fname
    else:
        msg = "The file '{}' does not exist".format(fname)
        raise ap.ArgumentTypeError(msg)

#-------------------------------------------------------------------------------
def positive_int(string):
    """
    Check if the input is integer positive
    
    Parameters
    ----------
    string: str
        string to parse
    output: int
        return the integer
    """
    try:
        value = int(string)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise ap.ArgumentTypeError("Argument requires a positive integer")
            
#-------------------------------------------------------------------------------
def initialize_parser():
    """
    Initialize the parser of command-line arguments. 
    
    The main parser has three subparsers, `run`, `restart`, and `analyze`. 
    If you run :code:`runRSDFitter -h`, this information will be printed to 
    the console. Further help information can be found by including the name 
    of a specific submode, i.e., run code:`pyRSDFitter run -h`.
    """
    # set up the main parser
    usage = """%(prog)s [-h] [--version] {run,restart,analyze} ... """
    usage += tw.dedent("""\n
        From more help on each of the subcommands, type:
        %(prog)s run -h
        %(prog)s restart -h
        %(prog)s analyze -h\n\n""")
    desc = "fitting redshift space power spectrum observations with the `pyRSD` model"
    kwargs = {}
    kwargs['usage'] = usage
    kwargs['description'] = desc
    kwargs['formatter_class'] = ap.ArgumentDefaultsHelpFormatter
    parser = ap.ArgumentParser(**kwargs)
    
    # add the subparsers
    subparser = parser.add_subparsers(dest='subparser_name')
    
    #---------------------------------------------------------------------------
    # RUN SUBPARSER
    #---------------------------------------------------------------------------
    run_parser = subparser.add_parser('run', help="run the MCMC chains")
    
    # the general driver parameters
    h = 'file name holding the driver, theory, and data parameters'
    kwargs = {'dest':'params', 'type':existing_file, 'help':h}
    run_parser.add_argument('-p', '--params', **kwargs) 
        
    # number of threads (OPTIONAL)
    h = 'number of python multiprocessing threads to spawn (default: 1)'
    run_parser.add_argument('-N', help=h, type=int, default=1, dest='threads')
    
    # silence the output (OPTIONAL)
    h = 'silence the standard output to the console'
    run_parser.add_argument('--silent', help=h, action='store_true')
    
    # number of walkers (OPTIONAL)
    h = 'number of walkers in the chain'
    run_parser.add_argument('-w', help=h, type=positive_int, dest='walkers')
    
    # number of iterations (OPTIONAL)
    h = 'number of steps in the chain to run'
    run_parser.add_argument('-i', help=h, type=positive_int, dest='iterations')
    
    # the output folder
    h = 'the folder where the results will be written'
    kwargs = {'help':h, 'type':str, 'required':True, 'dest':'folder'}
    run_parser.add_argument('-o', '--output', **kwargs)
    
    # arbitrary numbering of an output chain (OPTIONAL)
    h = """An arbitrary number for the output chain. \n
           By default, the chains are named `yyyy-mm-dd_KxM__i.txt` with
           year, month and day being extracted, `K` being the number of
           walkers, `M` being the number of steps, and `i` an 
           automatically updated index."""
    run_parser.add_argument('--chain-number', help=h)
    
    #---------------------------------------------------------------------------
    # RESTART SUBPARSER
    #---------------------------------------------------------------------------
    h = "restart a fit from an existing chain"
    restart_parser = subparser.add_parser('restart', help=h)
    
    # the general driver parameters (REQUIRED)
    h = 'the name of the existing results file to restart from'
    restart_parser.add_argument('restart_file', type=existing_file, help=h)
    
    # number of iterations (REQUIRED)
    h = 'the number of additional steps to run using the old chain'
    restart_parser.add_argument('-i', help=h, required=True, type=positive_int, 
                                default=0, dest='iterations')
                                
    # number of iterations (REQUIRED)
    h = 'the number of steps to consider burnin'
    restart_parser.add_argument('-b', help=h, type=positive_int, dest='burnin')
                                
    # number of threads (OPTIONAL)
    h = 'number of python multiprocessing threads to spawn (default: 1)'
    restart_parser.add_argument('-N', help=h, type=int, default=1, dest='threads')
    
    # arbitrary numbering of an output chain (OPTIONAL)
    h = """An arbitrary number for the output chain. \n
           By default, the chains are named `yyyy-mm-dd_KxM__i.txt` with
           year, month and day being extracted, `K` being the number of
           walkers, `M` being the number of steps, and `i` an 
           automatically updated index."""
    restart_parser.add_argument('--chain-number', help=h)
    
    h = 'silence the standard output to the console'
    restart_parser.add_argument('--silent', help=h, action='store_true')
    
    #---------------------------------------------------------------------------
    # ANALYZE SUBPARSER
    #---------------------------------------------------------------------------
    h = "analyze the MCMC chains"
    kwargs = {'help':h, 'formatter_class':ap.ArgumentDefaultsHelpFormatter}
    analyze_parser = subparser.add_parser('analyze', **kwargs)

    # the folder to analyze
    h = "files to analyze: either a single file, or a complete folder"
    analyze_parser.add_argument('files', help=h, nargs='+')
    
    # to only write the covmat and bestfit, without computing the posterior
    h = "use this flag to avoid computing the posterior distribution"
    analyze_parser.add_argument('--minimal', help=h, action='store_true')
    
    # the number of bins (defaulting to 20)
    h = """number of bins in the histograms used to derive posterior 
           probabilities and credible intervals"""
    analyze_parser.add_argument('--bins', help=h, type=int, default=20)
                                                 
    # to remove the mean-likelihood line
    h = "remove the mean likelihood from the 1D posterior plots"
    analyze_parser.add_argument('--no-mean', help=h, dest='mean_likelihood', 
                                action='store_false')
    
    # if you just want the covariance matrix, use this option
    h = "do not produce any plot, simply compute the posterior"
    analyze_parser.add_argument('--noplot', help=h, dest='plot', 
                                action='store_false')
                            
    # if you just want to output 1d posterior distributions (faster)
    h = "produce only the 1d posterior plot"
    analyze_parser.add_argument('--noplot-2d', help=h, dest='plot_2d', 
                                action='store_false')
                            
    # when plotting 2d posterior distribution, use contours and not contours
    # filled (might be useful when comparing several folders)
    h = "do not fill the contours on the 2d plot"
    analyze_parser.add_argument('--contours-only', help=h, dest='contours_only', 
                                action='store_true')
                                
    # if you want to output every single subplots
    h = "output every subplot and data in separate files"
    analyze_parser.add_argument('--all', help=h, dest='subplot', 
                                action='store_true')
    
    # output file extension
    h = "change the extension for the output file."
    analyze_parser.add_argument('--ext', help=h, type=str, dest='extension', 
                                default='pdf', choices=['pdf', 'png', 'eps'])
                                
    
    return parser

#-------------------------------------------------------------------------------
def parse_command_line():
    """
    Parse the command line arguments
    """
    # initialize the parser and parse
    parser = initialize_parser()
    args = parser.parse_args()
    
    # a few checks
    if args.subparser_name == 'restart':
        
        # if we are restarting, automatically use the same folder, 
        # and the driver.pickle
        args.folder = os.path.sep.join(args.restart_file.split(os.path.sep)[:-1])
        args.params = os.path.join(args.folder, 'driver.pickle')
        if not os.path.exists(args.params):
            raise rsd_io.ConfigurationError(
                  "Restarting but associated driver.pickle doesn't exist")
        logger.warning("Restarting from %s." %args.restart_file +
                       " Using associated driver.pickle")
    elif args.subparser_name == "run":

        # if the folder already exists, and no parameter files were specified
        # try to use an existing driver.pickle
        if os.path.isdir(args.folder):
            if os.path.exists(os.path.join(args.folder, 'driver.pickle')):
                # if the driver.pickle exists, and param files were given, 
                # use the driver.pickle, and notify the user
                old_params = args.params
                args.params = os.path.join(args.folder, 'driver.pickle')
                if old_params is not None:
                    logger.warning("Appending to an existing folder: using the "
                                   "driver.pickle instead of %s" %old_params)
            else:
                if args.params is None:
                    raise rsd_io.ConfigurationError(
                        "The requested output folder seems empty. "
                        "You must then provide a parameter file (command"
                        " line option -p any.param)")
        else:
            if args.params is None:
                raise rsd_io.ConfigurationError(
                    "The requested output folder appears to be non "
                    "existent. You must then provide a parameter file "
                    "(command line option -p any.param)")
            else:
                os.makedirs(args.folder)

    return args
