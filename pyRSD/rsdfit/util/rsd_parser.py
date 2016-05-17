from . import rsd_io
from .. import logging, model_filename, params_filename
from ... import os

import argparse as ap
import textwrap as tw

logger = logging.getLogger('rsdfit.parser')

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
        msg = "the file '{}' does not exist".format(fname)
        raise ap.ArgumentTypeError(msg)

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
        raise ap.ArgumentTypeError("argument requires a positive integer")
            

def setup_run_subparser(parent):
    """
    Setup the subparser for the ``run`` subcommand
    """
    subparser = parent.add_parser('run', help="run the MCMC chains from scratch")
    
    # the path to the model to read
    h = 'file name holding the model path'
    kwargs = {'dest':'model', 'type':existing_file, 'help':h}
    subparser.add_argument('-m', '--model', **kwargs)
    
    # the general driver parameters
    h = 'file name holding the driver, theory, and data parameters'
    kwargs = {'dest':'params', 'type':existing_file, 'help':h}
    subparser.add_argument('-p', '--params', **kwargs) 
    
    h = 'file name holding the names of any extra theory parameters'
    kwargs = {'dest':'extra_params', 'type':existing_file, 'help':h}
    subparser.add_argument('-xp', '--extra_params', **kwargs)
        
    # silence the output (OPTIONAL)
    h = 'silence the standard output to the console'
    subparser.add_argument('--silent', help=h, action='store_true')
    
    # number of walkers (OPTIONAL)
    h = 'number of walkers in the chain'
    subparser.add_argument('-w', help=h, type=positive_int, dest='walkers')
    
    # number of iterations (OPTIONAL)
    h = 'number of steps in the chain to run'
    subparser.add_argument('-i', help=h, type=positive_int, dest='iterations')
    
    # number of chains to run concurrently
    h = 'number of chains to run concurrently'
    subparser.add_argument('-n', '--nchains', help=h, type=positive_int, default=1)
    
    # the output folder
    h = 'the folder where the results will be written'
    kwargs = {'help':h, 'type':str, 'required':True, 'dest':'folder'}
    subparser.add_argument('-o', '--output', **kwargs)
        
    # debug
    h = 'whether to print more info about the mpi4py.Pool object'
    subparser.add_argument('--debug', help=h, action='store_true', default=False)
    
def setup_restart_subparser(parent):
    """
    Setup the subparser for the ``restart`` subcommand
    """
    h = "restart a MCMC fit from an existing chain"
    subparser = parent.add_parser('restart', help=h)
    
    # the path to the model to read
    h = 'file name holding the model path'
    kwargs = {'dest':'model', 'type':existing_file, 'help':h}
    subparser.add_argument('-m', '--model', **kwargs)
    
    # the general driver parameters (REQUIRED)
    h = 'the name of the existing results file to restart from'
    subparser.add_argument('restart_files', type=existing_file, nargs="+", help=h)
    
    # number of iterations (REQUIRED)
    h = 'the number of additional steps to run using the old chain'
    subparser.add_argument('-i', help=h, required=True, type=positive_int, dest='iterations')
                                
    # number of iterations (REQUIRED)
    h = 'the number of steps to consider burnin'
    subparser.add_argument('-b', help=h, type=positive_int, dest='burnin')
                                    
    h = 'silence the standard output to the console'
    subparser.add_argument('--silent', help=h, action='store_true')
    
    # debug
    h = 'whether to print more info about the mpi4py.Pool object'
    subparser.add_argument('--debug', help=h, action='store_true', default=False)

def setup_analyze_subparser(parent):
    """
    Setup the subparser for the ``restart`` subcommand
    """
    h = "analyze the MCMC chains"
    kwargs = {'help':h, 'formatter_class':ap.ArgumentDefaultsHelpFormatter}
    subparser = parent.add_parser('analyze', **kwargs)

    # the folder to analyze
    h = "files to analyze: either a file(s), or a complete directory name"
    subparser.add_argument('files', help=h, nargs='+')
    
    # to only write the covmat and bestfit, without computing the posterior
    h = "use this flag to avoid computing the posterior distribution"
    subparser.add_argument('--minimal', help=h, action='store_true')
    
    # the number of bins (defaulting to 20)
    h = """number of bins in the histograms used to derive posterior 
           probabilities. Decrease this number for smoother plots at the 
           expense of masking details."""
    subparser.add_argument('--bins', help=h, type=int, default=20)
                                                 
    # to remove the mean-likelihood line
    h = "remove the mean likelihood from the 1D posterior plots"
    subparser.add_argument('--show-mean', help=h, dest='mean_likelihood', 
                                action='store_false', default=False)
    
    # if you just want the covariance matrix, use this option
    h = "do not produce any plot, simply compute the posterior"
    subparser.add_argument('--noplot', help=h, dest='plot', 
                                action='store_false')
                            
    # if you just want to output 1d posterior distributions (faster)
    h = "produce only the 1d posterior plot"
    subparser.add_argument('--noplot-2d', help=h, dest='plot_2d', 
                                action='store_false')
    
    # don't include fiducial lines on 1D posterior plots
    h = "don't include fiducial lines on 1D posterior plots"
    subparser.add_argument('--show-fiducial', help=h, dest='show_fiducial', 
                                action='store_true', default=False)
                                
    # the fraction of samples to consider burnin
    h = 'the fraction of samples to consider burnin'
    subparser.add_argument('--burnin', '-b', type=float, help=h)
                            
    # when plotting 2d posterior distribution, use contours and not contours
    # filled (might be useful when comparing several folders)
    h = "do not fill the contours on the 2d plot"
    subparser.add_argument('--contours-only', help=h, dest='contours_only', 
                                action='store_true')
                                
    # if you want to output every single subplots
    h = "output every subplot and data in separate files"
    subparser.add_argument('--all', help=h, dest='subplot', 
                                action='store_true')
    
    # output file extension
    h = "change the extension for the output file"
    subparser.add_argument('--ext', help=h, type=str, dest='extension', 
                                default='pdf', choices=['pdf', 'png', 'eps'])
                                
    # -------------------------------------
    # fontsize of plots (defaulting to 16)
    h = 'the desired fontsize of output fonts'
    subparser.add_argument('--fontsize', help=h, type=int, default=16)
    
    # ticksize of plots (defaulting to 14)
    h = 'the tick size on the plots'
    subparser.add_argument('--ticksize', help=h, type=int, default=14)
    
    # linewidth of 1d plots (defaulting to 4, 2 being a bare minimum for
    # legible graphs
    h = 'the linewidth of 1d plots'
    subparser.add_argument('--line-width', help=h, type=int, default=4)
    
    # number of decimal places that appear on the tick legend. If you want
    # to increase the number of ticks, you should reduce this number
    h = "number of decimal places on ticks"
    subparser.add_argument('--decimal', help=h, type=int, default=3)
    
    # number of ticks that appear on the graph.
    h = "number of ticks on each axis"
    subparser.add_argument('--ticknumber', help=h, type=int, default=3)
    
    # thinning factor
    h = 'the thinning factor to use'
    subparser.add_argument('--thin', help=h, type=int, default=1)
    
    # thinning factor
    h = 'whether to rescale errors'
    subparser.add_argument('--rescale-errors', help=h, action='store_true')
                                
    # possible plot file describing custom commands
    h = """ extra file to customize the output plots. You can actually
        set all the possible options in this file, including line-width,
        ticknumber, ticksize, etc... You can specify four fields,
        `info.redefine` (dict with keys set to the previous variable, and
        the value set to a numerical computation that should replace this
        variable), `info.to_change` (dict with keys set to the old variable
        name, and value set to the new variable name), `info.to_plot` (list
        of variables with new names to plot), and `info.new_scales` (dict
        with keys set to the new variable names, and values set to the
        number by which it should be multiplied in the graph).
        For instance,
        .. code::
            analyze.to_plot=['name1','name2','newname3',...]
            analyze.new_scales={'name1':number1,'name2':number2,...}"""
    subparser.add_argument('--extra', help=h, dest='optional_plot_file', default='')
               
def rsdfit_parser():
    """
    Initialize the parser of command-line arguments. 
    
    The main parser has three subparsers, `run`, `restart`, and `analyze`. 
    If you run :code:`rsdfit -h`, this information will be printed to 
    the console. Further help information can be found by including the name 
    of a specific subcommand, i.e., run code:`rsdfit run -h`.
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
    
    # run subcommand
    setup_run_subparser(subparser)
    # restart subcommand
    setup_restart_subparser(subparser)
    # analyze subcommand
    setup_analyze_subparser(subparser)
                 
    # set the parse_args functions    
    def parse_known_args(parser, args=None, namespace=None):
        ns, unknown = ap.ArgumentParser.parse_known_args(parser, args=args, namespace=namespace)
        ns = verify_arguments(ns)
        return ns, unknown
    
    parser.parse_known_args = lambda *args, **kwargs: parse_known_args(parser, *args, **kwargs)
    return parser


def verify_arguments(ns):
    """
    Run a few quick verification tests on the supplied arguments
    """
    ## restart from existing
    if ns.subparser_name == 'restart':
        
        # if we are restarting, automatically use the same folder, 
        # and the driver.pickle
        ns.folder = os.path.sep.join(ns.restart_files[0].split(os.path.sep)[:-1])
        ns.params = os.path.join(ns.folder, params_filename)
        if not os.path.exists(ns.params):
            raise rsd_io.ConfigurationError("Restarting but associated `%s` doesn't exist" %params_filename)
        if ns.model is None:
            ns.model = os.path.join(ns.folder, model_filename)
            if not os.path.exists(ns.model):
                raise rsd_io.ConfigurationError("Restarting but cannot find existing model file to read")
        logger.warning("Restarting from %s and using associated params.dat" %ns.restart_files[0])
    
    ## run from new  
    elif ns.subparser_name == "run":

        # if the folder already exists, and no parameter files were specified
        # try to use an existing params.dat
        if os.path.isdir(ns.folder):
            params_path = os.path.join(ns.folder, params_filename)
            model_path = os.path.join(ns.folder, model_filename)
            if os.path.exists(params_path):
                # if the params.dat exists, and param files were given, 
                # use the params.dat, and notify the user
                if ns.params is not None:
                    logger.warning("Appending to an existing folder: using the "
                                   "existing `%s` instead of `%s`" %(params_filename, ns.params))
                ns.params = params_path
            else:
                if ns.params is None:
                    raise rsd_io.ConfigurationError(
                        "The requested output folder seems empty. "
                        "You must then provide a parameter file (command"
                        " line option -p any.param)")
                        
            # also check for existing model file now
            if os.path.exists(model_path):
                if ns.model is None:
                    ns.model = model_path
        else:
            if ns.params is None:
                raise rsd_io.ConfigurationError(
                    "The requested output folder appears to be non "
                    "existent. You must then provide a parameter file "
                    "(command line option -p any.param)")
            else:
                os.makedirs(ns.folder)

    # create a logs directory if it doesn't exist
    if hasattr(ns, 'folder'):
        log_dir = os.path.join(ns.folder, 'logs')
        if not os.path.exists(log_dir): 
            os.makedirs(log_dir)
    
    return ns
