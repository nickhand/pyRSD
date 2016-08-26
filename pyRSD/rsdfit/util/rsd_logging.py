import tempfile
import traceback
from .. import logging
from ... import os, sys

class Filter(logging.Filter):
    def filter(self, record):
        on = getattr(record, 'on', None)
        comm = getattr(record, 'comm', None)
        if on is None or on == comm.rank:
            return True
        return False

class MPILoggerAdapter(logging.LoggerAdapter):
    """ 
    A logger adapter that allows a single rank to log. 
        
    The log method takes additional arguments:
    
        on : None or int
            rank to record the message
        comm : communicator 
            (defaults to MPI.COMM_WORLD)
    """
    def __init__(self, obj):
        logging.LoggerAdapter.__init__(self, obj, {})
        obj.addFilter(Filter())

    def process(self, msg, kwargs):
        if 'mpi4py' in sys.modules:
            from mpi4py import MPI 
            on   = kwargs.pop('on', None)
            comm = kwargs.pop('comm', MPI.COMM_WORLD)
            hostname = MPI.Get_processor_name()
            format='rank %(rank)d on %(hostname)-12s '
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            d = kwargs['extra']
            d['on'] = on
            d['comm'] = comm
            d['rank'] = comm.rank
            d['hostname'] = hostname.split('.')[0]
            return ((format % d) + msg, kwargs)
        else:
            on   = kwargs.pop('on', None)
            return (msg, kwargs)
    def setLevel(self, level):
        self.logger.setLevel(level)


class FileLogger(object):
    """
    Class to serve as context manager for running multiple chains, which
    will handle exceptions (user-supplied or otherwise) and convergence
    criteria from multiple chains
    """
    logger = logging.getLogger("FileLogger")
        
    def __init__(self, rank, debug=False):
        self.rank = rank
        self.debug = debug
        if debug: self.logger.setLevel(logging.DEBUG)
        
        self.output_name = None
        self.restart     = None
        self.exception   = True
        
        
    def __enter__(self):
        
        # log to a temporary file (for now)
        log = tempfile.NamedTemporaryFile(delete=False)
        self.name = log.name
        log.close()
        
        # define a Handler which writes DEBUG messages or higher file
        f = logging.FileHandler(self.name, mode='w')
        f.setLevel(logging.DEBUG)
        formatter = logging.Formatter('rank #%d: '%self.rank + '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        f.setFormatter(formatter)
        logging.getLogger('').addHandler(f)
        
        return self
                
    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        if exc_value is not None:
            self.exception = True
            trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback, limit=5))
            self.logger.error("traceback:\n%s" %trace)  
        
        try:
            # now save the log to the logs dir
            copy_log(self.name, self.output_name, restart=self.restart)
        except Exception as e:
            pass
            
            
        # if we made it this far, it's safe to delete the old results
        try:
            if os.path.exists(self.name):
                self.logger.info("removing temporary logging file: `%s`" %self.name)
                os.remove(self.name)
            
            # only remove if we have no exceptions
            if exc_value is None:
                if self.restart is not None and os.path.exists(self.restart):
                    self.logger.info("removing original restart file: `%s`" %self.restart)
                    os.remove(self.restart)
        except Exception as e:
            pass
        
        return True

def copy_log(temp_log_name, output_name, restart=None):
    """
    Copy the contents of the log from the temporary log file    
    """
    output_dir, fname = output_name.rsplit(os.path.sep, 1)
    log_dir = os.path.join(output_dir, 'logs')
    
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
            
def add_console_logger(rank):
    """
    Add a logger that logs to the console at level `INFO`.
    """ 
    # set up console logging
    logging.basicConfig(level=logging.INFO,
                        format='rank #%d: '%rank + '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
