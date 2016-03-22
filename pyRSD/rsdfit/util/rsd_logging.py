import tempfile
import traceback
from .. import logging
from ... import os

class FileLogger(object):
    """
    Class to serve as context manager for running multiple chains, which
    will handle exceptions (user-supplied or otherwise) and convergence
    criteria from multiple chains
    """
    def __init__(self, comm, rank):
        self.rank = rank
        self.comm = comm
        
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
            logging.error("traceback:\n%s" %trace)  
            
        try:
            # setup the log directory
            if self.comm.rank == 0:
                output_dir, fname = self.output_name.rsplit(os.path.sep, 1)
                log_dir = os.path.join(output_dir, 'logs')
                if not os.path.exists(log_dir): 
                    os.makedirs(log_dir)
                
            # make sure the logs directory is made
            self.comm.Barrier()
            
            # now save the log to the logs dir
            copy_log(self.name, self.output_name, restart=self.restart)
        except Exception as e:
            pass
            
            
        # if we made it this far, it's safe to delete the old results
        try:
            if os.path.exists(self.name):
                logging.debug("removing temporary logging file: `%s`" %self.name)
                os.remove(self.name)
            if self.restart is not None and os.path.exists(self.restart):
                logging.debug("removing original restart file: `%s`" %self.restart)
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
