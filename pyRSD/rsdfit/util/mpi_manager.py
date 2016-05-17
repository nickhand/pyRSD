import traceback 
from ... import os, sys
from .. import logging

def split_ranks(N_ranks, N_chunks):
    """
    Divide the ranks into N chunks
    """
    seq = range(N_ranks)
    avg = int(N_ranks // N_chunks)
    remainder = N_ranks % N_chunks

    start = 0
    end = avg
    for i in range(N_chunks):
        if remainder:
            end += 1
            remainder -= 1
        yield i, seq[start:end]
        start = end
        end += avg
        
class MPIManager(object):
    """
    Class to serve as context manager to handle to MPI-related issues, 
    specifically, the managing of ``MPIPool`` and splitting of communicators
    """
    logger = logging.getLogger("MPIManager")
    
    def __init__(self, comm, nruns, debug=False):
        """
        Parameters
        ----------
        comm : MPI.Communicator
            the global communicator to split
        nruns : int
            the number of independent algorithms to run concurrently
        debug : bool, optional
            set the logging level to debug in the `MPIPool`; default
            is `False`
        """
        self.comm  = comm
        self.nruns = nruns
        self.debug = debug
        if debug: self.logger.setLevel(logging.DEBUG)
    
        # initialize comm for parallel runs
        self.par_runs_group = None
        self.par_runs_comm  = None
        
        # intiialize comm for pool of workers for each 
        # parallel run
        self.pool_comm = None
        self.pool      = None
    
    def __enter__(self):
        """
        Setup the MPIPool, such that only the ``pool`` master returns, 
        while the other processes wait for tasks
        """
        # split ranks if we need to
        if self.comm.size > 1:
            
            ranges = []
            for i, ranks in split_ranks(self.comm.size, self.nruns):
                ranges.append(ranks[0])
                if self.comm.rank in ranks: color = i
        
            # split the global comm into pools of workers
            self.pool_comm = self.comm.Split(color, 0)
            
            # make the comm to communicate b/w parallel runs
            if self.nruns > 1:
                self.par_runs_group = self.comm.group.Incl(ranges)
                self.par_runs_comm = self.comm.Create(self.par_runs_group)
    
        # initialize the MPI pool, if the comm has more than 1 process
        if self.pool_comm is not None and self.pool_comm.size > 1:
            from emcee.utils import MPIPool
            kws = {'loadbalance':True, 'comm':self.pool_comm, 'debug':self.debug}
            self.pool = MPIPool(**kws)
                    
        # explicitly force non-master ranks in pool to wait
        if self.pool is not None and not self.pool.is_master():
            self.pool.wait()
            self.logger.debug("exiting after pool closed")
            sys.exit(0)
            
        # log
        if self.pool is not None:
            self.logger.debug("using an MPIPool instance with %d worker(s)" %self.pool.size)
        
        self.rank = 0
        if self.par_runs_comm is not None:
            self.rank = self.par_runs_comm.rank
            
        return self
                
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit gracefully by closing and freeing the MPI-related variables
        """
        if exc_value is not None:
            trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback, limit=5))
            self.logger.error("traceback:\n%s" %trace)
        
        # wait for all the processes, if we more than one
        if self.par_runs_comm is not None and self.par_runs_comm.size > 1:
            self.par_runs_comm.Barrier()
            
        # close and free the MPI stuff
        self.logger.debug("beginning to close MPI variables...")
        
        if self.par_runs_group is not None:
            self.par_runs_group.Free()
        if self.par_runs_comm is not None:
            self.par_runs_comm.Free()
        if self.pool is not None:
            self.pool.close()
        self.logger.debug('...MPI variables closed')

        return True
