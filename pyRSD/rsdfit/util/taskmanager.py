import logging
import os
import sys
import traceback
import tempfile
import collections
from string import Formatter

from pyRSD.rsdfit.parameters import ParameterSet
from pyRSD.rsdfit.util import rsdfit_parser, rsd_io
from pyRSD.rsdfit import rsdfit, params_filename
from mpi4py import MPI

#------------------------------------------------------------------------------
# tools
#------------------------------------------------------------------------------        
def split_ranks(N_ranks, N):
    """
    Divide the ranks into chunks, attempting to have `N` ranks
    in each chunk. This removes the master (0) rank, such 
    that `N_ranks - 1` ranks are available to be grouped
    
    Parameters
    ----------
    N_ranks : int
        the total number of ranks available
    N : int
        the desired number of ranks per worker
    """
    available = list(range(1, N_ranks)) # available ranks to do work
    total = len(available)
    extra_ranks = total % N
  
    for i in range(total//N):
        yield i, available[i*N:(i+1)*N]
    
    if extra_ranks and extra_ranks >= N//2:
        remove = extra_ranks % 2 # make it an even number
        ranks = available[-extra_ranks:]
        if remove: ranks = ranks[:-remove]
        if len(ranks):
            yield i+1, ranks
        
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
    
                    
def replacements_from_file(value):
    """
    Provided an existing file name, read the file into 
    a dictionary. The keys are interpreted as the string format name, 
    and the values are a list of values to iterate over for each job
    """
    if not os.path.exists(value):
        raise RuntimeError("for `replacements_from_file`, file `%s` does not exist" %value)
    
    toret = {}
    with open(value) as f:
        code = compile(f.read(), value, 'exec')
        exec(code, globals(), toret)

    return toret
        
def tasks_parser(value):
    """
    Given a string of the format ``key tasks``, split the string and then
    try to parse the ``tasks``, by first trying to evaluate it, and then
    simply splitting it and interpreting the results as the tasks. 
    
    The general use cases are: 
    
    1) "box: range(2)" -> key = `box`, tasks = `[0, 1]`
    2) "box: ['A', 'B' 'C']" -> key = `box`, tasks = `['A', 'B', 'C']`
    """
    import yaml
    
    try:
        fields = yaml.load(value)
        keys = list(fields.keys())
        if len(fields) != 1:
            raise Exception
    except:
        raise ValueError("specify iteration tasks via the format: ``-i key: [task1, task2]``")
    
    key = keys[0]
    if isinstance(fields[key], list):
        parsed = fields[key]
    else:
        # try to eval into a list
        try:
            parsed = eval(fields[key])
            if not isinstance(parsed, list):
                raise ValueError("result of `eval` on iteration string should be list" %(fields[key]))
        except:
            raise ValueError("tried but failed to `eval` iteration string `%s`" %(fields[key]))

    return [key, parsed]

#------------------------------------------------------------------------------
# task manager
#------------------------------------------------------------------------------
class TaskManager(object):
    """
    Task manager for running a set of `Algorithm` computations,
    possibly in parallel using MPI
    """
    logger = logging.getLogger('TaskManager')
    
    def __init__(self, comm, 
                       rsdfit_cmd, 
                       template, 
                       cpus_per_worker, 
                       task_dims, 
                       task_values, 
                       log_level=logging.INFO, 
                       extras={}, 
                       theory_updates={}):
        """
        Parameters
        ----------
        comm : MPI communicator
            the global communicator that will be split and divided
            amongs the independent workers
        rsdfit_cmd : str
            the string specifying the `rsdfit` command to run; this can
            contain string formatting keys
        template : str
            the name of the file holding the template parameter file, which
            will be updated for each task that is performed
        cpus_per_worker : int
            the desired number of ranks assigned to each independent
            worker, when iterating over the tasks in parallel
        task_dims : list
            a list of strings specifying the names of the task dimensions -- 
            these specify the string formatting key when updating the config
            template file for each task value
        task_value : list
            a list of tuples specifying the task values which will be iterated 
            over -- each tuple should be the length of `task_dims`
        log_level : int, optional
            an integer specifying the logging level to use -- default
            is the `INFO` level
        extras : dict, optional
            a dictionary where the values are lists of string replacements, with
            length equal to the total number of tasks -- if the keys are present
            in the config file, the string formatting will update the config
            file with the `ith` element of the list for the `ith` iteration
        theory_updates : dict, optional
            similar to `extras`, but for each task, the dictionary gives the
            value of fiducial theory parameters to update
        """
        self.logger.setLevel(log_level)
        
        self.rsdfit_cmd      = rsdfit_cmd
        self.cpus_per_worker = cpus_per_worker
        self.task_dims       = task_dims
        self.task_values     = task_values
        self.extras          = extras
        self.theory_updates  = theory_updates
        
        # read the template file and determine which params get updated
        tags = ['driver', 'data', 'theory', 'theory_extra', 'model']
        self.template = ParameterSet.from_file(template, tags=tags)
        self.determine_template_kwargs()
        
        # MPI setup
        self.comm      = comm
        self.size      = comm.size
        self.rank      = comm.rank
        self.pool_comm = None
        
        # the parser
        self.parser = rsdfit_parser()
                
    def initialize_driver(self):
        """
        Initialize the `RSDFitDriver` object on all ranks
        """
        # update with first value
        itask = 0
        task = self.task_values[itask]

        # master will parse the args
        this_config = None; rsdfit_cmd = None
        if self.comm.rank == 0:
                  
            # copy the template parameters
            params = self.copy_template()
            
            # initialize a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as ff:
                
                this_config = ff.name
                self.logger.debug("creating temporary file: %s" %this_config)
                
                # get the task kwargs
                task_kwargs = self.get_task_kwargs(itask)
                        
                # write to file
                self.write_parameter_file(params, ff, task_kwargs)

            # update the rsdfit options
            rsdfit_cmd = self.make_rsdfit_cmd(task_kwargs)
        
        # bcast the file name to all in the worker pool
        this_config = self.comm.bcast(this_config, root=0)
        rsdfit_cmd = self.comm.bcast(rsdfit_cmd, root=0)
        self.temp_config = this_config

        # get the args
        options = rsdfit_cmd + ['-p', this_config]
        self.logger.debug("calling rsdfit with arguments: %s" %str(options))
        
        args = None
        if self.comm.size > 1:
            if self.comm.rank == 0:
                args = self.parser.parse_args(options)
            args = self.comm.bcast(args, root=0)
        else:
            args = self.parser.parse_args(options)
                      
        # load the driver for everyone but root 
        if self.comm.rank != 0:     
            args = vars(args)
            mode = args.pop('subparser_name')
            self.driver = rsdfit.RSDFitDriver(self.pool_comm, mode, **args)
        
    def copy_template(self):
        """
        Return a copy of the template parameters
        """
        params = collections.OrderedDict()
        for tag in self.template:
            params[tag] = self.template[tag].copy()
        return params
        
    def make_rsdfit_cmd(self, task_kwargs):
        """
        Return the updated `rsdfit` cmd, using the input `task_kwargs`
        """
        valid = {k:task_kwargs[k] for k in task_kwargs if k in self.rsdfit_cmd_kwargs}
        return self.rsdfit_cmd.format(**valid).split()
    
    def determine_template_kwargs(self):
        """
        Check the `driver` and `data` template parameters for 
        string replacements, and store them
        
        Also, check if we need to update the command-line options
        """
        formatter = Formatter()
        
        self.template_kwargs = {}
        for key in ['driver', 'data']:
            pars = self.template[key]
            for name in pars:
                p = pars[name].value
                if isinstance(p, str):
                    kwargs = [kw for _, kw, _, _ in formatter.parse(p) if kw]
                    if len(kwargs):
                        self.template_kwargs[(key, name)] = kwargs
                        
        self.rsdfit_cmd_kwargs = {}
        kwargs = [kw for _, kw, _, _ in formatter.parse(self.rsdfit_cmd) if kw]
        if len(kwargs):
            self.rsdfit_cmd_kwargs = kwargs
        
    def write_parameter_file(self, params, ff, task_kwargs):
        """
        Write the parameters to the temporary file
        """
        # update the values        
        for key in self.template_kwargs:
            tag, name = key
            v = params[tag][name].value
            kwargs = self.template_kwargs[key]
            
            # do the string formatting if the key is present in template
            valid = {k:task_kwargs[k] for k in task_kwargs if k in kwargs}
            params[tag][name].value = v.format(**valid)
        
        # update any fiducial values
        for k in task_kwargs:
            v = task_kwargs[k]
            name = '%s_%s' %(k, str(v))
            if name in self.theory_updates:
                toupdate = self.theory_updates[name]
                for name in toupdate:                
                    if name not in params['theory']:
                        raise ValueError("cannot update value of theory parameter '%s'; does not exist" %name)
                    params['theory'][name].value = toupdate[name]
                    params['theory'][name].fiducial = toupdate[name]
            
        # write to file
        for tag in params:
            p = params[tag]
            as_dict = True if tag == 'theory' else False
            p.to_file(ff, as_dict=as_dict)
        
    @classmethod
    def create(cls, comm=None, desc=None):
        """
        Parse the task manager and return the ``TaskManager`` instance
        """
        import inspect 
        
        if comm is None: comm = MPI.COMM_WORLD
        args_dict = cls.parse_args(desc)
        args_dict['comm'] = comm
        template = args_dict.pop('params')
        args_dict['template'] = template
        
        # inspect the __init__ function
        args, varargs, varkw, defaults = inspect.getargspec(cls.__init__)
        
        # determine the required arguments
        args = args[1:] # remove 'self'
        if defaults:
            required = args[:-len(defaults)]
        else:
            required = args
            
        # get the args, kwargs to pass to __init__
        fargs = tuple(args_dict[p] for p in required)
        fkwargs = {}
        if defaults:
            for i, p in enumerate(defaults):
                name = args[-len(defaults)+i]
                fkwargs[name] = args_dict.get(name, defaults[i])
        
        return cls(*fargs, **fkwargs)
        
    @classmethod
    def parse_args(cls, desc=None):
        """
        Parse command-line arguments that are needed to initialize a 
        `TaskManager` class
        
        Parameters
        ----------
        desc : str, optional
            the description of to use for this parser
        """
        import argparse
        import itertools
        
        # parse
        parser = argparse.ArgumentParser(description=desc) 
                
        # the number of independent workers
        h = """the desired number of ranks assigned to each independent
                worker, when iterating over the tasks in parallel""" 
        parser.add_argument('cpus_per_worker', type=int, help=h)
    
        # now do the required named arguments
        required_named = parser.add_argument_group('required named arguments')
        
        # specify the tasks along one dimension 
        h =  """given a string of the format ``key: tasks``, split the string and then
                try to parse the ``tasks``, by first trying to evaluate it, and then
                simply splitting it and interpreting the results as the tasks. 
        
                The general use cases are: 
        
                1) "box: range(2)" -> key = `box`, tasks = `[0, 1]`
                2) "box: [A, B, C]" -> key = `box`, tasks = `['A', 'B', 'C']`
                
                If multiple options passed with `-i` flag, then the total tasks to 
                perform will be the product of the tasks lists passed"""
        required_named.add_argument('-i', dest='tasks', action='append', 
                type=tasks_parser, required=True, help=h)
    
        # the template config file
        h = """the name of the template parameter file that provides the 
                `rsdfit` parameters; the file should use  ``string.format`` 
                syntax to indicate which variables will be updated for each task, 
                i.e., an input file could be specified as 'input/box{box}.dat', 
                if `box` were one of the task dimensions"""
        required_named.add_argument('-p', '--params', required=True, type=str, help=h)
    
        # the rsdfit command
        h = 'the command to pass to rsdfit, as a single string'
        required_named.add_argument('-cmd', '--rsdfit_cmd', required=True, type=str, help=h)
        
        # read any extra string replacements from file
        h = """file providing extra string replaces, with lines of the form 
                 `tag = ['tag1', 'tag2']`; if the keys match keywords in the 
                 template param file, the file with be updated with
                 the `ith` value for the `ith` task"""
        parser.add_argument('--extras', dest='extras', default={}, type=replacements_from_file, help=h)
        
        # read any parameter value replacements from file
        h = """file providing parameter values stored by key to update on each iteration,
                i.e., ``box_1 = {'nbar':3e-4}``"""
        parser.add_argument('-th', '--theory_updates', default={}, type=replacements_from_file, help=h)
    
        h = "set the logging output to debug, with lots more info printed"
        parser.add_argument('--debug', help=h, action="store_const", dest="log_level", 
                            const=logging.DEBUG, default=logging.INFO)
                                
        args = parser.parse_args()
        
        # format the tasks, taking the product of multiple task lists
        keys = []; values = []
        for [key, tasks] in args.tasks:
            keys.append(key)
            values.append(tasks)

        # take the product
        if len(keys) > 1:
            values = list(itertools.product(*values))
        else:
            values = values[0]
            
        # save
        args.task_dims = keys
        args.task_values = values
        
        return vars(args)
    
    def initialize_pool_comm(self):
        """
        Internal function that initializes the `MPI.Intracomm` used by the 
        pool of workers. This will be passed to the task function and used 
        in task computation
        """
        # split the ranks
        self.pool_comm = None
        chain_ranks = []
        color = 0
        total_ranks = 0
        i = 0
        for i, ranks in split_ranks(self.size, self.cpus_per_worker):
            chain_ranks.append(ranks[0])
            if self.rank in ranks: color = i+1
            total_ranks += len(ranks)
        
        self.workers = i+1 # store the total number of workers
        leftover= (self.size - 1) - total_ranks
        if leftover and self.rank == 0:
            args = (self.cpus_per_worker, self.size-1, leftover)
            self.logger.warning("with `cpus_per_worker` = %d and %d available ranks, %d ranks will do no work" %args)
            
        # crash if we only have one process or one worker
        if self.size <= self.workers:
            args = (self.size, self.workers+1, self.workers)
            raise ValueError("only have %d ranks; need at least %d to use the desired %d workers" %args)
            
        # ranks that will do work have a nonzero color now
        self._valid_worker = color > 0
        
        # track how many tasks each worker does
        if self._valid_worker:
            self._completed_tasks = 0
        
        # split the comm between the workers
        self.pool_comm = self.comm.Split(color, 0)
                
    def run_all(self):
        """
        Run all of the tasks
        """    
        # define MPI message tags
        tags = enum('READY', 'DONE', 'EXIT', 'START')
        status = MPI.Status()
         
        try:
            # make the pool comm
            self.initialize_pool_comm()
    
            # the total numbe rof tasks
            num_tasks = len(self.task_values)
            
            # initialize the driver for everyone but master
            self.initialize_driver()
    
            # master distributes the tasks
            if self.rank == 0:
        
                # initialize
                task_index = 0
                closed_workers = 0
        
                # loop until all workers have finished with no more tasks
                self.logger.info("master starting with %d worker(s) with %d total tasks" %(self.workers, num_tasks))
                while closed_workers < self.workers:
                    data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    source = status.Get_source()
                    tag = status.Get_tag()
            
                    # worker is ready, so send it a task
                    if tag == tags.READY:
                        if task_index < num_tasks:
                            self.comm.send(task_index, dest=source, tag=tags.START)
                            self.logger.info("sending task `%s` to worker %d" %(str(self.task_values[task_index]), source))
                            task_index += 1
                        else:
                            self.comm.send(None, dest=source, tag=tags.EXIT)
                    elif tag == tags.DONE:
                        results = data
                        self.logger.debug("received result from worker %d" %source)
                    elif tag == tags.EXIT:
                        closed_workers += 1
                        self.logger.debug("worker %d has exited, closed workers = %d" %(source, closed_workers))
    
            # worker processes wait and execute single jobs
            elif self._valid_worker:
                if self.pool_comm.rank == 0:
                    args = (self.rank, MPI.Get_processor_name(), self.pool_comm.size)
                    self.logger.info("pool master rank is %d on %s with %d processes available" %args)
                while True:
                    itask = -1
                    tag = -1
        
                    # have the master rank of the pool ask for task and then broadcast
                    if self.pool_comm.rank == 0:
                        self.comm.send(None, dest=0, tag=tags.READY)
                        itask = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                        tag = status.Get_tag()
                    itask = self.pool_comm.bcast(itask)
                    tag = self.pool_comm.bcast(tag)
        
                    # do the work here
                    if tag == tags.START:
                        result = self.run_rsdfit(itask)
                        self.pool_comm.Barrier() # wait for everyone
                        if self.pool_comm.rank == 0:
                            self.comm.send(result, dest=0, tag=tags.DONE) # done this task
                    elif tag == tags.EXIT:
                        break

                self.pool_comm.Barrier()
                if self.pool_comm.rank == 0:
                    self.comm.send(None, dest=0, tag=tags.EXIT) # exiting
        except Exception as e:
            self.logger.error("an exception has occurred on one of the ranks...all ranks exiting")
            self.logger.error(traceback.format_exc())
            
            # bit of hack that forces mpi4py to exit all ranks
            # see https://groups.google.com/forum/embed/#!topic/mpi4py/RovYzJ8qkbc
            os._exit(1)
        
        finally:
            # free and exit
            self.logger.debug("rank %d process finished" %self.rank)
            self.comm.Barrier()
            
            if self.rank == 0:
                self.logger.info("master is finished; terminating")
                if self.pool_comm is not None:
                    self.pool_comm.Free()
          
                if os.path.exists(self.temp_config): 
                    self.logger.debug("removing temporary file: %s" %self.temp_config)
                    os.remove(self.temp_config)
            
            
    def get_task_kwargs(self, itask):
        """
        This returns the keyword arguments that are possible for 
        this task
        """
        task = self.task_values[itask]
                  
        # key/values for this task 
        if len(self.task_dims) == 1:
            task_kwargs = {self.task_dims[0] : task}
        else:
            task_kwargs = dict(zip(self.task_dims, task))
            
        # any extra key/value pairs for this tasks
        if self.extras is not None:
            for k in self.extras:
                task_kwargs[k] = self.extras[k][itask]
                
        return task_kwargs
    
    def run_rsdfit(self, itask):
        """
        Run the algorithm once, using the parameters specified for this task
        iteration specified by `itask`
    
        Parameters
        ----------
        itask : int
            the integer index of this task
        """
        # get the possible keywords for this task
        task_kwargs = self.get_task_kwargs(itask)
                    
        # update the rsdfit options
        rsdfit_cmd = self.make_rsdfit_cmd(task_kwargs)
        
        # update the attributes of the RSDFitDriver
        args = vars(self.parser.parse_args(rsdfit_cmd + ['-p', self.temp_config]))
        for k in args:
            setattr(self.driver, k, args[k])
            
        # make a copy of the template args
        params = self.copy_template()
        
        # update data/driver and theory values        
        update_data = False
        for key in self.template_kwargs:
            tag, name = key
            v = params[tag][name].value
            kwargs = self.template_kwargs[key]
            
            # do the string formatting if the key is present in template
            valid = {k:task_kwargs[k] for k in task_kwargs if k in kwargs}
            v = v.format(**valid)
            
            if tag == 'driver':
                self.driver.algorithm.params.add(name, value=v)
            elif tag == 'data':
                self.driver.algorithm.data.params.add(name, value=v)
                update_data = True
                
        # re-initialize the data
        if update_data:
            self.driver.algorithm.data.initialize()
            
        # update any fiducial values
        fit_params = self.driver.algorithm.theory.fit_params
        update_model = False
        for k in task_kwargs:
            v = task_kwargs[k]
            name = '%s_%s' %(k, str(v))
            if name in self.theory_updates:
                toupdate = self.theory_updates[name]
                for name in toupdate:                
                    if name not in params['theory']:
                        raise ValueError("cannot update value of theory parameter '%s'; does not exist" %name)
                    
                    fit_params[name].value = toupdate[name]
                    fit_params[name].fiducial = toupdate[name]
                    update_model = True
        
        # update the model values        
        if update_model:
            fit_params.update_values()
            self.driver.algorithm.theory.update_model()
        
        # write the parameters to file
        filename = os.path.join(self.driver.folder, params_filename)
        self.driver.algorithm.to_file(filename)
            
        # okay, now run
        self.driver.run()