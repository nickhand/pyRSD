from ... import os
from datetime import date
import cPickle
import fcntl
import copy_reg
import types

class PickeableClass(type):
    
    def __init__(cls, name, bases, attrs):
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)
 
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def load_pickle(filename):
    """
    Load an instance, i.e., `FittingDriver`, `EmceeResults` that has been 
    pickled
    """
    try:
        return cPickle.load(open(filename, 'r'))
    except Exception as e:
        raise ConfigurationError("Cannot load the pickle `%s`; original message: %s" %(filename, e))
    
def save_pickle(obj, filename):
    """
    Pickle an instance using `cPickle`
    """
    # make sure pool is None, so it is pickable
    cPickle.dump(obj, open(filename, 'w'))
    
#-------------------------------------------------------------------------------
def create_output_file(args, solver_type, walkers, iterations):
    """
    Automatically create a new name for the results file.
    
    This routine takes care of organizing the folder for you. It will
    automatically generate names for the new chains according to the date,
    number of points chosen.
    """
    subparser = args.subparser_name
    
    if solver_type == 'emcee':
        tag = "{}x{}".format(walkers, iterations)
    else:
        tag = solver_type
        
    # output file
    outname_base = '{0}_{1}__'.format(date.today(), tag)
    suffix = 0
    trying = True
    if args.chain_number is None:
        for files in os.listdir(args.folder):
            if files.find(outname_base) != -1:
                if int(files.split('__')[-1].split('.')[0]) > suffix:
                    suffix = int(files.split('__')[-1].split('.')[0])
        suffix += 1
        while trying:
            outfile = open(os.path.join(
                        args.folder, outname_base)+str(suffix)+'.pickle', 'w')
            try:
                lock(outfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
                trying = False
            except LockError:
                suffix += 1
        outfile.close()
        outfile_name = os.path.join(args.folder, outname_base)+str(suffix)+'.pickle'
        print 'Creating %s\n' %outfile_name
        
    else:
        outfile_name = os.path.join(args.folder, outname_base)+args.chain_number+'.pickle'
        print 'Creating %s\n' %outfile_name
     
    # touch the file so it exists and then return
    open(outfile_name, 'a').close()   
    return outfile_name

#-------------------------------------------------------------------------------
def lock(file, flags):
    """
    Lock a given file to prevent other instances of the code to write to the
    same file.
    .. warning::
        in the process of being tested
    """
    import fcntl
    try:
        fcntl.flock(file.fileno(), flags)
    except IOError as exc_value:
        # The exception code varies on different systems so we'll catch
        # every IO error
        raise LockError(*exc_value)


#-------------------------------------------------------------------------------
class ConfigurationError(Exception):
    """Missing files, parameters, etc..."""
    pass


class AnalyzeError(Exception):
    """Used when encountering a fatal mistake in analyzing chains"""
    pass

class LockError(Exception):
    # Error codes:
    LOCK_FAILED = 1