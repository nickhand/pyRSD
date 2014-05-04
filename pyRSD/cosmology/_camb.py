import shutil, tempfile
import sys, os
import subprocess
from contextlib import contextmanager
import numpy as np

#-------------------------------------------------------------------------------
@contextmanager
def ignored(*exceptions):
    """
    Return a context manager that ignores the specified expections if they
    occur in the body of a with-statement.
    
    For example::
    from contextlib import ignored
    
    with ignored(OSError):
        os.remove('somefile.tmp')
    
    This code is equivalent to:
        try:
            os.remove('somefile.tmp')
        except OSError:
            pass
    
    This will be in python 3.4
    """
    try:
        yield
    except exceptions:
        pass
#-------------------------------------------------------------------------------
def get_camb_params(input_params):
    
    s = "DEFAULT(%s/params.ini)\n" %os.environ['CAMB_DIR']
    for k, v in input_params.iteritems():
        s += "%s = %s\n" %(k, v)
        
    return s

#-------------------------------------------------------------------------------
def set_transfer_params(params):
    
    vals = {'get_scalar_cls' : 'T', 'get_vector_cls' : 'F', 'get_tensor_cls' : 'F',
            'get_transfer' : 'T', 'transfer_high_precision' : 'T', 'output_root' : 'test',
            're_delta_redshift' : 0.5}
            
    for k, v in vals.iteritems():
        params += "%s = %s\n" %(k, v)
    
    return params
#-------------------------------------------------------------------------------    
def camb_transfer(*args, **kwargs):
    """
    Compute the matter transfer function using CAMB with parameters specifed by 
    the input parameters.
    """
    if len(args) > 1 or (len(args) > 0 and not isinstance(args[0], dict)):
        raise TypeError("Single argument must be a dictionary.")
        
    if len(args) > 0:
        input_params = dict(args[0].items() + kwargs.items())
    else:
        input_params = kwargs
    
    # get the cosmo params in string form
    params = get_camb_params(input_params)
    
    # add on any transfer specific keywords
    params = set_transfer_params(params)
    
    # get the initial directory for later use
    initialDir = os.getcwd()
    try:
        # safely make a temporaty directory and cd there
        tmp_dir = tempfile.mkdtemp()
        os.chdir(tmp_dir)
                
        param_file = tempfile.NamedTemporaryFile(delete=False)  
        param_file.write(params)
        
        # copy some needed files
        os.system("cp %s/HighLExtrapTemplate_lenspotentialCls.dat ." %os.environ['CAMB_DIR'])
        
        # save the name and close the file
        param_file_name = param_file.name
        param_file.close()

        # call camb
        ans = subprocess.call(['%s/camb' %os.environ['CAMB_DIR'], param_file_name], stdout=open(os.devnull, 'w'))
        
        # get the transfer data
        data = np.loadtxt('./test_transfer_out.dat')
        k, T = data[:,0], data[:,6]
        
    except:
        raise
    finally:
        
        # go back to old directory and delete temporary directory
        os.chdir(initialDir)
        
        with ignored(OSError):
            shutil.rmtree(tmp_dir) # delete directory
        
        with ignored(OSError):
            os.remove(param_file_name)
        
    return k, T/T.max()