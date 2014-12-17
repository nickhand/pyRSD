"""
 param_reader.py
 pyRSD/fit: module for reading parameters
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 12/13/2014
"""
from .. import data_dir, os
from . import parsing_tools
import string

class ParamDict(dict):
    """
    Class to read parameters from file into a dictionary
    """
    def __init__(self, *args):
        """
        Optionally provide a filename to load parameters from
        """
        super(ParamDict, self).__init__()
        
        if len(args) > 0:
            self.load(args[0])
    
    #---------------------------------------------------------------------------
    def __getitem__(self, item):
        if item not in self:
            return None
        return dict.__getitem__(self, item)
    
    #---------------------------------------------------------------------------
    def load(self, filename, clear_current=False):
        """
        Fill the dictionary with the parameters specified in the filename. If 
        `clear_current` is `True`, first empty current parameter settings.        
        """
        # try to find the file 
        if os.path.exists(filename):
            pass
        elif (os.path.exists("%s/%s" %(data_dir, filename))):
            filename = "%s/%s" %(data_dir, filename)
        else:
            raise ValueError("Could not find file '%s', tried ./%s and %s/%s" \
                             %(filename, filename, data_dir, filename))
        
        D = {} if clear_current else self.copy()
            
        linecount = 0
        old = ''
        for line in open(filename, 'r'):
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            s = line.split('#')
            line = s[0]
            s = line.split('\\')
            if len(s) > 1:
                old = string.join([old, s[0]])
                continue
            else:
                line = string.join([old, s[0]])
                old = ''
            for i in xrange(len(line)):
                if line[i]!=' ':
                    line = line[i:]
                    break
                    
            line = line.split('=')
            line = [x.strip() for x in line]

            if (len(line) == 0 or line[0] == ''): 
                continue
            if len(line)==1:
                raise ValueError("Must specify value for parameter %s on line %i" %(line[0], linecount))
            elif len(line) != 2:
                raise ValueError("Cannot understand line %i of %s" %(linecount,filename))
                
            if (line[0][0] >= '0' and line[0][0] <= '9'):
                raise ValueError, "invalid variable name %s" % line[0]

            # check for variables in the value
            if '$' in line[1]: line[1] = parsing_tools.replace_vars(line[1], D)
            
            # check for any functions calls in the line
            modules = parsing_tools.import_function_modules(line[1])

            # now save to the dict, eval'ing the line
            D[line[0].strip()] = eval(line[1].strip(), globals().update(modules), D)

        if clear_current:
            self.clear()
        self.update(D)
            
    #end load
    
    #---------------------------------------------------------------------------

#endclass ParamDict
#-------------------------------------------------------------------------------

    
       
    
        
