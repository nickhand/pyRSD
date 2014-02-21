"""
 cosmo_dict.py
 pyPT: class to handle a dictionary of cosmological parameters
 
 author: Nick Hand 
 contact: nhand@berkeley.edu
 creation date: 02/21/2014
"""
import os
import string
import re
import parameters
import collections
import numpy as np
import utils.physical_constants as pc

_allowable_params = {'omega_c_0' : "current cold dark matter density parameter", \
                     'omega_b_0' : "current baryon density parameter",\
                     'omega_m_0' : "current matter density parameter; omega_c_0 + omega_m_0",\
                     'flat'     : "assume total density parameter is unity; boolean", \
                     'omega_l_0': "current dark energy density parameter", \
                     'omega_r_0': "current radiation density parameter", \
                     'omega_k_0': "the current curvature parameter; 1 - Omega_tot", \
                     'h'        : "dimensionaless Hubble parameter", \
                     'n_s'      : "density perturbation spectral index", \
                     'Tcmb_0'   : "current CMB temperature", \
                     'Neff'     : "effective number of relativistic species", \
                     'sigma_8'  : "power spectrum normalization", \
                     'tau'      : "reionization optical depth", \
                     'z_reion'  : "redshift of hydrogen reionization", \
                     'z_star'   : "redshift of the surface of last scattering", \
                     't0'       : "age of the universe", \
                     'w0'       : "dark energy equation of state, w = w0 + w1*z", \
                     'w1'       : "dark energy equation of state, w = w0 + w1*z", \
                     'name'     : "name of this parameter set", \
                     'reference': "publication reference for this parameter set"}
                      
_default_params = parameters.Planck13()

class params(collections.MutableMapping):
    """
    A dictionary to hold the values for a pre-defined set of 
    cosmological parameters.
    """

    def __init__(self, *args, **kwargs):
        
        if len(args) > 2:
            raise TypeError("update expected at most 1 argument, got 2")       
    
        self.dict = dict()
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        return self.dict[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        if key in _allowable_params.keys():
            self.dict[self.__keytransform__(key)] = value
        else:
            raise KeyError(key)

    def __delitem__(self, key):
        del self.dict[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.dict)

    def __len__(self):
        return len(self.dict)

    def __keytransform__(self, key):
        return key

    def __str__(self):
        return self.dump(return_str=True)

    #---------------------------------------------------------------------------
    def keys(self):
        return sorted(self.dict.keys(), key=str.lower)
    #end keys
   
    #---------------------------------------------------------------------------
    def allowable(self):
        for k in sorted(_allowable_params.keys(), key=str.lower):
            print "%s : %s" %(k.ljust(12), _allowable_params[k])
    #end allowable
    
    #---------------------------------------------------------------------------
    def available(self):
        print("Valid cosmologies:\n%s" %([x()['name'] for x in parameters.available]))
    #end available
        
    #---------------------------------------------------------------------------
    def update(self, *args, **kwargs):
        """ 
        Update the current cosmology parameters. 
        
        Notes
        -----
        At most 1 positional argument can be supplied. If a string, the 
        parameters will be updated from file if the path exists, or to the 
        builtin parameter set defined by the input name. If a dictionary, 
        the parameters will be updated from it. Parameters will be updated 
        from keyword arguments as well.  
        """ 
        N = len(args)
        if N > 2:
            raise TypeError("update expected at most 1 argument, got 2")
            
        fromFile = False
        if N == 1:
            if isinstance(args[0], basestring):
                if not os.path.exists(args[0]):
                    _current = parameters.get_cosmology_from_string(args[0])
                else:
                    self.load(args[0])
                    fromFile = True
            elif isinstance(args[0], dict):
                _current = args[0]
            else:
                raise ValueError("Argument must be a string or dictionary. Valid strings:" + \
                            "\n%s" %([x()['name'] for x in parameters.available]))
        else:
            _current = kwargs
        if not fromFile:
            for k, v in _current.iteritems(): self[k] = v
    
        # make sure to remove omega_c_0 and omega_b_0 if omega_m_0 is 
        # specified and they are not
        keys = self.keys()
        if 'omega_m_0' in keys:
            if 'omega_c_0' not in keys and 'omega_b_0' not in keys:
                self['omega_c_0'] = None
                self['omega_b_0'] = None
        
        # verify the parameters
        self.verify()
        
        if 'name' not in keys or 'reference' not in keys:
            self['name'] = None
            self['reference'] = None
    #end update
    
    #---------------------------------------------------------------------------   
    def _set_extras(self):
        
        if self['Tcmb_0'] > 0:
            
            # Compute photon density from Tcmb
            constant = pc.a_rad / pc.c_light**2
            rho_crit = 3.*(100*self['h']*pc.km/pc.Mpc)**2 / (8*np.pi*pc.G)
            omega_gam_0 =  constant*self['Tcmb_0']**4 / rho_crit

            # compute neutrino omega
            omega_nu_0 = 7./8.*(4./11)**(4./3)*self['Neff']*omega_gam_0
        else:
            omega_gam_0 = 0.
            omega_nu_0 = 0.
        
        if 'omega_r_0' not in self.keys():
            self['omega_r_0'] = omega_nu_0 + omega_gam_0
        
        # set the dark energy density, if we're assuming flat
        if self['flat']:
            self['omega_l_0'] = 1. - self['omega_m_0'] - self['omega_r_0']
            self['omega_k_0'] = 0.
        else:
            self['omega_k_0'] = 1.-self['omega_m_0']-self['omega_l_0']-self['omega_r_0']
    
    #---------------------------------------------------------------------------
    def verify(self):
        """
        Verify the input cosmology parameters
        """
        used_default = False
        if self.is_empty():
            self.update(_default_params)
            used_default = True
            added = _default_params
        else:
            # determine if we are using flat cosmology
            flat = self['flat'] if 'flat' in self.keys() else _default_params['flat']
        
            added = {}
            # replace missing with default params
            for k in _allowable_params:
                if k is 'reference' or k is 'name': continue
                if k is 'omega_l_0':
                    if flat: continue
                if k is 'omega_r_0' or k is 'omega_k_0': continue
                if k not in self.keys():
                    used_default = True
                    self[k] = _default_params[k]
                    added[k] = _default_params[k]
                
        self._set_extras()
        
        # print a warning
        if used_default:
            print "Warning: Missing cosmology parameters, using '%s' parameters for these." %_default_params['name']
            print "\n".join("   %s = %s" %(k.ljust(10),v) for k, v in added.iteritems())
            
                          
    #end _verify_params
    #---------------------------------------------------------------------------
    def dump(self, return_str=False):
        """
        Print all parameters and types
        """
        s = "parameters\n"

        lines = [ (k, str(self[k]), type(self[k]).__name__) \
                      for k in self.keys() ]

        if len(lines)==0:
            s = ' [no parameters]'

        else:
            L0 = max([len(l[0]) for l in lines])
            L1 = max([len(l[1]) for l in lines])

            for p,v,t in lines:
                s += ' %s %s %s\n' % (p.ljust(L0),v.ljust(L1),t)
                
        if return_str: 
            return s
        else:
            print s
    #end dump
    
    #---------------------------------------------------------------------------
    def load(self, filename, clear_current=False):
        """
        Fill the params dictionary with the parameters specified
        in the filename.

        if clear_current is True, then first empty current parameter settings

        filename has one parameter per line
        name = val #comment

        For example:
          #start params.dict
          dir = "/local/tmp/"              
          pi  = 3.14
          num_entries = 3                       
          file1 = "$(dir)/myfile1.txt"          # dollar sign indicates variable
                                                # substitution
          file2 =  dir + "myfile2.txt"
          #end params.dat
        """
        if clear_current:
            D = {}
        else:
            D = self
        linecount = 0
        for line in open(filename):
            linecount += 1
            line = ' '.join(line.split('#')[:1]).split('\\')
            line = ' '.join(line)
            exec(line) # this keeps track of any variables input in the file
            line = line.split('=')
            line = [x.strip() for x in line]

            if len(line)==0 or line[0]=='': 
                continue
            if len(line)==1:
                raise ValueError("Must specify value for parameter %s on line %i" \
                    %(line[0], linecount))
            elif len(line) != 2:
                raise ValueError("Cannot understand line %i of %s" %(linecount,filename))
            if (line[0][0]>='0' and line[0][0]<='9'):
                raise ValueError("Invalid variable name %s" %line[0])

            # check for variables in the value
            if '$' in line[1]:
                try:
                    line[1] = replace_vars(line[1], dict(dict(D).items() + os.environ.items()))
                except:
                    raise ValueError("Variable replacement error in line %s" %line)
            
            if line[0] in _allowable_params.keys():
                D[line[0]] = eval(line[1])

        if clear_current:
            self.clear()
            self.update(D)
    #end load

    #---------------------------------------------------------------------------
    def write(self, filename, mode = 'w'):
        f = open(filename, mode)
        for key in self.keys():
            f.write("%s = %s\n" % (key, repr(self[key])))
        f.close()
    #end write
        
    #---------------------------------------------------------------------------
    def unify(self, *args, **kwargs):
        self.clear()
        self.update(dict(*args, **kwargs))
    #end unify
        
    #---------------------------------------------------------------------------
    def is_empty(self):
        return len(self.keys()) == 0
    #end is_empty
    
    #---------------------------------------------------------------------------
    def set_default(self, clear_current=False):
        if clear_current:
            self.clear()
            self.update(_default_params)
        else:
            for key, val in _default_params.iteritems():
                if key in self.keys():
                    continue
                self[key] = val
    #end set_default
    
    #---------------------------------------------------------------------------
#endclass params

#-------------------------------------------------------------------------------        
def replace_vars(s, D):
    """
    Given a string s and a dictionary of variables D, replace all variable
    names with the value from the dict D

    variable names in s are denoted by '$(' at the beginning and ')' at
    the end, or '$' at the beginning and a non-variable character at the
    end.  Variables must be valid python variable names, that is they
    consist of only alphanumeric characters (A-Z,a-z,0-9) and underscores,
    and cannot start with a number.

    example:
    >> D = {'my_var1' : 'abc',
            'my_var2' : '123' }
    >> s = "I know my $(my_var1)s and $my_var2's"
    >> print replace_vars(s,D)

    I know my abcs and 123's
    """
    s_in = str(s) 
    s_out = ''

    while True:
        i = s_in.find('$')
        if i==-1:
            s_out += s_in
            break

        s_out += s_in[:i]
        s_in = s_in[i+1:]

        if len(s_in)==0:
            raise ValueError, "trailing $"

        elif s_in[0] == '(':
            i = s_in.find(')')
            if i==-1:
                raise ValueError, "unmatched '('"
            var = s_in[1:i]

            s_in = s_in[i+1:]
            try:
                s_out += str(D[var])
            except:
                s_out += os.environ[var]

        else:
            var = ''
            i = 0
            while True:
                if i>=len(s_in):
                    break
                s = s_in[i]
                if (s >= 'a' and s <= 'z') \
                        or (s >= 'A' and s <= 'Z') \
                        or (s >= '0' and s <= '9') \
                        or s=='_':
                    var += s
                    i += 1
                else:
                    break
            s_in = s_in[i:]
            s_out += str(D[var])
    return s_out
#end replace_vars

#-------------------------------------------------------------------------------
