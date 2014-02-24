"""
 cosmo.py
 pyPT: class to handle a set of cosmological parameters
 
 author: Nick Hand 
 contact: nhand@berkeley.edu
 creation date: 02/21/2014
"""
import os
import parameters
import constants as c

#-------------------------------------------------------------------------------
class Cosmology(object):
    """
    A class to hold a set of cosmological parameters, and to determine a 
    pre-defined set of parameters robustly, given any dependencies between
    parameters.
    
    
    Notes
    -----
    Currently, only several combinations of the density parameters are valid:
    
    1. 'omegab' and 'omegac'
    2. 'omegam'
    3. 'omegab_h2' and 'omegac_h2'
    4. None of them
    
    If more than enough density parameters are supplied, the parameters will
    be used in this order.
    """
    
    # A dictionary of bounds for each parameter
    # This also forms a list of all parameters possible
    _bounds = {"sigma_8"  : [0.1, 10],
               "n"        : [-3, 4],
               "w"        : [-1.5, 0],
               "Tcmb"    : [0, 10.0],
               "Y_he"     : [0, 1],
               "N_eff"    : [1, 10],
               "z_reion"  : [2, 1000],
               'z_star'   : [2, 1e10], 
               'delta_c'  : [1, 1000], 
               "tau"      : [0, 1],
               "h"        : [0.05, 5],
               "H0"       : [5, 500],
               "omegab"   : [0, 1],
               "omegac"   : [0, 2],
               "omegal"   : [0, 2],
               "omegar"   : [0, 2],
               "omegam"   : [0, 3],
               "omegab_h2": [0, 1],
               "omegac_h2": [0, 2],
               "age"      : [0., 100.]}
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the parameters. 
        
        Parameters
        ----------
        *args : {str, dict}, len=1
            A single positional argument can be supplied. A str is interpreted
            as the name of a file to read parameters from. The key/value 
            pairs of a dict are interpreted as the parameters. 
        **kwargs : 
            The list of available keyword arguments are:
                default : {str, NoneType}, default = "Planck1_lens_WP_highL"
                    The name of a default cosmology to use from the parameters 
                    module
                force_flat : {bool}, default = False
                    If True, enforce a flat cosmology. This will modify 'omegal' 
                    only and never 'omegam'
            and the parameter keywords are:

                1. 'sigma_8': The normalization; mass variance in top-hat spheres with R = 8 Mpc/h
                2. 'n': The spectral index
                3. 'w': The dark-energy equation of state
                4. 'Tcmb': Temperature of the CMB
                5. 'Y_he': Helium fraction
                6. 'N_eff': Number of massless neutrino species
                7. 'z_reion': Redshift of reionization
                8. 'z_star' : Redshift of the surface of last scattering
                9. 'delta_c': The criticial ovedensity for collapse
                10. 'tau': Optical depth at reionization
                11. 'delta_c': The critical overdensity for collapse
                12. 'h': The hubble parameter
                12. 'H0': The hubble constant
                13. 'omegam': The normalized density of matter
                14. 'omegal': The normalized density of dark energy
                15. 'omegab': The normalized baryon density
                16. 'omegac': The normalized CDM density
                17. 'omegar : The normalized radiation density
                17. 'omegab_h2': The physical baryon density
                18. 'omegac_h2': The physical CDM density
                19. 'flat' : Whether to set force_flat = True
                    
        """
        # can only handle one positional 
        if len(args) > 2:
            raise TypeError("Expected at most 1 positional argument, got 2")
        
        # set the default cosmology parameter
        self.default = kwargs.pop('default', 'Planck1_lens_WP_highL')

        # set the base, updating the extra parameters with the default set
        if self.default is not None:
            self.__base = parameters.get_cosmology_from_string(self.default)
            
        # Set some simple parameters
        self.force_flat = kwargs.pop('force_flat', False)
        
        # critical density in units of h^2 M_sun / Mpc^3
        self.crit_dens = 3*(100.*c.km/c.second/c.Mpc)**2/(8*c.pi*c.G)/(c.M_sun/c.Mpc**3)
        
        self.update(*args, **kwargs)
    #end __init__
    
    #---------------------------------------------------------------------------
    def _check_bounds(self, item, low=None, high=None):
        """
        Check the bounds a specified parameter
        """
        if low is not None and high is not None:
            if self.__dict__[item] < low or self.__dict__[item] > high:
                raise ValueError("%s must be between %s and %s" %(item, low, high))
        elif low is not None:
            if self.__dict__[item] < low:
                raise ValueError("%s must be greater than %s" %(item, low))
        elif high is not None:
            if self.__dict__[item] > high:
                raise ValueError("%s must be less than %s" %(item, high))
    #end _check_bounds
    
    #---------------------------------------------------------------------------
    def __str__(self):
        """
        Return a dump of the parameters as the string representation 
        """
        return self.dump(return_str=True)
        
    #---------------------------------------------------------------------------
    def allowable(self):
        """
        Return the allowable input parameters
        """
        for k in sorted(Cosmology._bounds.keys(), key=str.lower):
            print "%s : %s" %(k.ljust(12), Cosmology._bounds[k])
    #end allowable
    
    #---------------------------------------------------------------------------
    def available(self):
        """
        Print out the string names of the available cosmologies
        """
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
            raise TypeError("Expected at most 1 positional argument, got 2")
            
        fromFile = False
        if N == 1:
            if isinstance(args[0], basestring):
                if not os.path.exists(args[0]):
                    _current = parameters.get_cosmology_from_string(args[0])
                else:
                    _current = self._load(args[0])
                    fromFile = True
            elif isinstance(args[0], dict):
                _current = dict(args[0], **kwargs)
            else:
                raise ValueError("Argument must be a string or dictionary. Valid strings:" + \
                            "\n%s" %([x()['name'] for x in parameters.available]))
        else:
            _current = kwargs
        
        self._set_params(_current)
    #end update
    
    #---------------------------------------------------------------------------   
    def _set_params(self, input_params):
        
        input_params.pop('reference', None)
        input_params.pop('name', None)
        
        flat = input_params.pop('flat', None)
        if flat is not None:
            self.force_flat = flat
        
        # check the input parameter keys       
        for k in input_params:
            if k not in Cosmology._bounds:
                raise ValueError("'%s' is not a valid parameter for Cosmology" %k)
                             
        #-----------------------------------------------------------------------
        # set the parameters with no other dependencies
        #-----------------------------------------------------------------------
        easy_params = ["sigma_8", "n", 'w', 'Tcmb', 'Y_he', "N_eff",
                        "z_reion", "tau", 'delta_c', "z_star", "age"]
        for p in easy_params:
            if p in input_params:
                self.__dict__.update({p : input_params.pop(p)})
            elif self.default is not None:
                self.__dict__.update({p:self.__base[p]})
                
        #-----------------------------------------------------------------------
        # set the parameters with dependencies
        #-----------------------------------------------------------------------
        
        ### h/H0
        #-----------------------------------------------------------------------
        if "h" in input_params and "H0" in input_params:
            if input_params['h'] != input_params["H0"] / 100.:
                raise AttributeError("h and H0 specified inconsistently")

        if "H0" in input_params:
            self.H0 = input_params.pop("H0")
            self.h = self.H0 / 100.

        if "h" in input_params:
            self.h = input_params.pop("h")
            self.H0 = 100 * self.h

        if not hasattr(self, "h") and self.default is not None:
            self.H0 = self.__base["H0"]
            self.h = self.H0 / 100.
            
        ### now the do the omega parameters
        #-----------------------------------------------------------------------
        
        # first compute default omega radiation, assuming N_eff massless neutrinos
        
        if self.default is not None:
            
            # Compute photon density from Tcmb
            constant = c.a_rad/c.c_light**2
            rho_crit = self.crit_dens * self.h**2 * (c.M_sun/c.Mpc**3) # now in g/cm^3
            omega_gam =  constant*self.Tcmb**4 / rho_crit

            # compute neutrino omega, assuming N_eff massless neutrinos
            omega_nu = 7./8.*(4./11)**(4./3)*self.N_eff*omega_gam
        
        # set the radiation density first
        if "omegar" in input_params:
            self.omegar = input_params.pop("omegar")
        else:
            if self.default is None:
                self.omegar = 0.
            else:
                self.omegar = omega_gam + omega_nu
            
        if "omegal" in input_params:
            self.omegal = input_params.pop("omegal")

        # make sure the omega matters are one of well-defined cases
        if 'omegab' and 'omegac' in input_params:
            for k in ['omegam', 'omegab_h2', 'omegac_h2']:
                input_params.pop(k, None)
            
        if 'omegam' in input_params:
            for k in ['omegab', 'omegac', 'omegab_h2', 'omegac_h2']:
                input_params.pop(k, None)
                
        if 'omegab_h2' and 'omegac_h2' in input_params:
            for k in ['omegam', 'omegab', 'omegac']:
                input_params.pop(k, None)
        
        if len(input_params) == 0:
            if self.force_flat and hasattr(self, "omegal"):
                self.omegam = 1 - self.omegal - self.omegar
                self.omegak = 0.
            elif self.default is not None:
                self.omegab_h2 = self.__base["omegab_h2"]
                self.omegac_h2 = self.__base["omegac_h2"]
                self.omegab = self.omegab_h2 / self.h**2
                self.omegac = self.omegac_h2 / self.h**2
                self.omegam = self.omegab + self.omegac
                
        elif "omegab" in input_params and "omegac" in input_params and len(input_params) == 2:
            self.omegab = input_params["omegab"]
            self.omegac = input_params["omegac"]
            self.omegam = self.omegab + self.omegac
            if hasattr(self, "h"):
                self.omegab_h2 = self.omegab * self.h**2
                self.omegac_h2 = self.omegac * self.h**2

        elif "omegam" in input_params and len(input_params) == 1:
            self.omegam = input_params["omegam"]

        elif "omegab_h2" in input_params and "omegac_h2" in input_params and len(input_params) == 2:
            if not hasattr(self, 'h'):
                raise AttributeError("You need to specify h as well")
            self.omegab_h2 = input_params["omegab_h2"]
            self.omegac_h2 = input_params["omegac_h2"]
            self.omegab = self.omegab_h2 / self.h**2
            self.omegac = self.omegac_h2 / self.h**2
            self.omegam = self.omegab + self.omegac

        else:
            raise AttributeError("Input values for omega* arguments are invalid" + str(input_params))

        if hasattr(self, "omegam"):
            self.mean_dens = self.crit_dens*self.omegam
            if self.force_flat:
                self.omegal = 1 - self.omegam - self.omegar
                self.omegak = 0.
            elif self.default is not None and not hasattr(self, "omegal"):
                self.omegal = self.__base["omegal"]

            if hasattr(self, "omegal") and not self.force_flat:
                self.omegak = 1 - self.omegal - self.omegam - self.omegar

        # check all the parameter values
        for k, v in Cosmology._bounds.iteritems():
            if k in self.__dict__:
                self._check_bounds(k, v[0], v[1])
    #end update
    
    #---------------------------------------------------------------------------
    def keys(self):
        """
        Return the names of all parameters
        """
        params = []
        for k in sorted(self.__dict__.keys(), key=str.lower):
            if k not in ['default', 'force_flat', '_Cosmology__base', 'mean_dens', 'crit_dens']:
                params.append(k)
        return params
    #end keys
    
    #---------------------------------------------------------------------------
    def dump(self, return_str=False):
        """
        Print all parameter names and values to std out.
        """
        s = "parameters\n"
        lines = [(k, str(self.__dict__[k])) for k in self.keys()]
        if len(lines) == 0:
            s = ' [no parameters]'
        else:
            L0 = max([len(l[0]) for l in lines])
            L1 = max([len(l[1]) for l in lines])

            for p, v in lines:
                s += ' %s %s\n' % (p.ljust(L0),v.ljust(L1))    
        if return_str: 
            return s
        else:
            print s
    #end dump
    
    #---------------------------------------------------------------------------
    def _load(self, filename, clear_current=False):
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
        D = {}
        if not clear_current:
            for k in self.keys(): D[k] = self.__dict__[k]
        
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
            
            if line[0] in Cosmology._bounds.keys():
                D[line[0]] = eval(line[1])

        if clear_current:
            self.__dict__.clear()
        
        return D
    #end load

    #---------------------------------------------------------------------------
    def write(self, filename, mode = 'w'):
        f = open(filename, mode)
        for key in self.keys():
            f.write("%s = %s\n" % (key, repr(self.__dict__[key])))
        f.close()
    #end write
    
    #---------------------------------------------------------------------------
    def clear(self):
        for k in self.keys():
            del self.__dict__[k]
    #end clear
    
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
