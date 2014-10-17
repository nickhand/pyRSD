"""
 cosmo.py
 pyRSD: class to handle a set of cosmological parameters
 
 author: Nick Hand 
 contact: nhand@berkeley.edu
 creation date: 02/21/2014
"""
import os
from . import parameters
from . import _constants as c
import copy

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
    #. 'omegab_h2' and 'omegac_h2'
    #. 'omegam'
    #. None of them
    
    If more than enough density parameters are supplied, the parameters will
    be used in this order.
    """
    
    # A dictionary of bounds for each parameter
    # This also forms a list of all parameters possible
    _cp = ['sigma_8', 'n', 'w', 'cs2_lam', 'Tcmb', 'Y_he', 'N_nu', 'N_nu_massive', 
            'z_reion', 'tau', 'delta_c', 'h', 'H0', 'omegan', 'omegam', 'omegal',
            'omegab', 'omegac', 'omegab_h2', 'omegac_h2', 'omegan_h2', 'z_star',
            'age', 'flat', 'default', 'omegar']
            
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
                flat : {bool}, default = False
                    If True, enforce a flat cosmology. This will modify 'omegal' 
                    only and never 'omegam'
            and the parameter keywords are:

            1. ``sigma_8``: Mass variance in top-hat spheres with :math:`R=8Mpc h^{-1}`
            #. ``n``: The spectral index
            #. ``w``: The dark-energy equation of state
            #. ``cs2_lam``: The constant comoving sound speed of dark energy
            #. ``Tcmb``: Temperature of the CMB
            #. ``Y_he``: Helium fraction
            #. ``N_nu``: Number of massless neutrino species
            #. ``N_nu_massive``: Number of massive neutrino species
            #. ``z_reion``: Redshift of reionization
            #. ``z_star`` : Redshift of the surface of last scattering
            #. ``tau``: Optical depth at reionization
            #. ``delta_c``: The critical overdensity for collapse
            #. ``h``: The hubble parameter
            #. ``H0``: The hubble constant
            #. ``omegan``: The normalized density of neutrinos
            #. ``omegam``: The normalized density of matter
            #. ``omegal``: The normalized density of dark energy
            #. ``omegab``: The normalized baryon density
            #. ``omegac``: The normalized CDM density
            #. ``omegab_h2``: The normalized baryon density times ``h**2``
            #. ``omegac_h2``: The normalized CDM density times ``h**2``
            #. ``omegan_h2``: The normalized neutrino density times ``h**2``
            
        """
        # can only handle one positional 
        if len(args) > 2:
            raise TypeError("Expected at most 1 positional argument, got 2")
        
        # explicitly copy the input args
        args = copy.deepcopy(args)
       
        # set the default cosmology parameter
        try:
            self.default = args[0].pop('default')
        except:
            self.default = kwargs.pop('default', 'Planck1_lens_WP_highL')

        # set the base, updating the extra parameters with the default set
        if self.default is not None:
            self.__base = parameters.get_cosmology_from_string(self.default)
                    
        # critical density in units of h^2 M_sun / Mpc^3
        self.crit_dens = 3*(100.*c.km/c.second/c.Mpc)**2/(8*c.pi*c.G)/(c.M_sun/c.Mpc**3)
        
        self.update(*args, **kwargs)
    #end __init__
    
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
        for k in sorted(Cosmology._cp, key=str.lower):
            print "%s" %(k.ljust(12))
    #end allowable
    
    #---------------------------------------------------------------------------
    def available(self):
        """
        Print out the string names of the available cosmologies
        """
        print("Valid cosmologies:\n%s" %([x.func_name for x in parameters.available]))
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
                if args[0] is None:
                    _current = {}
                else:
                    raise ValueError("Argument must be a string or dictionary. Valid strings:" + \
                                "\n%s" %([x.func_name for x in parameters.available]))
        else:
            _current = kwargs
        
        self._set_params(_current)
    #end update
    
    #---------------------------------------------------------------------------   
    def _set_params(self, input_params):
            
        # remove any incorrect parameters
        input_params = {k:v for k, v in input_params.iteritems() if k in Cosmology._cp}
                                             
        #-----------------------------------------------------------------------
        # set the parameters with no other dependencies
        #-----------------------------------------------------------------------
        easy_params = ["sigma_8", "n", 'w', 'cs2_lam', 'Tcmb', 'Y_he', "N_nu", 
                        "N_nu_massive", 'z_star', "z_reion", "tau", 'delta_c', 
                        "age", "flat"]
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
        #-----------------------------------------------------------------------
        # set massive neutrinos contribution
        #-----------------------------------------------------------------------
        if "omegan" in input_params and "omegan_h2" in input_params:
            if input_params['omegan'] != input_params["omegan_h2"] / self.h**2:
                raise AttributeError("omegan and omegan_h2 specified inconsistently")

        if "omegan" in input_params:
            self.omegan = input_params.pop("omegan")
            self.omegan_h2 = self.omegan * self.h**2

        if "omegan_h2" in input_params:
            self.omegan_h2 = input_params.pop("omegan_h2")
            self.omegan = self.omegan_h2 / self.h**2

        if not hasattr(self, "omegan_h2") and self.default is not None:
            self.omegan_h2 = self.__base["omegan_h2"]
            self.omegan = self.omegan_h2 / self.h**2
            
        ### now the do the omega parameters
        #-----------------------------------------------------------------------
        
        # first compute default omega radiation, assuming N_eff massless neutrinos
        if "omegar" in input_params:
            self.omegar = input_params.pop("omegar")
        else:
            try:
                # Compute photon density from Tcmb
                constant = c.a_rad/c.c_light**2
                rho_crit = self.crit_dens * self.h**2 * (c.M_sun/c.Mpc**3) # now in g/cm^3
                omega_gam =  constant*self.Tcmb**4 / rho_crit

                # compute neutrino omega, assuming N_nu massless neutrinos + omegan from massive neutrinos
                omega_nu = 7./8.*(4./11)**(4./3)*self.N_nu*omega_gam + self.omegan
        
                self.omegar = omega_gam + omega_nu
            except:
                self.omegar = 0.
                
        if "omegal" in input_params:
            self.omegal = input_params.pop("omegal")
        
        # make sure omega matters are specified correctly
        if all(k in input_params for k in ['omegam', 'omegac', 'omegab']):
            omm = input_params['omegam']
            omb = input_params["omegab"]
            omc = input_params["omegac"]
            if omm != (omb + omc):
                raise AttributeError("input omegam not equal to input omegac + omegab")
                
        if all(k in input_params for k in ['omegam', 'omegac_h2', 'omegab_h2']):
            tmp = (input_params["omegab_h2"] + input_params['omegac_h2'])/self.h**2
            if input_params['omegam'] != tmp:
                raise AttributeError("input omegam not equal to input (omegac_h2 + omegab_h2)/h**2")

        # make sure the omega matters are one of well-defined cases
        if all(k in input_params for k in ['omegab', 'omegac']):
            for k in ['omegam', 'omegab_h2', 'omegac_h2']:
                input_params.pop(k, None)
                
        if all(k in input_params for k in ['omegab_h2', 'omegac_h2']):
            for k in ['omegam', 'omegab', 'omegac']:
                input_params.pop(k, None)
        
        if 'omegam' in input_params:
            for k in ['omegab', 'omegac', 'omegab_h2', 'omegac_h2']:
                input_params.pop(k, None)
                        
        if len(input_params) == 0:
            if self.flat and hasattr(self, "omegal"):
                self.omegam = 1 - self.omegal - self.omegar
                self.omegak = 0.
            elif self.default is not None:
                self.omegab_h2 = self.__base["omegab_h2"]
                self.omegac_h2 = self.__base["omegac_h2"]
                self.omegab = self.omegab_h2 / self.h**2
                self.omegac = self.omegac_h2 / self.h**2
                self.omegam = self.omegab + self.omegac
                
        elif all(k in input_params for k in ['omegab', 'omegac']) and len(input_params) == 2:
            self.omegab = input_params["omegab"]
            self.omegac = input_params["omegac"]
            self.omegam = self.omegab + self.omegac
            if hasattr(self, "h"):
                self.omegab_h2 = self.omegab * self.h**2
                self.omegac_h2 = self.omegac * self.h**2
    
        elif all(k in input_params for k in ['omegab_h2', 'omegac_h2']) and len(input_params) == 2:
            if not hasattr(self, 'h'):
                raise AttributeError("You need to specify h as well")
            self.omegab_h2 = input_params["omegab_h2"]
            self.omegac_h2 = input_params["omegac_h2"]
            self.omegab = self.omegab_h2 / self.h**2
            self.omegac = self.omegac_h2 / self.h**2
            self.omegam = self.omegab + self.omegac

        elif "omegam" in input_params and len(input_params) == 1:
            self.omegam = input_params["omegam"]
            self.omegab_h2 = self.omegac_h2 = 0.
            self.omegab = self.omegac = 0.

        else:
            raise AttributeError("Input values for omega* arguments are invalid" + str(input_params))

        if hasattr(self, "omegam"):
            self.mean_dens = self.crit_dens*self.omegam
            if self.flat:
                self.omegal = 1 - self.omegam - self.omegar
                self.omegak = 0.
            elif self.default is not None and not hasattr(self, "omegal"):
                self.omegal = self.__base["omegal"]

            if hasattr(self, "omegal") and not self.flat:
                self.omegak = 1 - self.omegal - self.omegam - self.omegar

    #end update
    #---------------------------------------------------------------------------
    def camb_dict(self):
        """
        Collect parameters into a dictionary suitable for camb.
        
        Returns
        -------
        dict
            Dictionary of values appropriate for camb
        """
        map = {"w" : "w",
               "Tcmb" : "temp_cmb",
               "Y_he" : "helium_fraction",
               "tau" : "re_optical_depth",
               "N_nu" : "massless_neutrinos",
               "N_nu_massive" : "massive_neutrinos",
               "omegab_h2" : "ombh2",
               "omegac_h2" : "omch2",
               "omegan_h2" : "omnuh2",
               "H0" : "hubble",
               "omegak" : "omk",
               "cs2_lam" : "cs2_lam",
               "n" : "scalar_spectral_index(1)"}

        return_dict = {}
        for k, v in self.__dict__.iteritems():
            if k in map:
                return_dict.update({map[k]: v})

        return return_dict
    #---------------------------------------------------------------------------
    def dict(self):
        """
        Collect parameters defining this cosmology in dictionary form.
        """
        return_dict = {k:self.__dict__[k] for k in self.keys() if k not in ['omegar', 'omegak']}
        
        # remove parameters with duplicate info
        pairs = [('h', 'H0'), ('omegab', 'omegab_h2'), ('omegac', 'omegac_h2'), 
                    ('omegan', 'omegan_h2')]
        for pair in pairs:
            if pair[0] in return_dict and pair[1] in return_dict:
                del return_dict[pair[1]]
        if 'omegab' in return_dict and 'omegac' in return_dict:
            if 'omegam' in return_dict:
                del return_dict['omegam']
        return return_dict
    #---------------------------------------------------------------------------
    def keys(self):
        """
        Return the names of all parameters
        """
        params = []
        for k in sorted(self.__dict__.keys(), key=str.lower):
            if k not in ['default', '_Cosmology__base', 'mean_dens', 'crit_dens']:
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
            
            if line[0] in Cosmology._cp:
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
