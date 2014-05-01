"""
 parameters.py: this module contains dictionaries with sets of parameters for a
 given cosmology.

 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 05/01/2013
"""
import sys

#-------------------------------------------------------------------------------
def Planck1():
    """
    Planck 2013 DR1 cosmology parameters from the best-fit values listed 
    in Table 2 of arXiv:1303.5076v2.
    """ 
    c = {'omegac_h2' : 0.12029,
         'omegab_h2' : 0.022068,   
         'omegan_h2' : 0.000645,
         'N_nu'      : 2.046,
         'N_nu_massive' : 1,
         'H0'        : 67.11, 
         'z_reion'   : 11.35, 
         'n'         : 0.9624,
         'tau'       : 0.0925, 
         'sigma_8'   : 0.8344,
         'age'       : 13.819, 
         'flat'      : True}
    return c

#-------------------------------------------------------------------------------
def Planck1_lens_WP_highL():
    """
    Planck 2013 DR1 + lensing + WMAP low ell polarization + highL data
    cosmology parameters from the best-fit values listed in Table 5 of 
    arXiv:1303.5076v2
    """ 
    c = {'omegac_h2' : 0.11847,
         'omegab_h2' : 0.022199,   
         'omegan_h2' : 0.000645,
         'N_nu'      : 2.046,
         'N_nu_massive' : 1, 
         'H0'        : 67.94, 
         'z_reion'   : 11.42, 
         'n'         : 0.9624,
         'tau'       : 0.0943, 
         'sigma_8'   : 0.8271,
         'age'       : 13.7914, 
         'flat'      : True}
    return c
    
#-------------------------------------------------------------------------------
def Planck1_WP_highL_BAO():
    """
    Planck 2013 DR1 + WMAP low ell polarization + highL data + BAO
    cosmology parameters from the best-fit values listed in Table 5 of 
    arXiv:1303.5076v2
    """ 
    c = {'omegac_h2' : 0.11889,
         'omegab_h2' : 0.022161,   
         'omegan_h2' : 0.000645,
         'N_nu'      : 2.046,
         'N_nu_massive' : 1, 
         'H0'        : 67.77, 
         'z_reion'   : 11.52, 
         'n'         : 0.9611,
         'tau'       : 0.0952, 
         'sigma_8'   : 0.8288,
         'age'       : 13.7965, 
         'flat'      : True}
    return c

#-------------------------------------------------------------------------------
def WMAP9_eCMB():
    """
    WMAP9 + eCMB cosmology parameters from Hinshaw et al. 2012, 
    arxiv:1212.5226v3, (Table 4, last column)
    """
    c = {'omegac_h2' : 0.1126,
         'omegab_h2' : 0.02229, 
         'omegan_h2' : 0.0,
         'N_nu'      : 3.04,
         'N_nu_massive' : 0, 
         'H0'        : 70.5,
         'z_reion'   : 10.3, 
         'n'         : 0.9646,
         'tau'       : 0.084,
         'sigma_8'   : 0.810,
         'age'       : 13.742,
         'flat': True}
    return c
        
#-------------------------------------------------------------------------------    
def WMAP7():
    """
    WMAP7 maximum likelihood cosmology parameters from Komatsu et al. 2011, 
    arxiv:1001.4538v3 (Table 1 WMAP7 ML)
    """
    c = {'omegac_h2' : 0.1116,
         'omegab_h2' : 0.02227, 
         'omegan_h2' : 0.0,
         'N_nu'      : 3.04,
         'N_nu_massive' : 0,
         'H0'        : 70.3,
         'z_reion'   : 10.4,
         'n'         : 0.966,
         'tau'       : 0.085,
         'sigma_8'   : 0.809, 
         'age'       : 13.79,
         'flat' : True}
    return c
    
#-------------------------------------------------------------------------------
def WMAP5():
    """
    WMAP5 maximum likelihood cosmology parameters from Komatsu et al. 2009, 
    ApJS, 180, 330., arxiv:0803.0547v2 (Table 1 WMAP5 ML)
    """
    c = {'omegac_h2' : 0.1081,
         'omegab_h2' : 0.02268, 
         'omegan_h2' : 0.0,
         'N_nu'      : 3.04,
         'N_nu_massive' : 0,
         'H0'        : 72.4,
         'z_reion'   : 11.2,
         'n'         : 0.961,
         'tau'       : 0.089, 
         'sigma_8'   : 0.787,
         'age'       : 13.69,
         'flat': True}
    return c
#-------------------------------------------------------------------------------
def get_cosmology_from_string(arg):
    """ 
    Return a cosmology instance from a string.
    """
    try:
        cosmo_func = getattr(sys.modules[__name__], arg)
        cosmo_dict = cosmo_func()
    except AttributeError:
        s = "Unknown cosmology '%s'. Valid cosmologies:\n%s" %(
                arg, [x.func_name for x in available])
        raise ValueError(s)
    cosmo_dict.update(extras)
    return cosmo_dict
    
#-------------------------------------------------------------------------------
available = ( Planck1, 
              Planck1_lens_WP_highL, 
              Planck1_WP_highL_BAO, 
              WMAP9_eCMB, 
              WMAP7, 
              WMAP5)

# define parameters common to all sets
extras = {'w'            : -1., 
          'z_star'       : 1090., 
          'Tcmb'         : 2.7255,
          'delta_c'      : 1.686, 
          'Y_he'         : 0.2477,
          'cs2_lam'      : 1.}
          