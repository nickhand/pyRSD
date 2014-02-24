"""
 parameters.py: this module contains dictionaries with sets of parameters for a
 given cosmology.

 Each cosmology has the following parameters defined:

     ==========  =====================================
     omega_c_0   Omega cold dark matter at z=0
     omega_b_0   Omega baryon at z=0
     omega_m_0   Omega matter at z=0
     flat        Is this assumed flat?  If not, omega_l_0 must be specified
     omega_l_0   Omega dark energy at z=0 if flat is False
     omega_r_0   Omega radiation at z=0
     h           Dimensionless Hubble parameter at z=0 in km/s/Mpc
     n           Density perturbation spectral index
     Tcmb_0      Current temperature of the CMB
     Neff        Effective number of neutrino species
     sigma_8     Density perturbation amplitude
     tau         Ionization optical depth
     z_reion     Redshift of hydrogen reionization
     z_star      Redshift of the surface of last scattering
     t0          Age of the universe in Gyr
     w0          The dark energy equation of state
     w1          The redshift derivative of w0
     name        The name of this parameter set
     reference   Reference for the parameters
     ==========  =====================================

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
    ref = "Planck Collaboration 2013, Paper XVI, arXiv:1303.5076v2 Table 2 (Planck)"
    c = {'omegac_h2' : 0.12029,
         'omegab_h2' : 0.022068,   
         'omegal'    : 0.6825, 
         'H0'        : 67.11, 
         'z_reion'   : 11.35, 
         'n'         : 0.9624,
         'tau'       : 0.0925, 
         'sigma_8'   : 0.8344,
         'age'       : 13.819, 
         'reference' : ref, 
         'name'      : 'Planck1'}
    return c

#-------------------------------------------------------------------------------
def Planck1_lens_WP_highL():
    """
    Planck 2013 DR1 + lensing + WMAP low ell polarization + highL data
    cosmology parameters from the best-fit values listed in Table 5 of 
    arXiv:1303.5076v2
    """ 
    ref =  "Planck Collaboration 2013, Paper XVI, arXiv:1303.5076v2 Table 5 " + \
            "(Planck + lensing + WP + highL)"
    c = {'omegac_h2' : 0.11847,
         'omegab_h2' : 0.022199,   
         'omegal'    : 0.6939, 
         'H0'        : 67.94, 
         'z_reion'   : 11.42, 
         'n'         : 0.9624,
         'tau'       : 0.0943, 
         'sigma_8'   : 0.8271,
         'age'       : 13.7914, 
         'reference' : ref, 
         'name'      : 'Planck1_lens_WP_highL'}
    return c
    
#-------------------------------------------------------------------------------
def Planck1_WP_highL_BAO():
    """
    Planck 2013 DR1 + WMAP low ell polarization + highL data + BAO
    cosmology parameters from the best-fit values listed in Table 5 of 
    arXiv:1303.5076v2
    """ 
    ref =  "Planck Collaboration 2013, Paper XVI, arXiv:1303.5076v2 Table 5 " + \
            "(Planck + WP + highL + BAO)"
    c = {'omegac_h2' : 0.11889,
         'omegab_h2' : 0.022161,   
         'omegal'    : 0.6914, 
         'H0'        : 67.77, 
         'z_reion'   : 11.52, 
         'n'         : 0.9611,
         'tau'       : 0.0952, 
         'sigma_8'   : 0.8288,
         'age'       : 13.7965, 
         'reference' : ref, 
         'name'      : 'Planck1_WP_highL_BAO'}
    return c

#-------------------------------------------------------------------------------
def WMAP9_eCMB():
    """
    WMAP9 + eCMB cosmology parameters from Hinshaw et al. 2012, 
    arxiv:1212.5226v3, (Table 4, last column)
    """
    ref = "Hinshaw et al. 2012, arXiv 1212.5226v3. Table 4 (WMAP + eCMB)"
    c = {'omegac_h2' : 0.1126,
         'omegab_h2' : 0.02229, 
         'omegal'    : 0.728, 
         'H0'        : 70.5,
         'z_reion'   : 10.3, 
         'n'         : 0.9646,
         'tau'       : 0.084,
         'sigma_8'   : 0.810,
         'age'       : 13.742,
         'reference' : ref,
         'name': 'WMAP9_eCMB'}
    return c
        
#-------------------------------------------------------------------------------    
def WMAP7():
    """
    WMAP7 maximum likelihood cosmology parameters from Komatsu et al. 2011, 
    arxiv:1001.4538v3 (Table 1 WMAP7 ML)
    """
    ref = "Komatsu et al. 2011, ApJS, 192, 18. Table 1 (WMAP7 ML)"
    c = {'omegac_h2' : 0.1116,
         'omegab_h2' : 0.02227, 
         'omegal'    : 0.729,
         'H0'        : 70.3,
         'z_reion'   : 10.4,
         'n'         : 0.966,
         'tau'       : 0.085,
         'sigma_8'   : 0.809, 
         'age'       : 13.79,
         'reference' : ref, 
         'name': 'WMAP7'}
    return c
    
#-------------------------------------------------------------------------------
def WMAP5():
    """
    WMAP5 maximum likelihood cosmology parameters from Komatsu et al. 2009, 
    ApJS, 180, 330., arxiv:0803.0547v2 (Table 1 WMAP5 ML)
    """
    ref = "Komatsu et al. 2009, ApJS, 180, 330. Table 1 (WMAP5 ML)"
    c = {'omegac_h2' : 0.1081,
         'omegab_h2' : 0.02268, 
         'omegal'    : 0.751,
         'H0'        : 72.4,
         'z_reion'   : 11.2,
         'n'         : 0.961,
         'tau'       : 0.089, 
         'sigma_8'   : 0.787,
         'age'       : 13.69,
         'reference' : ref, 
         'name': 'WMAP5'}
    return c
    
#-------------------------------------------------------------------------------
def Matter_Dominated():
    """
    A flat, matter-dominated universe, with Omega_m = 1
    """
    c = {'omegam' : 1.0,
         'omegar' : 0.,  
         'omegal' : 0.,
         'name'   : 'Matter_Dominated'}
    return c
    
#-------------------------------------------------------------------------------
def Radiation_Dominated():
    """
    A flat, radiation-dominated universe, with Omega_r = 1 
    """
    c = {'omegam' : 0.,
         'omegal' : 0.,
         'omegar' : 1., 
         'name'   : 'Radiation_Dominated'}
            
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
                arg, [x()['name'] for x in available])
        raise ValueError(s)
    cosmo_dict.update(extras)
    return cosmo_dict
    
#-------------------------------------------------------------------------------
available = ( Planck1, 
              Planck1_lens_WP_highL, 
              Planck1_WP_highL_BAO, 
              WMAP9_eCMB, 
              WMAP7, 
              WMAP5, 
              Matter_Dominated, 
              Radiation_Dominated)

# define parameters common to all sets
extras = {'w'            : -1.,
          'z_star'       : 1090., 
          'Tcmb'         : 2.725,
          'N_eff'        : 3.046, 
          'delta_c'      : 1.686, 
          'Y_he'         : 0.2477}
          