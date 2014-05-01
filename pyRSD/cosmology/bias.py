"""
 bias.py
 pyPT: bias as a function of halo mass
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/24/2014
"""
import numpy as np

def bias_Tinker(sigmas, delta_c, delta_halo):
    """
    Return the halo bias for the Tinker form.
    
    Tinker, J., et al., 2010. ApJ 724, 878-886.
    http://iopscience.iop.org/0004-637X/724/2/878
    """

    y = np.log10(delta_halo)
    
    # get the parameters as a function of halo overdensity
    A = 1. + 0.24*y*np.exp(-(4./y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4)
    c = 2.4
    
    nu = delta_c / sigmas
    return 1. - A * (nu**a)/(nu**a + delta_c**a) + B*nu**b + C*nu**c
#end bias_Tinker

#-------------------------------------------------------------------------------

def bias_PS(sigmas, delta_c):
    """
    Return the halo bias for the Press-Schechter form, as derived by Mo & White
    1996.
    
    Mo, H. J., & White, S. D. M. 1996, MNRAS, 282, 347
    http://adsabs.harvard.edu/abs/1996MNRAS.282..347M
    
    Notes
    -----
    The PS mass function fails to reproduce the dark matter halo mass function
    found in simulations. Bias model overpredicts bias in the range 1 < nu < 3
    and underpredicts at lower masses.
    """
    nu = delta_c / sigmas
    return 1 + (nu**2 - 1)/delta_c
#end bias_PS

#-------------------------------------------------------------------------------
def bias_SMT(sigmas, delta_c):
    """
    Return the halo bias for the Sheth-Mo-Tormen form
    
    Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
    http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x
    
    Notes
    -----
    Model underpredicts the clustering of high-peak halos and overpredicts
    the asymptotic bias of low-mass objects. Derived using FOF halos.
    """
    a = 0.707
    b = 0.5
    c = 0.6
    nu = delta_c / sigmas
    
    sqrta = np.sqrt(a)
    term1 = sqrta*(a*nu**2) + sqrta*b*(a*nu**2)**(1-c)
    term2 = (a*nu**2)**c / ((a*nu**2)**c + b*(1-c)*(1-0.5*c))
    return 1 + 1./(sqrta*delta_c)*(term1 - term2)
#end bias_SMT

#-------------------------------------------------------------------------------
    
