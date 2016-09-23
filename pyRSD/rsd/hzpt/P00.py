from . import HaloZeldovichBase, tools
from ... import pygcl, numpy as np

class HaloZeldovichP00(HaloZeldovichBase):
    """
    The dark matter auto-spectrum P00 using Halo-Zel'dovich Perturbation Theory
    """ 
    def __init__(self, zel, sigma8_z, enhance_wiggles=False):
        """
        Parameters
        ----------
        zel : ZeldovichP00
            the Zel'dovich power spectrum
        sigma8_z : float
            The desired sigma8 to compute the power at
        enhance_wiggles : bool, optional (`False`)
            using the Hy1 model from arXiv:1509.02120, enhance the wiggles
            of pure HZPT
        """   
        super(HaloZeldovichP00, self).__init__(zel, enhance_wiggles)
        self.sigma8_z = sigma8_z
        
    @classmethod
    def from_cosmo(cls, cosmo, sigma8_z, enhance_wiggles=False):
        """
        Initialize from a cosmology instance
        """
        Pzel = pygcl.ZeldovichP00(cosmo, 0.)
        return cls(Pzel, sigma8_z, enhance_wiggles)
                        
class HaloZeldovichCF00(HaloZeldovichBase):
    """
    The dark matter correlation function using Halo-Zel'dovich Perturbation Theory
    """ 
    def __init__(self, zel, sigma8_z):
        """
        Parameters
        ----------
        zel : ZeldovichCF
            the Zel'dovich correlation function
        sigma8_z : float
            The desired sigma8 to compute the power at
        """ 
        super(HaloZeldovichCF00, self).__init__(zel, False)
        self.sigma8_z = sigma8_z
          
    @classmethod
    def from_cosmo(cls, cosmo, sigma8_z):
        """
        Initialize from a cosmology instance
        """
        zel = pygcl.ZeldovichCF(cosmo, 0)
        return cls(zel, sigma8_z)
        
    def __broadband__(self, r):
        """
        The broadband power correction as given by Eq. 7 in arXiv:1501.07512.
        """
        A0, R, R1, R1h, R2h = self.A0, self.R, self.R1, self.R1h, self.R2h
        
        # define some values
        S = np.sqrt(R1h**4 - 4*R2h**4)
        norm = -A0*np.exp(-r/R) / (4*np.pi*r*R**2)
        A = R**2*(-2*R2h**4 + R1**2*(R1h**2-S)) + R2h**4*(R1h**2-S) + R1**2*(-R1h**4+2*R2h**4+R1h**2*S)
        A /= 2*R2h**4*S
        B = R2h**4*(R1h**2+S) - R1**2*(R1h**4-2*R2h**4+R1h**2*S) + R**2*(-2*R2h**4+R1**2*(R1h**2+S))
        B *= -1
        B /= 2*R2h**4*S
        
        num_term1 = 1. - (R1/R)**2
        num_term2 = A * np.exp(r*(1./R - np.sqrt(0.5*(R1h**2 - S)) / R2h**2))
        num_term3 = B * np.exp(r*(1./R - np.sqrt(0.5*(R1h**2 + S)) / R2h**2))
        denom = (1 - (R1h/R)**2 + (R2h/R)**4)
        
        return norm * (num_term1 + num_term2 + num_term3) / denom
        
    @tools.unpacked
    def __zeldovich__(self, r, ignore_interpolated=False):
        """
        Return the Zel'dovich correlation at the specified `r`
        """
        return np.nan_to_num(self.zeldovich(r)) # set any NaNs to zero
            
    def __call__(self, r):
        """
        Return the total correlation
        """
        return self.__broadband__(r) + self.__zeldovich__(r)
