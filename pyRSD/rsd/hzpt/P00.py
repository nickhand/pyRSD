from . import HaloZeldovichBase, tools
from ... import pygcl, numpy as np

class HaloZeldovichP00(HaloZeldovichBase):
    """
    The dark matter auto-spectrum P00 using Halo-Zel'dovich Perturbation Theory
    """
    zeldovich_class = pygcl.ZeldovichP00

    def __init__(self, cosmo, z):
        """
        Parameters
        ----------
        cosmo : cosmology.Cosmology, pygcl.Cosmology
            the cosmology instance
        z : float
            the redshift; this determines the values of sigma8(z) to
            use in the HZPT equations
        """
        super(HaloZeldovichP00, self).__init__(cosmo, z)

class HaloZeldovichCF00(HaloZeldovichBase):
    """
    The dark matter correlation function using Halo-Zel'dovich Perturbation Theory
    """
    zeldovich_class = pygcl.ZeldovichCF

    def __init__(self, cosmo, z):
        """
        Parameters
        ----------
        cosmo : cosmology.Cosmology, pygcl.Cosmology
            the cosmology instance
        z : float
            the redshift; this determines the values of sigma8(z) to
            use in the HZPT equations
        """
        super(HaloZeldovichCF00, self).__init__(cosmo, z)

    def broadband(self, r):
        """
        The correlation function broadband correction term

        This is given by the Fourier transform of the Pade function

        Parameters
        ----------
        r : float, array_like
            the separation array, in units of :math:`\mathrm{Mpc}/h`

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
    def zeldovich(self, r):
        """
        Return the Zel'dovich correlation at the specified `r`

        Parameters
        ----------
        r : float, array_like
            the separation array, in units of :math:`\mathrm{Mpc}/h`
        """
        return np.nan_to_num(self._driver(r)) # set any NaNs to zero

    def __call__(self, r):
        """
        Return the total correlation function
        """
        return self.broadband(r) + self.zeldovich(r)
