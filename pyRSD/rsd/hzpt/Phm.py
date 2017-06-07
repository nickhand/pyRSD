from . import HaloZeldovichBase, tools
from . import Cache, parameter, cached_property
from ... import pygcl, numpy as np

class HaloMatterBase(HaloZeldovichBase):
    """
    The halo-matter cross-correlation Phm, using HZPT
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
        super(HaloMatterBase, self).__init__(cosmo, z)

    @staticmethod
    def default_parameters():
        """
        The default parameters for P_hm, CF_hm

        References
        ----------
        These parameters are from:

        file: ``global_Phm_kmax-0.5_final.npz``
        directory: ``$RSD_DIR/SimCalibrations/HaloMatterHZPT/results``
        git hash: 72834b10d
        """
        d = {}

        # A0 power law
        d['_A0_amp']   = 751.9
        d['_A0_alpha'] = 1.657
        d['_A0_beta']  = 3.752
        d['_A0_run']   = 0.

        # R1 power law
        d['_R1_amp']   = 5.192
        d['_R1_alpha'] = -0.5739
        d['_R1_beta']  = 0.1616
        d['_R1_run']   = 0.

        # R1h power law
        d['_R1h_amp']   = 8.253
        d['_R1h_alpha'] = -0.8421
        d['_R1h_beta']  = -0.1345
        d['_R1h_run']   = 0.

        # R2h power law
        d['_R2h_amp']   = 3.053
        d['_R2h_alpha'] = -1.03
        d['_R2h_beta']  = -0.3596
        d['_R2h_run']   = 0.

        # R power law
        d['_R_amp']   = 16.9
        d['_R_alpha'] = -0.1185
        d['_R_beta']  = -1.067
        d['_R_run']   = 0

        return d

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def b1(self, val):
        """
        The linear bias
        """
        return val

    #---------------------------------------------------------------------------
    # A0 parameter
    #---------------------------------------------------------------------------
    @parameter
    def _A0_amp(self, val):
        return val

    @parameter
    def _A0_alpha(self, val):
        return val

    @parameter
    def _A0_beta(self, val):
        return val

    @parameter
    def _A0_run(self, val):
        return val

    #---------------------------------------------------------------------------
    # R parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R_amp(self, val):
        return val

    @parameter
    def _R_alpha(self, val):
        return val

    @parameter
    def _R_beta(self, val):
        return val

    @parameter
    def _R_run(self, val):
        return val

    #---------------------------------------------------------------------------
    # R1 parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R1_amp(self, val):
        return val

    @parameter
    def _R1_alpha(self, val):
        return val

    @parameter
    def _R1_beta(self, val):
        return val

    @parameter
    def _R1_run(self, val):
        return val

    #---------------------------------------------------------------------------
    # R1h parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R1h_amp(self, val):
        return val

    @parameter
    def _R1h_alpha(self, val):
        return val

    @parameter
    def _R1h_beta(self, val):
        return val

    @parameter
    def _R1h_run(self, val):
        return val

    #---------------------------------------------------------------------------
    # R2h parameter
    #---------------------------------------------------------------------------
    @parameter
    def _R2h_amp(self, val):
        return val

    @parameter
    def _R2h_alpha(self, val):
        return val

    @parameter
    def _R2h_beta(self, val):
        return val

    @parameter
    def _R2h_run(self, val):
        return val

    #---------------------------------------------------------------------------
    # cached parameters
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', 'b1', '_A0_amp', '_A0_alpha', '_A0_beta', '_A0_run')
    def A0(self):
        """
        Returns the A0 radius parameter

        Note: the units are power [(h/Mpc)^3]
        """
        return self._powerlaw(self._A0_amp, self._A0_alpha, self._A0_beta, self._A0_run)

    @cached_property('sigma8_z', 'b1', '_R1_amp', '_R1_alpha', '_R1_beta', '_R1_run')
    def R1(self):
        """
        Returns the R1 radius parameter

        Note: the units are length [Mpc/h]
        """
        return self._powerlaw(self._R1_amp, self._R1_alpha, self._R1_beta, self._R1_run)

    @cached_property('sigma8_z', 'b1', '_R1h_amp', '_R1h_alpha', '_R1h_beta', '_R1h_run')
    def R1h(self):
        """
        Returns the R1h radius parameter

        Note: the units are length [Mpc/h]
        """
        return self._powerlaw(self._R1h_amp, self._R1h_alpha, self._R1h_beta, self._R1h_run)

    @cached_property('sigma8_z', 'b1', '_R2h_amp', '_R2h_alpha', '_R2h_beta', '_R2h_run')
    def R2h(self):
        """
        Returns the R2h radius parameter

        Note: the units are length [Mpc/h]
        """
        return self._powerlaw(self._R2h_amp, self._R2h_alpha, self._R2h_beta, self._R2h_run)

    @cached_property('sigma8_z', 'b1', '_R_amp', '_R_alpha', '_R_beta', '_R_run')
    def R(self):
        """
        Returns the R radius parameter

        Note: the units are length [Mpc/h]
        """
        return self._powerlaw(self._R_amp, self._R_alpha, self._R_beta, self._R_run)

    def _powerlaw(self, A, alpha, beta, running):
        """
        Return a power law as a linear bias and sigma8(z) with the specified
        parameters
        """
        return A * (self.b1)**(alpha+running*np.log(self.b1)) * (self.sigma8_z/0.8)**beta


class HaloZeldovichPhm(HaloMatterBase):
    """
    The halo-matter cross-correlation Phm, using HZPT
    """
    def __call__(self, b1, k):
        """
        Return the total power, equal to the b1 * Zeldovich power + broadband
        correction

        Parameters
        ----------
        b1 : float
            the linear bias to compute the bias at
        k : array_like
            the wavenumbers in :math:`h/\mathrm{Mpc}` to compute the power at
        """
        # make sure sigma8 is set properly
        if self._driver.GetSigma8AtZ() != self.sigma8_z:
            self._driver.SetSigma8AtZ(self.sigma8_z)

        self.b1 = b1
        return self.broadband(k) + b1*self.zeldovich(k)

class HaloZeldovichCFhm(HaloMatterBase):
    """
    The dark matter - halo correlation function using Halo-Zel'dovich
    Perturbation Theory
    """
    zeldovich_class = pygcl.ZeldovichCF

    def broadband(self, r):
        """
        The correlation function broadband correction term

        This is given by the Fourier transform of the Pade function
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
        """
        return np.nan_to_num(self._driver(r)) # set any NaNs to zero

    def __call__(self, b1, r):
        """
        Return the total correlation

        Parameters
        ----------
        b1 : float
            the linear bias to compute correlation at
        r : array_like
            the separations in :math:`\mathrm{Mpc}/h` to correlation at
        """
        self.b1 = b1
        return self.broadband(r) + b1*self.zeldovich(r)
