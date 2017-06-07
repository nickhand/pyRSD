from .._cache import Cache, parameter, cached_property
from ... import numpy as np, pygcl
from .. import tools, cosmology

class HaloZeldovichBase(Cache):
    """
    Base class to represent a Halo Zel'dovich power spectrum or
    correlation function
    """
    zeldovich_class = None

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
        # make sure cosmo is set properly
        if isinstance(cosmo, cosmology.Cosmology):
            cosmo = cosmo.to_class()
        if not isinstance(cosmo, (cosmology.Cosmology, pygcl.Cosmology)):
            raise TypeError("'cosmo' must be a cosmology.Cosmology or pygcl.Cosmology object")
        self.cosmo = cosmo

        # update the normalization
        self.sigma8_z = cosmo.Sigma8_z(z)

        # default base parameters
        self.update(**self.default_parameters())

        # try to turn on the low-k approximation
        # this will fail for correlation function (ok)
        try: self._driver.SetLowKApprox()
        except: pass

    @classmethod
    def from_zeldovich(cls, zel, *args):
        """
        Initialize the class from a Zel'dovich object instance
        """
        toret = Cache.__new__(cls)
        c = zel.GetCosmology()
        toret.cosmo = pygcl.Cosmology(c.GetParams())
        toret._driver = zel

        # set sigma8 and f
        if len(args) not in [1, 2]:
            raise ValueError("please specify either sigma8_z or sigma8_z and f")
        toret.sigma8_z = args[0]
        if len(args) == 2:
            toret.f = args[1]

        # default base parameters
        toret.update(**toret.default_parameters())

        # try to turn on the low-k approximation
        # this will fail for correlation function (ok)
        try: toret._driver.SetLowKApprox()
        except: pass

        return toret

    @staticmethod
    def default_parameters():
        """
        Set the default parameters for P00/P01 HZPT

        References
        ----------
        These parameters are from:

        file: ``P00_P01_CF_rmin-0.3_kmax-0.5.npz``
        directory: ``$RSD_DIR/SimCalibrations/MatterHZPT/results``
        git hash: aed2b025
        """
        d = {}
        d['_A0_amp']    = 707.8
        d['_A0_alpha']  = 3.654
        d['_R_amp']     = 31.76
        d['_R_alpha']   = 0.1257
        d['_R1_amp']    = 3.243
        d['_R1_alpha']  = 0.3729
        d['_R1h_amp']   = 3.768
        d['_R1h_alpha'] = -0.1043
        d['_R2h_amp']   = 1.699
        d['_R2h_alpha'] = 0.4224
        return d

    @property
    def _driver(self):
        """
        The driver object that computes the Zel'dovich term
        """
        try:
            return self.__driver
        except:
            assert self.zeldovich_class is not None, "please set 'zeldovich_class'"
            self.__driver = self.zeldovich_class(self.cosmo, 0.)
            return self.__driver

    @_driver.setter
    def _driver(self, val):
        self.__driver = val

    @parameter
    def cosmo(self, val):
        """
        The cosmology parameter object
        """
        return val

    @property
    def z(self):
        """
        The redshift; this should only be set when initializing
        """
        return NotImplemented

    @z.setter
    def z(self, val):
        raise AttributeError("to change the redshift, set the `sigma8_z` attribute")

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    @parameter
    def sigma8_z(self, val):
        """
        The sigma8 value at z
        """
        if val <= 0.: raise ValueError("`sigma8_z` must be positive")
        self._driver.SetSigma8AtZ(val)
        return val

    #---------------------------------------------------------------------------
    # broadband power-law parameters
    #---------------------------------------------------------------------------
    @parameter
    def _A0_amp(self, val):
        return val

    @parameter
    def _A0_alpha(self, val):
        return val

    @parameter
    def _R_amp(self, val):
        return val

    @parameter
    def _R_alpha(self, val):
        return val

    @parameter
    def _R1_amp(self, val):
        return val

    @parameter
    def _R1_alpha(self, val):
        return val

    @parameter
    def _R1h_amp(self, val):
        return val

    @parameter
    def _R1h_alpha(self, val):
        return val

    @parameter
    def _R2h_amp(self, val):
        return val

    @parameter
    def _R2h_alpha(self, val):
        return val

    #---------------------------------------------------------------------------
    # broadband model parameters
    #---------------------------------------------------------------------------
    @cached_property('sigma8_z', '_A0_amp', '_A0_alpha')
    def A0(self):
        """
        Returns the A0 radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are power [(h/Mpc)^3]
        """
        return self._A0_amp*(self.sigma8_z/0.8)**self._A0_alpha

    @cached_property('sigma8_z', '_R_amp', '_R_alpha')
    def R(self):
        """
        Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R_amp*(self.sigma8_z/0.8)**self._R_alpha

    @cached_property('sigma8_z', '_R1_amp', '_R1_alpha')
    def R1(self):
        """
        Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1_amp*(self.sigma8_z/0.8)**self._R1_alpha

    @cached_property('sigma8_z', '_R1h_amp', '_R1h_alpha')
    def R1h(self):
        """
        Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R1h_amp*(self.sigma8_z/0.8)**self._R1h_alpha

    @cached_property('sigma8_z', '_R2h_amp', '_R2h_alpha')
    def R2h(self):
        """
        Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

        Note: the units are length [Mpc/h]
        """
        return self._R2h_amp*(self.sigma8_z/0.8)**self._R2h_alpha

    #---------------------------------------------------------------------------
    # function calls
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Update any attributes with the specified values
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def broadband(self, k):
        r"""
        The broadband power in units of :math:`(\mathrm{Mpc}/h)^3`

        The functional form is given by:

        .. math::

            P_\mathrm{BB} = A_0 F(k) \left[ \frac{1 + (k R_1)^2}{1 + (k R_{1h})^2 + (k R_{2h})^4} \right],

        as given by Eq. 1 in arXiv:1501.07512.

        Parameters
        ----------
        k : float, array_like
            the wavenumber in units of :math:`h/\mathrm{Mpc}`
        """
        F = 1. - 1./(1. + (k*self.R)**2)
        return F*self.A0*(1 + (k*self.R1)**2) / (1 + (k*self.R1h)**2 + (k*self.R2h)**4)

    @tools.unpacked
    def zeldovich(self, k):
        """
        Return the Zel'dovich power term at the specified `k`

        Parameters
        ----------
        k : float, array_like
            the wavenumber in units of :math:`h/\mathrm{Mpc}`
        """
        return np.nan_to_num(self._driver(k)) # set any NaNs to zero

    def __call__(self, k):
        """
        Return the total power at the specified `k`

        The total power is equal to the Zel'dovich power + broadband term

        Parameters
        ----------
        k : float, array_like
            the wavenumber in units of :math:`h/\mathrm{Mpc}`
        """
        # make sure sigma8 is set properly
        if self._driver.GetSigma8AtZ() != self.sigma8_z:
            self._driver.SetSigma8AtZ(self.sigma8_z)

        return self.broadband(k) + self.zeldovich(k)

from .P00 import HaloZeldovichP00, HaloZeldovichCF00
from .P01 import HaloZeldovichP01
from .P11 import HaloZeldovichP11
from .Phm import HaloZeldovichPhm, HaloZeldovichCFhm
from .interpolated import InterpolatedHZPTModels
