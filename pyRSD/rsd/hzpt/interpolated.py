from . import parameter
from .. import INTERP_KMIN, INTERP_KMAX
from ... import pygcl, numpy as np
from .._interpolate import RegularGridInterpolator, InterpolationDomainError
from . import HaloZeldovichP00, HaloZeldovichP01, HaloZeldovichP11, HaloZeldovichPhm


def zeldovich_from_table(self, interpolator, k):
    """
    Compute the Zel'dovich term from the table
    """
    try:
        if np.isscalar(k):
            pts = [self.sigma8_z, k]
        else:
            pts = np.vstack((np.repeat(self.sigma8_z, len(k)), k)).T
        return interpolator(pts)
    except InterpolationDomainError:
        return np.nan_to_num(self.zeldovich(k))

class InterpolationTable(dict):
    """
    Dict that returns an interpolation table for
    the given Zel'dovich terms
    """
    grid = {}
    grid['sigma8_z'] = np.linspace(0.3, 1.0, 100)
    grid['k'] = np.logspace(np.log10(INTERP_KMIN), np.log10(INTERP_KMAX), 300)

    def __init__(self, models):
        self.models = models

    def __missing__(self, key):
        """
        Evaluate the Zeldovich terms and store in an interpolation table.

        Notes
        -----
        This does not depend on redshift, as we are interpolating as a function
        of sigma8(z)
        """
        if key == 'P00':
            model = self.models._P00
        elif key == 'P01':
            model = self.models._P01
        elif key == 'P11':
            model = self.models._P11
        elif key == 'Phm':
            model = self.models._Phm
        else:
            raise KeyError("key '%s' not understood" %key)

        # save the original sigma8_z to restore later
        original_s8z = model.sigma8_z

        # the interpolation grid points
        sigma8s = self.grid['sigma8_z']
        ks      = self.grid['k']

        # get the grid values
        grid_vals = []
        for i, s8 in enumerate(sigma8s):
            model._driver.SetSigma8AtZ(s8)
            grid_vals += list(model.zeldovich(ks))
        grid_vals = np.array(grid_vals).reshape((len(sigma8s), len(ks)))

        # create the interpolator
        model.sigma8_z = original_s8z
        interpolator = RegularGridInterpolator((sigma8s, ks), grid_vals)

        super(InterpolationTable, self).__setitem__(key, interpolator)
        return interpolator


class InterpolatedHZPTModels(object):
    """
    Class to handle interpolating HZPT models
    """
    def __init__(self, cosmo, sigma8_z, f, interpolate=True):
        """
        Parameters
        ----------
        cosmo : pygcl.Cosmology
            the cosmology instance
        sigma8_z : float
            the desired sigma8 to compute the power at
        f : float
            the growth rate
        interpolate : bool, optional
            whether to turn on interpolation
        """
        # the base Zel'dovich object
        self._base_zeldovich = pygcl.ZeldovichPS(cosmo, 0.)

        self.cosmo           = cosmo
        self.sigma8_z        = sigma8_z
        self.f               = f
        self.interpolate     = interpolate

        # the interpolation table
        self.table = InterpolationTable(self)

    def _hasattr(self, m):
        """
        Internal utility function
        """
        return '_%s_driver' %m in self.__dict__

    @parameter
    def interpolate(self, val):
        """
        If `True`, return the Zel'dovich power term from an interpolation table
        """
        return val

    @property
    def sigma8_z(self):
        """
        The value of sigma8 at z
        """
        return self._sigma8_z

    @sigma8_z.setter
    def sigma8_z(self, val):
        """
        Set `sigma8_z` for each existing model
        """
        self._sigma8_z = val
        self._base_zeldovich.SetSigma8AtZ(val)

        for model in ['P00', 'P01', 'P11', 'Phm']:
            if self._hasattr(model):
                m = getattr(self, '_'+model)
                m.sigma8_z = val

    @property
    def f(self):
        """
        The value of the growth rate
        """
        return self._f

    @f.setter
    def f(self, val):
        """
        Set `f` for the models that need it
        """
        self._f = val
        for model in ['P01', 'P11']:
            if self._hasattr(model):
                m = getattr(self, '_'+model)
                m.f = val

    #--------------------------------------------------------------------------
    # the models
    #--------------------------------------------------------------------------
    @property
    def _P00(self):
        """
        The P00 Halo Zel'dovich model
        """
        try:
            return self._P00_driver
        except AttributeError:
            Pzel = pygcl.ZeldovichP00(self._base_zeldovich)
            self._P00_driver = HaloZeldovichP00.from_zeldovich(Pzel, self.sigma8_z)
            return self._P00_driver

    @property
    def _P01(self):
        """
        The P01 Halo Zel'dovich model
        """
        try:
            return self._P01_driver
        except AttributeError:
            Pzel = pygcl.ZeldovichP01(self._base_zeldovich)
            self._P01_driver = HaloZeldovichP01.from_zeldovich(Pzel, self.sigma8_z, self.f)
            return self._P01_driver

    @property
    def _P11(self):
        """
        The P11 Halo Zel'dovich model
        """
        try:
            return self._P11_driver
        except AttributeError:
            Pzel = pygcl.ZeldovichP11(self._base_zeldovich)
            self._P11_driver = HaloZeldovichP11.from_zeldovich(Pzel, self.sigma8_z, self.f)
            return self._P11_driver

    @property
    def _Phm(self):
        """
        The Phm Halo Zel'dovich model
        """
        try:
            return self._Phm_driver
        except AttributeError:
            Pzel = pygcl.ZeldovichP00(self._base_zeldovich)
            self._Phm_driver = HaloZeldovichPhm.from_zeldovich(Pzel, self.sigma8_z)
            return self._Phm_driver

    #--------------------------------------------------------------------------
    # the callables
    #--------------------------------------------------------------------------
    def P00(self, k):
        """
        Return HZPT P00
        """
        m = self._P00
        if m._driver.GetSigma8AtZ() != m.sigma8_z:
            m._driver.SetSigma8AtZ(m.sigma8_z)

        if self.interpolate:
            zel = zeldovich_from_table(m, self.table['P00'], k)
        else:
            zel = m.zeldovich(k)
        return m.broadband(k) + zel

    def P01(self, k):
        """
        Return HZPT P01
        """
        m = self._P01
        if m._driver.GetSigma8AtZ() != m.sigma8_z:
            m._driver.SetSigma8AtZ(m.sigma8_z)

        if self.interpolate:
            zel = zeldovich_from_table(m, self.table['P01'], k)
        else:
            zel = m.zeldovich(k)
        return m.broadband(k) + 2*m.f*zel

    def P11(self, k):
        """
        Return HZPT P11
        """
        m = self._P11
        if m._driver.GetSigma8AtZ() != m.sigma8_z:
            m._driver.SetSigma8AtZ(m.sigma8_z)

        if self.interpolate:
            zel = zeldovich_from_table(m, self.table['P11'], k)
        else:
            zel = m.zeldovich(k)
        return m.broadband(k) + m.f**2 * zel

    def Phm(self, b1, k):
        """
        Return HZPT Phm
        """
        m = self._Phm
        if m._driver.GetSigma8AtZ() != m.sigma8_z:
            m._driver.SetSigma8AtZ(m.sigma8_z)

        m.b1 = b1
        if self.interpolate:
            zel = zeldovich_from_table(m, self.table['Phm'], k)
        else:
            zel = m.zeldovich(k)
        return m.broadband(k) + b1 * zel
