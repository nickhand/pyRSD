from . import parameter
from .. import INTERP_KMIN, INTERP_KMAX
from ... import pygcl, numpy as np
from .._interpolate import RegularGridInterpolator, InterpolationDomainError
from . import HaloZeldovichP00, HaloZeldovichP01, HaloZeldovichP11, HaloZeldovichPhm 


def wrap_zeldovich(interpolator):
    """
    Wrap the call to :func:`__zeldovich__` to use an
    interpolation table
    """
    def wrapper(f):
        def wrapped(k):
            self = f.__self__
            try:
                if np.isscalar(k):
                    pts = [self.sigma8_z, k]
                else:
                    pts = np.vstack((np.repeat(self.sigma8_z, len(k)), k)).T
                return interpolator(pts)
            except InterpolationDomainError:
                return np.nan_to_num(self.zeldovich(k))
        
        wrapped.__wrapped__ = f
        return wrapped    
    return wrapper


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
            model = self.models.P00
        elif key == 'P01':
            model = self.models.P01
        elif key == 'P11':
            model = self.models.P11
        elif key == 'Phm':
            model = self.models.Phm
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
            model.zeldovich.SetSigma8AtZ(s8)
            grid_vals += list(model.__zeldovich__(ks))
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
    def __init__(self, cosmo, sigma8_z, f, interpolate=True, enhance_wiggles=False):
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
        enhance_wiggles : bool, optional
            whether to enhance the BAO wiggles
        """
        # the base Zel'dovich object
        self._base_zeldovich = pygcl.ZeldovichPS(cosmo, 0.)
        
        self.cosmo           = cosmo
        self.sigma8_z        = sigma8_z
        self.f               = f
        self.interpolate     = interpolate
        self.enhance_wiggles = enhance_wiggles
        
        # the interpolation table
        self.table = InterpolationTable(self)
    
        
    def _hasattr(self, m):
        """
        Internal utility function
        """
        return '_'+m in self.__dict__
    
    @parameter
    def interpolate(self, val):
        """
        If `True`, return the Zel'dovich power term from an interpolation table
        """
        for model in ['P00', 'P01', "P11", 'Phm']:
            if self._hasattr(model):
                m = getattr(self, model)
                
                if val:
                    m.__zeldovich__ = wrap_zeldovich(self.table[model])(m.__zeldovich__)
                else:
                    if hasattr(m.__zeldovich__, '__wrapped__'):
                        m.__zeldovich__ = m.__zeldovich__.__wrapped__
        
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
                m = getattr(self, model)
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
                m = getattr(self, model)
                m.f = val
                
    @property
    def enhance_wiggles(self):
        """
        Whether to enhance the BAO wiggles
        """
        return self._enhance_wiggles
        
    @enhance_wiggles.setter
    def enhance_wiggles(self, val):
        """
        Set `enhance_wiggles` for each existing model
        """
        self._enhance_wiggles = val
        
        for model in ['P00', 'P01', 'P11', 'Phm']:
            if self._hasattr(model):
                m = getattr(self, model)
                m.enhance_wiggles = val
        
    
    #--------------------------------------------------------------------------
    # the models
    #--------------------------------------------------------------------------
    @property
    def P00(self):
        """
        The P00 Halo Zel'dovich model
        """
        try:
            return self._P00
        except AttributeError:
            Pzel = pygcl.ZeldovichP00(self._base_zeldovich)
            self._P00 = HaloZeldovichP00(Pzel, self.sigma8_z, self.enhance_wiggles)
            if self.interpolate:
                self._P00.__zeldovich__ = wrap_zeldovich(self.table['P00'])(self._P00.__zeldovich__)
                
            return self._P00
            
    @property
    def P01(self):
        """
        The P01 Halo Zel'dovich model
        """
        try:
            return self._P01
        except AttributeError:
            Pzel = pygcl.ZeldovichP01(self._base_zeldovich)
            self._P01 = HaloZeldovichP01(Pzel, self.sigma8_z, self.f, self.enhance_wiggles)
            if self.interpolate:
                self._P01.__zeldovich__ = wrap_zeldovich(self.table['P01'])(self._P01.__zeldovich__)
            
            return self._P01
            
    @property
    def P11(self):
        """
        The P11 Halo Zel'dovich model
        """
        try:
            return self._P11
        except AttributeError:
            Pzel = pygcl.ZeldovichP11(self._base_zeldovich)
            self._P11 = HaloZeldovichP11(Pzel, self.sigma8_z, self.f, self.enhance_wiggles)
            if self.interpolate:
                self._P11.__zeldovich__ = wrap_zeldovich(self.table['P11'])(self._P11.__zeldovich__)
                
            return self._P11
            
    @property
    def Phm(self):
        """
        The Phm Halo Zel'dovich model
        """
        try:
            return self._Phm
        except AttributeError:
            Pzel = pygcl.ZeldovichP00(self._base_zeldovich)
            self._Phm = HaloZeldovichPhm(Pzel, self.sigma8_z, self.enhance_wiggles)
            if self.interpolate:
                self._Phm.__zeldovich__ = wrap_zeldovich(self.table['Phm'])(self._Phm.__zeldovich__)
                
            return self._Phm
            
        
            
            
        
        
    
