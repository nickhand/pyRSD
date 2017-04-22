from pyRSD.rsd._cache import parameter, cached_property
from pyRSD.rsd import tools, BiasedSpectrum
from pyRSD.rsd.window import WindowTransfer
from pyRSD import numpy as np

from .fog_kernels import FOGKernel
from . import Pgal

from scipy.special import legendre
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import contextlib
import warnings

          
class GalaxySpectrum(BiasedSpectrum):
    """
    The galaxy redshift space power spectrum, a subclass of  
    :class:`~pyRSD.rsd.BiasedSpectrum` for biased redshift space power spectra
    """    
    def __init__(self, fog_model='modified_lorentzian', 
                       use_so_correction=False,
                       **kwargs):
        """
        Initialize the GalaxySpectrum
                           
        Additional `GalaxySpectrum`-specific parameters are listed
        below. Keywords accepted by :class:`pyRSD.rsd.BiasedSpectrum`
        and :class:`pyRSD.rsd.DMSpectrum`. See their documenation for further 
        details.
        
        Parameters
        ----------
        fog_model : str, optional
            the string specifying the FOG model to use; one of 
            ['modified_lorentzian', 'lorentzian', 'gaussian']. 
            Default is 'modified_lorentzian'
        use_so_correction : bool, optional
            Boost the centrals auto spectrum with a correction
            accounting for extra structure around centrals due
            to SO halo finders; default is `False`
        """
        # the base class
        super(GalaxySpectrum, self).__init__(**kwargs)
        
        # the underlying driver
        self._Pgal = Pgal(self)
        
        # set the defaults
        self.fog_model     = fog_model
        self.include_2loop = False
        self.fs            = 0.10
        self.fcB           = 0.08
        self.fsB           = 0.40
        self.b1_cA         = 1.85
        self.b1_cB         = 2.8
        self.b1_sA         = 2.6
        self.b1_sB         = 3.6
        self.sigma_c       = 1.
        self.sigma_s       = 5.
        self.sigma_sA      = 4.2
        self.sigma_sB      = 6.
        self.NcBs          = 3e4
        self.NsBsB         = 9e4
        self.N             = 0.
        
        # SO corretion
        self.use_so_correction = use_so_correction
        self.f_so              = 0.
        self.sigma_so          = 0.
        
    def default_params(self):
        """
        Return a GalaxyPowerParameters instance holding the default
        model parameters configuration
        
        The model associated with the parameter is ``self``
        """
        from pyRSD.rsdfit.theory import GalaxyPowerParameters
        return GalaxyPowerParameters.from_defaults(model=self)
    
    @contextlib.contextmanager
    def use_cache(self):
        """
        Cache repeated calls to functions defined in this class, assuming
        constant `k` and `mu` input values
        """
        from pyRSD.rsd.tools import cache_on, cache_off
        
        try:
            cache_on()
            yield
        except:
            raise
        finally:
            cache_off()
        
    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------  
    @parameter
    def f_so(self, val):
        """
        The fraction of satellites in SO halo finders compared to FOF
        """
        return val
        
    @parameter
    def sigma_so(self, val):
        """
        The FOG velocity dispersion for type A centrals in Mpc/h, accounting
        for FOG from SO/FOF differences around central type A galaxies
        """
        return val
              
    @parameter
    def fog_model(self, val):
        """
        Function to return the FOG suppression factor, which reads in a 
        single variable `x = k \mu \sigma`
        """
        allowable = ['modified_lorentzian', 'lorentzian', 'gaussian']
        if val not in allowable:
            raise ValueError("`fog_model` must be one of %s" %allowable)
        
        return val
        
    @parameter
    def fs(self, val):
        """
        The satellite fraction, fs = N_sat / N_gal 
        """
        return val

    @parameter
    def fcB(self, val):
        """
        The centrals with sats (cB) fraction, fcB = N_cB / N_cen 
        """
        return val
    
    @parameter
    def fsB(self, val):
        """
        The satellite with sats fraction, fsB = N_sB / N_sat
        """
        return val
        
    @parameter
    def b1_cA(self, val):
        """
        The linear bias factor for the centrals with no sats in same halo.
        """
        return val
        
    @parameter
    def b1_cB(self, val):
        """
        The linear bias factor for the centrals with sats in same halo.
        """
        return val
        
    @parameter
    def b1_sA(self, val):
        """
        The linear bias factor for satellites with no other sats in same halo.
        """
        return val
        
    @parameter
    def b1_sB(self, val):
        """
        The linear bias factor for satellites with other sats in same halo.
        """
        return val
    
    @parameter
    def sigma_c(self, val):
        """
        The FOG velocity dispersion for centrals in Mpc/h
        """
        return val
                
    @parameter
    def sigma_s(self, val):
        """
        The FOG velocity dispersion for satellites in Mpc/h
        """
        return val
        
    @parameter
    def sigma_sA(self, val):
        """
        The FOG velocity dispersion for "type A" satellites in Mpc/h
        """
        return val
        
    @parameter
    def sigma_sB(self, val):
        """
        The FOG velocity dispersion for "type B" satellites in Mpc/h
        """
        return val
    
    @parameter
    def NcBs(self, val):
        """
        Constant for the P_cBs 1-halo term
        """
        return val
    
    @parameter
    def NsBsB(self, val):
        """
        Constant for the P_sBsB 1-halo term
        """
        return val
        
    @parameter
    def N(self, val):
        """
        Constant offset to model, set to 0 by default
        """
        return val
                   
    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("fog_model")
    def FOG(self):
        """
        Return the FOG function
        """
        return FOGKernel.factory(self.fog_model)
        
    @cached_property('fcB', 'b1_cB', 'b1_cA')
    def b1_c(self):
        """
        The linear bias factor for all centrals. This is not a free parameter, 
        but is computed as weighted mean of b1_cA and b1_cB.
        """
        return self.fcB*self.b1_cB + (1.-self.fcB)*self.b1_cA

    @cached_property('fsB', 'b1_sB', 'b1_sA')
    def b1_s(self):
        """
        The linear bias factor for all satellites. This is not a free parameter, 
        but is computed as weighted mean of b1_sA and b1_sB.
        """
        return self.fsB*self.b1_sB + (1.-self.fsB)*self.b1_sA
    
    def initialize(self):
        """
        Initialize the underlying splines, etc
        """
        k = 0.5*(self.kmin+self.kmax)
        return self.Pgal(k, 0.5)
                
    #---------------------------------------------------------------------------
    # centrals power spectrum
    #--------------------------------------------------------------------------- 
    @tools.alcock_paczynski
    def Pgal_cAcA(self, k, mu, flatten=False):
        """
        The auto power spectrum of type A centrals
        """
        toret = self._Pgal['Pcc']['PcAcA'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_cAcB(self, k, mu, flatten=False):
        """
        The cross power spectrum of type A and type B centrals
        """
        toret = self._Pgal['Pcc']['PcAcB'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_cBcB(self, k, mu, flatten=False):
        """
        The auto power spectrum of type B centrals
        """
        toret = self._Pgal['Pcc']['PcBcB'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_cc(self, k, mu, flatten=False):
        """
        The auto power spectrum of all centrals
        """
        toret = self._Pgal['Pcc'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    # central-satellite cross spectrum
    #---------------------------------------------------------------------------
    @tools.alcock_paczynski
    def Pgal_cAsA(self, k, mu, flatten=False):
        """
        The cross power spectrum of type A centrals and type A sats
        """
        toret = self._Pgal['Pcs']['PcAsA'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
    
    @tools.alcock_paczynski
    def Pgal_cAsB(self, k, mu, flatten=False):
        """
        The cross power spectrum of type A centrals and type B sats
        """
        toret = self._Pgal['Pcs']['PcAsB'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_cBsA(self, k, mu, flatten=False):
        """
        The cross power spectrum of type B centrals and type A sats
        """
        toret = self._Pgal['Pcs']['PcBsA'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_cBsB(self, k, mu, flatten=False):
        """
        The cross power spectrum of type B centrals and type B sats
        """
        toret = self._Pgal['Pcs']['PcBsB'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_cs(self, k, mu, flatten=False):
        """
        The cross power spectrum of centrals and satellites
        """
        toret = self._Pgal['Pcs'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    # satellites auto spectrum
    #---------------------------------------------------------------------------
    @tools.alcock_paczynski
    def Pgal_sAsA(self, k, mu, flatten=False):
        """
        The auto power spectrum of type A satellites
        """
        toret = self._Pgal['Pss']['PsAsA'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_sAsB(self, k, mu, flatten=False):
        """
        The cross power spectrum of type A and type B satellites
        """
        toret = self._Pgal['Pss']['PsAsB'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_sBsB(self, k, mu, flatten=False):
        """
        The auto power spectrum of type B satellites
        """
        toret = self._Pgal['Pss']['PsBsB'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.alcock_paczynski
    def Pgal_ss(self, k, mu, flatten=False):
        """
        The auto power spectrum of all satellites
        """
        toret = self._Pgal['Pss'](k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
                    
    #---------------------------------------------------------------------------
    # total galaxy P(k,mu)
    #---------------------------------------------------------------------------
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def Pgal(self, k, mu, flatten=False):
        """
        The total redshift-space galaxy power spectrum, combining the individual
        terms.
        
        Parameters
        ----------
        k : float, array_like
            The wavenumbers to evaluate the power spectrum at, in `h/Mpc`
        mu : float, array_like
            The cosine of the angle from the line of sight. If a float is provided,
            the value is used for all input `k` values. If array-like and `mu` has
            the same shape as `k`, the power at each (k,mu) pair is returned. If
            `mu` has a shape different than `k`, the returned power has shape
            ``(len(k), len(mu))``.
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `len(k) * len(mu)`
        """                
        toret = self._Pgal(k, mu)
        return toret if not flatten else np.ravel(toret, order='F')
        
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def Pgal_derivative_k(self, k, mu):
        """
        The derivative with respect to `k_AP`
        """
        return self._Pgal.derivative_k(k, mu)
        
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def Pgal_derivative_mu(self, k, mu):
        """
        The derivative with respect to `mu_AP`
        """
        return self._Pgal.derivative_mu(k, mu)
        
    def Pgal_poles(self, k, poles, flatten=False, Nmu=41):
        """
        Return the multipole moments specified by `poles`, where `poles` is a
        list of integers, i.e., [0, 2, 4]
        
        Parameter
        ---------
        k : float, array_like
            The wavenumbers to evaluate the power spectrum at, in `h/Mpc`
        poles : init, array_like
            The `ell` values of the multipole moments
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `len(k) * len(poles)`
            
        Returns
        -------
        poles : array_like
            returns array for each ell value in ``poles``
        """
        scalar = np.isscalar(poles)
        if scalar: poles = [poles]
          
        mus = np.linspace(0., 1., Nmu)
        Pkmus = self.Pgal(k, mus)
        
        if len(poles) != len(k):
            toret = ()
            for ell in poles:
                kern = (2*ell+1.)*legendre(ell)(mus)
                val = np.array([simps(kern*d, x=mus) for d in Pkmus])
                toret += (val,)
                          
            if scalar:
                return toret[0]
            else:
                return toret if not flatten else np.ravel(toret, order='F')
        else:
            kern = np.asarray([(2*ell+1.)*legendre(ell)(mus) for ell in poles])
            return np.array([simps(d, x=mus) for d in kern*Pkmus])
                            
    @tools.monopole
    def Pgal_mono(self, k, mu, **kwargs):
        """
        The total redshift-space galaxy monopole (ell=0) moment
        """
        return self.Pgal(k, mu, **kwargs)

    @tools.quadrupole
    def Pgal_quad(self, k, mu, **kwargs):
        """
        The total redshift-space galaxy quadrupole (ell=2) moment
        """
        return self.Pgal(k, mu, **kwargs)

    @tools.hexadecapole
    def Pgal_hexadec(self, k, mu, **kwargs):
        """
        The total redshift-space galaxy hexadecapole (ell=4) moment
        """
        return self.Pgal(k, mu, **kwargs)
        
    @tools.tetrahexadecapole
    def Pgal_tetrahex(self, k, mu, **kwargs):
        """
        The total redshift-space galaxy tetrahexadecapole (ell=6) moment
        """
        return self.Pgal(k, mu, **kwargs)
        
    def from_transfer(self, transfer, flatten=False, **kws):
        """
        Return the power (either P(k,mu) or multipoles), accounting for 
        discrete binning effects using the input transfer function
    
        Parameter
        ---------
        transfer : PkmuTransfer, PolesTransfer, WindowTransfer
            the transfer class which accounts for the discrete binning effects
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `Nk * Nmu` or `Nk * Nell`
        """  
        grid = transfer.grid
        
        # check some bounds for window convolution  
        from pyRSD.rsd.window import WindowTransfer
        if isinstance(transfer, WindowTransfer):
            
            # check k-range inconsistency
            kmin, kmax = transfer.kmin, transfer.kmax
            if (kmin < self.kmin).any():
                warnings.warn("min k of window transfer (%s) is less than model's kmin (%.2e)" %(kmin, self.kmin))
            if (kmax > self.kmax).any():
                warnings.warn("max k of window transfer (%s) is greater than model's kmax (%.2e)" %(kmax, self.kmax))
                
            # check bad values
            if self.kmin > 5e-3:
                warnings.warn("doing window convolution with dangerous model kmin (%.2e)" %self.kmin)
            if self.kmax < 0.5:
                warnings.warn("doing window convolution with dangerous model kmax (%.2e)" %self.kmax)

        power = self.Pgal(grid.k[grid.notnull], grid.mu[grid.notnull])
        transfer.power = power        
        return transfer(flatten=flatten, **kws)