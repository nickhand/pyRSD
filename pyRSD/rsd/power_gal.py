from ._cache import parameter, cached_property
from . import power_biased, tools
from .. import numpy as np, sys
from scipy.special import legendre
from scipy.integrate import simps

#-------------------------------------------------------------------------------
# finger-of-god models
#-------------------------------------------------------------------------------
def fog_modified_lorentzian(x):
    """
    Modified Lorentzian FOG model 
    """
    return 1./(1 + 0.5*x**2)**2

def fog_lorentzian(x):
    """
    Lorentzian FOG model 
    """
    return 1./(1 + 0.5*x**2)

def fog_gaussian(x):
    """
    Gaussian FOG model 
    """
    return np.exp(-0.5*x**2)
    
#-------------------------------------------------------------------------------
# tools
#-------------------------------------------------------------------------------
class SOCorrection(object):
    """
    Class to manage the handling of `sigma_c` when using an SO correction
    """
    def __init__(self, model):
        self.model = model
        self._sigma_c = self.model.sigma_c
        
    def __enter__(self):
        
        if self.model.use_so_correction:
            self.model.sigma_c = 0
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.sigma_c = self._sigma_c
        
class ZeroShotNoise(object):
    """
    Class to manage the handling of `N` when computing component spectra
    """
    def __init__(self, model):
        self.model = model
        self._N = self.model.N
        
    def __enter__(self):
        self.model.N = 0
        return self._N

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.N = self._N
    

def compute_component_spectra(x, model, k, mu):
    
    params, name = x
    model.update(**params)
    return getattr(model, name)(k, mu)
    
class GalaxySpectrum(power_biased.BiasedSpectrum):
    """
    The galaxy redshift space power spectrum, a subclass of the `BiasedSpectrum`
    for biased redshift space power spectra
    """    
    def __init__(self, fog_model='modified_lorentzian', 
                       use_so_correction=False,
                       **kwargs):
        """
        Additional `GalaxySpectrum`-specific parameters:
        
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
        # initalize the dark matter power spectrum
        super(GalaxySpectrum, self).__init__(**kwargs)
        
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
    def fog_function(self):
        """
        Return the FOG function
        """
        return getattr(sys.modules[__name__], 'fog_%s' %self.fog_model)
        
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
    
    #---------------------------------------------------------------------------
    # utility functions
    #---------------------------------------------------------------------------
    def initialize(self):
        """
        Initialize the underlying splines, etc
        """
        k = 0.5*(self.kmin+self.kmax)
        return self.Pgal(k, 0.5)
            
    @tools.broadcast_kmu
    def evaluate_fog(self, k_obs, mu_obs, sigma):
        """
        Compute the FOG damping, evaluating at `k` and `mu`. The 
        `alpha_par` dependence here is just absorbed into the `sigma` parameter
        """    
        k = self.k_true(k_obs, mu_obs)
        mu = self.mu_true(mu_obs)
        return self.fog_function(k*mu*sigma)
    
    #---------------------------------------------------------------------------
    # centrals power spectrum
    #--------------------------------------------------------------------------- 
    def Pgal_cAcA(self, k, mu, flatten=False):
        """
        The central type `A` galaxy auto spectrum, which is a 2-halo term only.
        """ 
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_cA
        
        # FOG damping
        with SOCorrection(self) as socorr:
            G = self.evaluate_fog(k, mu, socorr.sigma_c)

        # now return the power spectrum here
        toret = G**2 * self.power(k, mu) + self.N        
        return toret if not flatten else np.ravel(toret, order='F')
    
    def Pgal_cAcB(self, k, mu, flatten=False):
         """
         The centrals galaxy cross spectrum, which is a 2-halo term only.
         """
         # set the linear biases first
         self.b1     = self.b1_cA
         self.b1_bar = self.b1_cB

         # FOG damping
         with SOCorrection(self) as socorr:
             G = self.evaluate_fog(k, mu, socorr.sigma_c)

         # now return the power spectrum here
         toret = G**2 * self.power(k, mu) + self.N
         return toret if not flatten else np.ravel(toret, order='F')
         
    def Pgal_cBcB(self, k, mu, flatten=False):
        """
        The central type `B` galaxy auto spectrum, which is a 2-halo term only.
        """        
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_cB

        # FOG damping
        with SOCorrection(self) as socorr:
            G = self.evaluate_fog(k, mu, socorr.sigma_c)

        # now return the power spectrum here
        toret = G**2*self.power(k, mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    def Pgal_cc(self, k, mu, flatten=False):
        """
        The totals centrals galaxy spectrum, which is a 2-halo term only.
        """
        with ZeroShotNoise(self) as N:
        
            # sum the individual components of Pcc                    
            PcAcA = (1.-self.fcB)**2 * self.Pgal_cAcA(k, mu)
            PcAcB = 2*self.fcB*(1-self.fcB)*self.Pgal_cAcB(k, mu)
            PcBcB = self.fcB**2 * self.Pgal_cBcB(k, mu)
            pk = PcAcA + PcAcB + PcBcB
        
            # add in an optional SO correction
            toret = self.Pgal_cc_so(pk, k, mu) + N
           
        return toret if not flatten else np.ravel(toret, order='F')
        
    def Pgal_cc_so(self, Pcc, k, mu):
        """
        The SO correction term to add to `Pgal_cc`
        """
        # add the correction
        if self.use_so_correction:
            
            G = self.evaluate_fog(k, mu, self.sigma_c)
            G2 = self.evaluate_fog(k, mu, self.sigma_so)
            term1 = (1 - self.f_so)**2 * G**2 * Pcc
            term2 = 2*self.f_so*(1-self.f_so) * G*G2 * Pcc
            term3 = self.f_so**2 * G2**2 * Pcc
            term4 = 2*G*G2*self.f_so*self.fcB*self.NcBs/(self.alpha_perp**2 * self.alpha_par)
            return term1 + term2 + term3 + term4
            
        # just return the input Pcc
        else:
            return Pcc
        
    #---------------------------------------------------------------------------
    # central-satellite cross spectrum
    #---------------------------------------------------------------------------
    def Pgal_cAs(self, k, mu, flatten=False):
        """
        The cross spectrum between centrals with no satellites and satellites. 
        This is a 2-halo term only. 
        """
        # set the linear biases first
        self.b1     = self.b1_cA
        self.b1_bar = self.b1_s
        
        # the FOG damping
        G_c = self.evaluate_fog(k, mu, self.sigma_c)
        G_s = self.evaluate_fog(k, mu, self.sigma_s)
        
        # now return the power spectrum here
        toret = G_c*G_s*self.power(k, mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    def Pgal_cBs(self, k, mu, flatten=False):
        """
        The cross spectrum of centrals with sats in the same halo and satellites.
        This has both a 1-halo and 2-halo term only.
        """
        # the FOG damping
        G_c = self.evaluate_fog(k, mu, self.sigma_c)
        G_s = self.evaluate_fog(k, mu, self.sigma_s)
        
        # return
        vol_scaling = 1./(self.alpha_perp**2 * self.alpha_par)
        toret = G_c*G_s * (self.Pgal_cBs_2h(k, mu) + self.NcBs*vol_scaling) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
    
    def Pgal_cBs_2h(self, k, mu):
        """
        The 2-halo term for the cross spectrum of centrals with sats in 
        the same halo and satellites.
        """
        # set the linear biases first
        self.b1     = self.b1_cB
        self.b1_bar = self.b1_s
            
        return self.power(k, mu)
    
    def Pgal_cs(self, k, mu, flatten=False):
        """
        The total central-satellite cross spectrum.
        """
        with ZeroShotNoise(self) as N:
                    
            PcAs = (1. - self.fcB)*self.Pgal_cAs(k, mu)
            PcBs = self.fcB*self.Pgal_cBs(k, mu)
            toret = PcAs + PcBs + N
        
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    # satellites auto spectrum
    #---------------------------------------------------------------------------
    def Pgal_sAsA(self, k, mu, flatten=False):
        """
        The auto spectrum of satellites with no other sats in same halo. This 
        is a 2-halo term only.
        """            
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_sA
        
        # the FOG damping
        G = self.evaluate_fog(k, mu, self.sigma_sA)
        
        # now return the power spectrum here
        toret = G**2 * self.power(k, mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    def Pgal_sAsB(self, k, mu, flatten=False):
        """
        The cross spectrum of satellites with and without no other sats in 
        same halo. This is a 2-halo term only.
        """        
        # set the linear biases first
        self.b1     = self.b1_sA
        self.b1_bar = self.b1_sB
        
        # the FOG damping
        G_sA = self.evaluate_fog(k, mu, self.sigma_sA)
        G_sB = self.evaluate_fog(k, mu, self.sigma_sB)
        
        # now return the power spectrum here
        toret = G_sA*G_sB*self.power(k, mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')

    def Pgal_sBsB(self, k, mu, flatten=False):
        """
        The auto spectrum of satellits with other sats in the same halo.
        This has both a 1-halo and 2-halo term only.
        """
        # the FOG damping terms
        G = self.evaluate_fog(k, mu, self.sigma_sB)
        
        vol_scaling = 1./(self.alpha_perp**2 * self.alpha_par)
        toret = G**2 * (self.Pgal_sBsB_2h(k, mu) + self.NsBsB*vol_scaling) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
    
    def Pgal_sBsB_2h(self, k, mu):
        """
        The 2-halo term for the auto spectrum of satellits with other sats 
        in the same halo.
        """
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_sB

        return self.power(k, mu)
            
    def Pgal_ss(self, k, mu, flatten=False):
        """
        The total satellites auto spectrum
        """
        with ZeroShotNoise(self) as N:

            PsAsA = (1. - self.fsB)**2 * self.Pgal_sAsA(k, mu)
            PsAsB = 2*self.fsB*(1-self.fsB)*self.Pgal_sAsB(k, mu)
            PsBsB = self.fsB**2 * self.Pgal_sBsB(k, mu) 
            toret = PsAsA + PsAsB + PsBsB + N
            
        return toret if not flatten else np.ravel(toret, order='F')
                    
    #---------------------------------------------------------------------------
    # total galaxy P(k,mu)
    #---------------------------------------------------------------------------
    def Pgal(self, k, mu, flatten=False, update={}):
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
        with ZeroShotNoise(self) as N:
                            
            # sum the component spectra
            Pcc = (1. - self.fs)**2 * self.Pgal_cc(k, mu)
            Pcs = 2.*self.fs*(1 - self.fs) * self.Pgal_cs(k, mu)
            Pss = self.fs**2 * self.Pgal_ss(k, mu)
            toret = Pcc + Pcs + Pss + N

        return toret if not flatten else np.ravel(toret, order='F')
        
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
        
        if not all(ell in [0,2,4] for ell in poles):
            raise ValueError("the only valid multipoles are ell = 0, 2, 4")
            
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
        The total redshift-space galaxy monopole moment
        """
        return self.Pgal(k, mu, **kwargs)

    @tools.quadrupole
    def Pgal_quad(self, k, mu, **kwargs):
        """
        The total redshift-space galaxy quadrupole moment
        """
        return self.Pgal(k, mu, **kwargs)

    @tools.hexadecapole
    def Pgal_hexadec(self, k, mu, **kwargs):
        """
        The total redshift-space galaxy hexadecapole moment
        """
        return self.Pgal(k, mu, **kwargs)
        
    def from_transfer(self, transfer, flatten=False):
        """
        Return the power (either P(k,mu) or multipoles), accounting for 
        discrete binning effects using the input transfer function
    
        Parameter
        ---------
        transfer : PkmuTransfer or PolesTransfer
            the transfer class which accounts for the discrete binning effects
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `Nk * Nmu` or `Nk * Nell`
        """
        # compute P(k,mu) on the grid and update the grid transfer
        grid = transfer.grid
        power = self.Pgal(grid.k[grid.notnull], grid.mu[grid.notnull])
        transfer.power = power
        
        # call the transfer to evaluate P(k,mu) on the grid
        return transfer(flatten=flatten)
    
