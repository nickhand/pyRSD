from ._cache import parameter, cached_property
from . import power_biased, tools
from .. import numpy as np, sys
from scipy.special import legendre
from scipy.integrate import simps

#-------------------------------------------------------------------------------
# FOG MODELS
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
class GalaxySpectrum(power_biased.BiasedSpectrum):
    """
    The galaxy redshift space power spectrum, a subclass of the `BiasedSpectrum`
    for biased redshift space power spectra
    """
    allowable_kwargs = power_biased.BiasedSpectrum.allowable_kwargs + \
                        ['fog_model', 'use_so_correction']
    
    def __init__(self, fog_model='modified_lorentzian', 
                       use_so_correction=False,
                       **kwargs):
        
        # initalize the dark matter power spectrum
        super(GalaxySpectrum, self).__init__(**kwargs)
        
        # set the parameters
        self.fog_model         = fog_model

        # set the defaults
        self.include_2loop = False
        self.fs            = 0.10
        self.fcB           = 0.08
        self.fsB           = 0.40
        self.fso           = 0.
        self.b1_cA         = 1.85
        self.b1_cB         = 2.8
        self.b1_sA         = 2.6
        self.b1_sB         = 3.6
        self.sigma_c       = 1.
        self.sigma_cA      = 0.
        self.sigma_s       = 5.
        self.sigma_sA      = 4.2
        self.sigma_sB      = 6.
        self.NcBs          = 3e4
        self.NsBsB         = 9e4
        self.N             = 0.
        
        self.use_so_correction = use_so_correction
        self.f_so = 0.
        self.sigma_so = 0.
        
     
    #---------------------------------------------------------------------------
    # PARAMETERS
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
            
        mod = sys.modules[__name__]
        return getattr(mod, 'fog_'+val)
        
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
    # CACHED PROPERTIES
    #---------------------------------------------------------------------------
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
    # UTILITY FUNCTIONS
    #---------------------------------------------------------------------------
    def initialize(self):
        """
        Initialize the underlying splines, etc
        """
        k = 0.5*(self.kmin+self.kmax)
        return self.Pgal(k, 0.5)
            
    def evaluate_fog(self, k, mu, sigma):
        """
        Compute the FOG damping, evaluating at `k` and `mu`. The 
        `alpha_par` dependence here is just absorbed into the `sigma` parameter
        """        
        if np.isscalar(mu) or len(mu) == len(k):
            return self.fog_model(k*mu*sigma)
        else:
            return np.vstack([self.fog_model(k*imu*sigma) for imu in mu]).T
    
    #---------------------------------------------------------------------------
    # Centrals power spectrum
    #--------------------------------------------------------------------------- 
    def Pgal_cAcA(self, k, mu, flatten=False):
        """
        The central type `A` galaxy auto spectrum, which is a 2-halo term only.
        """    
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_cA
        
        # FOG damping
        G = self.evaluate_fog(k, mu, self.sigma_c)

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
         G = self.evaluate_fog(k, mu, self.sigma_c)

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
        G = self.evaluate_fog(k, mu, self.sigma_c)

        # now return the power spectrum here
        toret = G**2*self.power(k, mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    def Pgal_cc(self, k, mu, flatten=False):
        """
        The totals centrals galaxy spectrum, which is a 2-halo term only.
        """
        N = self.N
        self.N = 0
                
        if self.use_so_correction:
            sigma_c = self.sigma_c
            self.sigma_c = 0
            
            PcAcA = (1.-self.fcB)**2 * self.Pgal_cAcA(k, mu)
            PcAcB = 2*self.fcB*(1-self.fcB)*self.Pgal_cAcB(k, mu)
            PcBcB = self.fcB**2 * self.Pgal_cBcB(k, mu)
            pk = PcAcA + PcAcB + PcBcB
            
            G = self.evaluate_fog(k, mu, sigma_c)
            G2 = self.evaluate_fog(k, mu, self.sigma_so)
            term1 = (1 - self.f_so)**2 * G**2 * pk
            term2 = 2*self.f_so*(1-self.f_so)*G*G2*pk
            term3 = self.f_so**2 * G2**2 * pk
            term4 = 2*G*G2*self.f_so*self.fcB/(1-self.fcB)*self.NcBs
            toret = term1 + term2 + term3 + term4 
            self.sigma_c = sigma_c
        else:
            PcAcA = (1.-self.fcB)**2 * self.Pgal_cAcA(k, mu)
            PcAcB = 2*self.fcB*(1-self.fcB)*self.Pgal_cAcB(k, mu)
            PcBcB = self.fcB**2 * self.Pgal_cBcB(k, mu)
            toret = PcAcA + PcAcB + PcBcB + N
        
        self.N = N      
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    # Central-satellite cross spectrum
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
        toret = G_c*G_s * (self.Pgal_cBs_2h(k, mu) + self.NcBs) + self.N
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
        N = self.N
        self.N = 0
                    
        PcAs = (1. - self.fcB)*self.Pgal_cAs(k, mu)
        PcBs = self.fcB*self.Pgal_cBs(k, mu)
        toret = PcAs + PcBs + N
        
        self.N = N
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    # Satellites auto spectrum
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
        
        toret = G**2 * (self.Pgal_sBsB_2h(k, mu) + self.NsBsB) + self.N
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
        N = self.N
        self.N = 0
        
        PsAsA = (1. - self.fsB)**2 * self.Pgal_sAsA(k, mu)
        PsAsB = 2*self.fsB*(1-self.fsB)*self.Pgal_sAsB(k, mu)
        PsBsB = self.fsB**2 * self.Pgal_sBsB(k, mu) 
        
        # now return
        toret = PsAsA + PsAsB + PsBsB + N
        self.N = N
        return toret if not flatten else np.ravel(toret, order='F')
                    
    #---------------------------------------------------------------------------
    # Total galaxy P(k,mu)
    #---------------------------------------------------------------------------
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
        N = self.N
        self.N = 0
        
        # get the model
        Pcc = (1. - self.fs)**2 * self.Pgal_cc(k, mu)
        Pcs = 2.*self.fs*(1 - self.fs) * self.Pgal_cs(k, mu)
        Pss = self.fs**2 * self.Pgal_ss(k, mu)
        toret = Pcc + Pcs + Pss + N
        self.N = N
        return toret if not flatten else np.ravel(toret, order='F')
        
    def Pgal_poles(self, k, poles, flatten=False):
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
            `len(k) * len(mu)`
            
        Returns
        -------
        poles : array_like
            returns array for each ell value in ``poles``
        """
        scalar = np.isscalar(poles)
        if scalar: poles = [poles]
        
        if not all(ell in [0,2,4] for ell in poles):
            raise ValueError("the only valid multipoles are ell = 0, 2, 4")
            
        toret = ()
        mus = np.linspace(0., 1., 41)
        Pkmus = self.Pgal(k, mus)
        for ell in poles:
            kern = (2*ell+1.)*legendre(ell)(mus)
            val = np.array([simps(kern*d, x=mus) for d in Pkmus])
            toret += (val,)
            
        if scalar:
            return toret[0]
        else:
            return toret if not flatten else np.ravel(toret)

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
        
    
