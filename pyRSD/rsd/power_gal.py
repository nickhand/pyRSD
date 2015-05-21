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
                        ['use_mean_bias', 'fog_model']
    
    def __init__(self, use_mean_bias=True, fog_model='modified_lorentzian', **kwargs):
        
        # initalize the dark matter power spectrum
        super(GalaxySpectrum, self).__init__(**kwargs)
        
        # set the parameters
        self.use_mean_bias = use_mean_bias
        self.fog_model     = fog_model
        
        # set the defaults
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
        
    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
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
        Constant offset to model, returns 0 by default
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
        return self.Pgal(0.5)
            
    #---------------------------------------------------------------------------
    def evaluate_fog(self, sigma, mu_obs):
        """
        Compute the FOG damping, evaluating at `k_true` and `mu_true`. The 
        `alpha_par` dependence here is just absorbed into the `sigma` parameter
        """
        if np.isscalar(mu_obs):
            return self.fog_model(sigma*self.k_obs*mu_obs)
        else:
            return np.vstack([self.fog_model(sigma*self.k_obs*imu) for imu in mu_obs]).T
    
    #---------------------------------------------------------------------------
    # Central power spectrum
    #--------------------------------------------------------------------------- 
    def Pgal_cAcA(self, mu, flatten=False, hires=False):
        """
        The central type `A` galaxy auto spectrum, which is a 2-halo term only.
        """
        self.hires = hires
        
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_cA
        
        # FOG damping
        G = self.evaluate_fog(self.sigma_c, mu)

        # now return the power spectrum here
        toret = G**2 * self.power(mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    def Pgal_cAcB(self, mu, flatten=False, hires=False):
         """
         The centrals galaxy cross spectrum, which is a 2-halo term only.
         """
         self.hires = hires
         
         # set the linear biases first
         if self.use_mean_bias:
             mean_bias = np.sqrt(self.b1_cA*self.b1_cB)
             self.b1 = self.b1_bar = mean_bias
         else:
             self.b1     = self.b1_cA
             self.b1_bar = self.b1_cB

         # FOG damping
         G = self.evaluate_fog(self.sigma_c, mu)

         # now return the power spectrum here
         toret = G**2 * self.power(mu) + self.N
         return toret if not flatten else np.ravel(toret, order='F')
         
    #---------------------------------------------------------------------------
    def Pgal_cBcB(self, mu, flatten=False, hires=False):
        """
        The central type `B` galaxy auto spectrum, which is a 2-halo term only.
        """
        self.hires = hires
        
        # set the linear biases first
        self.b1 = self.b1_bar = self.b1_cB

        # FOG damping
        G = self.evaluate_fog(self.sigma_c, mu)

        # now return the power spectrum here
        toret = G**2*self.power(mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    def Pgal_cc(self, mu, flatten=False, hires=False):
        """
        The totals centrals galaxy spectrum, which is a 2-halo term only.
        """
        self.hires = hires
        N = self.N
        self.N = 0
        
        PcAcA = (1.-self.fcB)**2 * self.Pgal_cAcA(mu, hires=hires)
        PcAcB = 2*self.fcB*(1-self.fcB)*self.Pgal_cAcB(mu, hires=hires)
        PcBcB = self.fcB**2 * self.Pgal_cBcB(mu, hires=hires)
        toret = PcAcA + PcAcB + PcBcB + N
                
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    # Central-satellite cross spectrum
    #---------------------------------------------------------------------------
    def Pgal_cAs(self, mu, flatten=False, hires=False):
        """
        The cross spectrum between centrals with no satellites and satellites. 
        This is a 2-halo term only. 
        """
        self.hires = hires
        
        # set the linear biases first
        if self.use_mean_bias:
            mean_bias = np.sqrt(self.b1_cA*self.b1_s)
            self.b1 = self.b1_bar = mean_bias
        else:
            self.b1     = self.b1_cA
            self.b1_bar = self.b1_s
        
        # the FOG damping
        G_c = self.evaluate_fog(self.sigma_c, mu)
        G_s = self.evaluate_fog(self.sigma_s, mu)
        
        # now return the power spectrum here
        toret = G_c*G_s*self.power(mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    def Pgal_cBs(self, mu, flatten=False, hires=False):
        """
        The cross spectrum of centrals with sats in the same halo and satellites.
        This has both a 1-halo and 2-halo term only.
        """
        self.hires = hires
        
        # the FOG damping
        G_c = self.evaluate_fog(self.sigma_c, mu)
        G_s = self.evaluate_fog(self.sigma_s, mu)
        
        # return
        toret = G_c*G_s * (self.Pgal_cBs_2h(mu) + self.NcBs) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    def Pgal_cBs_2h(self, mu):
        """
        The 2-halo term for the cross spectrum of centrals with sats in 
        the same halo and satellites.
        """
        # set the linear biases first
        if self.use_mean_bias:
            mean_bias = np.sqrt(self.b1_cB*self.b1_s)
            self.b1 = self.b1_bar = mean_bias
        else:
            self.b1     = self.b1_cB
            self.b1_bar = self.b1_s
            
        return self.power(mu)
    
    #---------------------------------------------------------------------------
    def Pgal_cs(self, mu, flatten=False, hires=False):
        """
        The total central-satellite cross spectrum.
        """
        self.hires = hires
        N = self.N
        self.N = 0
                    
        PcAs = (1. - self.fcB)*self.Pgal_cAs(mu, hires=hires)
        PcBs = self.fcB*self.Pgal_cBs(mu, hires=hires)
        toret = PcAs + PcBs + N
        
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    # Satellites auto spectrum
    #---------------------------------------------------------------------------
    def Pgal_sAsA(self, mu, flatten=False, hires=False):
        """
        The auto spectrum of satellites with no other sats in same halo. This 
        is a 2-halo term only.
        """            
        self.hires = hires
        
        # set the linear biases first
        self.b1     = self.b1_sA
        self.b1_bar = self.b1_sA
        
        # the FOG damping
        G = self.evaluate_fog(self.sigma_sA, mu)
        
        # now return the power spectrum here
        toret = G**2 * self.power(mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    def Pgal_sAsB(self, mu, flatten=False, hires=False):
        """
        The cross spectrum of satellites with and without no other sats in 
        same halo. This is a 2-halo term only.
        """
        self.hires = hires
        
        # set the linear biases first
        if self.use_mean_bias:
            mean_bias = np.sqrt(self.b1_sA*self.b1_sB)
            self.b1 = self.b1_bar = mean_bias
        else:
            self.b1     = self.b1_sA
            self.b1_bar = self.b1_sB
        
        # the FOG damping
        G_sA = self.evaluate_fog(self.sigma_sA, mu)
        G_sB = self.evaluate_fog(self.sigma_sB, mu)
        
        # now return the power spectrum here
        toret = G_sA*G_sB*self.power(mu) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    def Pgal_sBsB(self, mu, flatten=False, hires=False):
        """
        The auto spectrum of satellits with other sats in the same halo.
        This has both a 1-halo and 2-halo term only.
        """
        self.hires = hires
        
        # the FOG damping terms
        G = self.evaluate_fog(self.sigma_sB, mu)
        
        toret = G**2 * (self.Pgal_sBsB_2h(mu) + self.NsBsB) + self.N
        return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    def Pgal_sBsB_2h(self, mu):
        """
        The 2-halo term for the auto spectrum of satellits with other sats 
        in the same halo.
        """
        # set the linear biases first
        self.b1     = self.b1_sB
        self.b1_bar = self.b1_sB

        return self.power(mu)
            
    #---------------------------------------------------------------------------
    def Pgal_ss(self, mu, flatten=False, hires=False):
        """
        The total satellites auto spectrum
        """
        self.hires = hires
        N = self.N
        self.N = 0
        
        PsAsA = (1. - self.fsB)**2 * self.Pgal_sAsA(mu, hires=hires)
        PsAsB = 2*self.fsB*(1-self.fsB)*self.Pgal_sAsB(mu, hires=hires)
        PsBsB = self.fsB**2 * self.Pgal_sBsB(mu, hires=hires) 
        
        # now return
        toret = PsAsA + PsAsB + PsBsB + N
        return toret if not flatten else np.ravel(toret, order='F')
                    
        
    #---------------------------------------------------------------------------
    # Total galaxy P(k,mu)
    #---------------------------------------------------------------------------
    def Pgal(self, mu, flatten=False, hires=False):
        """
        The total redshift-space galaxy power spectrum, combining the individual
        terms.
        
        Parameters
        ----------
        mu : float, array
            The cosine of the angle from the line of sight. Can be either a 
            single or array of values
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `len(self.k_obs) * len(mu)`, or `len(self.k) * len(mu)` (the 
            latter if `hires=True`). Default is `False`
        hires : bool, optional
            If `True`, return the values corresponding to `self.k`, otherwise
            return those corresponding to `self.k`
        """        
        self.hires = hires
        N = self.N
        self.N = 0
        
        # get the model
        Pcc = (1. - self.fs)**2 * self.Pgal_cc(mu, hires=hires)
        Pcs = 2.*self.fs*(1 - self.fs) * self.Pgal_cs(mu, hires=hires)
        Pss = self.fs**2 * self.Pgal_ss(mu, hires=hires)
        
        toret = Pcc + Pcs + Pss + N
        return toret if not flatten else np.ravel(toret, order='F')
        
    #---------------------------------------------------------------------------
    def Pgal_poles(self, poles, flatten=False, hires=False):
        """
        Return the multipole moments specified by `poles`, where `poles` is a
        list of integers, i.e., [0, 2, 4]
        
        Parameter
        ---------
        poles : float, array_like
            The `ell` values of the multipole moments
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `len(self.k_obs) * len(mu)`, or `len(self.k) * len(mu)` (the 
            latter if `hires=True`). Default is `False`
        hires : bool, optional
            If `True`, return the values corresponding to `self.k`, otherwise
            return those corresponding to `self.k`
        """
        scalar = False
        if np.isscalar(poles):
            poles = [poles]
            scalar = True
            
        toret = ()
        mus = np.linspace(0., 1., 41)
        Pkmus = self.Pgal(mus, hires=hires)
        for ell in poles:
            kern = (2*ell+1.)*legendre(ell)(mus)
            val = np.array([simps(kern*d, x=mus) for d in Pkmus])
            toret += (val,)
            
        if scalar:
            return toret[0]
        else:
            return toret if not flatten else np.ravel(toret)
            
    #---------------------------------------------------------------------------
    @tools.monopole
    def Pgal_mono(self, mu, hires=False):
        """
        The total redshift-space galaxy monopole moment
        """
        return self.Pgal(mu, hires=hires)
        
    #---------------------------------------------------------------------------
    @tools.quadrupole
    def Pgal_quad(self, mu, hires=False):
        """
        The total redshift-space galaxy quadrupole moment
        """
        return self.Pgal(mu, hires=hires)
        
    #---------------------------------------------------------------------------
    @tools.hexadecapole
    def Pgal_hexadec(self, mu, hires=False):
        """
        The total redshift-space galaxy hexadecapole moment
        """
        return self.Pgal(mu, hires=hires)
        
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    
