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
    # POWER SPECTRA TERMS
    #---------------------------------------------------------------------------
    def Pgal_cc(self, mu):
        """
        The central galaxy auto spectrum, assuming no FOG here. This is a 2-halo
        term only.
        """
        # set the linear biases first
        self.b1     = self.b1_c
        self.b1_bar = self.b1_c
        
        # FOG damping
        G = self.evaluate_fog(self.sigma_c, mu)

        # now return the power spectrum here
        return G**2*self.power(mu)
        
    #---------------------------------------------------------------------------
    def Pgal_cAs(self, mu):
        """
        The cross spectrum between centrals with no satellites and satellites. 
        This is a 2-halo term only. 
        """
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
        return G_c*G_s*self.power(mu)
        
    #---------------------------------------------------------------------------
    def Pgal_sAsA(self, mu):
        """
        The auto spectrum of satellites with no other sats in same halo. This 
        is a 2-halo term only.
        """            
        # set the linear biases first
        self.b1     = self.b1_sA
        self.b1_bar = self.b1_sA
        
        # the FOG damping
        G = self.evaluate_fog(self.sigma_sA, mu)
        
        # now return the power spectrum here
        return G**2 * self.power(mu)
        
    #---------------------------------------------------------------------------
    def Pgal_sAsB(self, mu):
        """
        The cross spectrum of satellites with and without no other sats in 
        same halo. This is a 2-halo term only.
        """
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
        return G_sA*G_sB*self.power(mu)
    
    #---------------------------------------------------------------------------
    def Pgal_cBs(self, mu):
        """
        The cross spectrum of centrals with sats in the same halo and satellites.
        This has both a 1-halo and 2-halo term only.
        """
        # the FOG damping
        G_c = self.evaluate_fog(self.sigma_c, mu)
        G_s = self.evaluate_fog(self.sigma_s, mu)
        
        return G_c*G_s * (self.Pgal_cBs_2h(mu) + self.NcBs)
    
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
    def Pgal_sBsB(self, mu):
        """
        The auto spectrum of satellits with other sats in the same halo.
        This has both a 1-halo and 2-halo term only.
        """
        # the FOG damping terms
        G = self.evaluate_fog(self.sigma_sB, mu)
        
        return G**2 * (self.Pgal_sBsB_2h(mu) + self.NsBsB)
    
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
    def Pgal_ss(self, mu):
        """
        The 2-halo part of the total satellite auto spectrum.
        """
            
        return (1. - self.fsB)**2 * self.Pgal_sAsA(mu) + \
                    2*self.fsB*(1-self.fsB)*self.Pgal_sAsB(mu) + \
                    self.fsB**2 * self.Pgal_sBsB(mu) 
                    
    #---------------------------------------------------------------------------
    def Pgal_cs(self, mu):
        """
        The total central-satellite cross spectrum.
        """            
        return (1. - self.fcB)*self.Pgal_cAs(mu) + self.fcB*self.Pgal_cBs(mu) 
        
    #---------------------------------------------------------------------------
    def Pgal(self, mu, flatten=False, hires=False, dmu=None):
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
        dmu : array_like, optional
            If not `None`, specifies the bin width for each mu specified, and
            the `mu` values given are interpreted as the mean of each bin. The
            returned P(k,mu) values are integrated over the bin
        """
        # setup if we are integrating across a mu bin
        if dmu is not None:
            mu, dmu = np.array(mu, ndmin=1), np.array(dmu, ndmin=1)
            N = 21
            lower = mu - dmu/2
            upper = mu + dmu/2
            Nmu = len(mu)
            mu = np.concatenate([np.linspace(lower[i], upper[i], N) for i in range(Nmu)])
        
        self.hires = hires
        
        fss = self.fs**2
        fcs = 2.*self.fs*(1 - self.fs)
        fcc = (1. - self.fs)**2
        toret = fcc * self.Pgal_cc(mu) + fcs * self.Pgal_cs(mu) + fss * self.Pgal_ss(mu) + self.N
        
        # integrate over each mu bin
        if dmu is not None:
            pkmus = []
            for i in range(Nmu):
                val = np.array([simps(d[i*N : (i+1)*N], x=mu[i*N : (i+1)*N]) for d in toret])
                pkmus.append(val / dmu[i])
            return np.vstack(pkmus).T if not flatten else np.concatenate(pkmus)
        # just return values at mu specified
        else:
            return toret if not flatten else np.ravel(toret, order='F')
    
    #---------------------------------------------------------------------------
    def Pgal_diff(self, mu_pairs, flatten=False, hires=False):
        """
        The difference in P(k, mu)
        
        Parameters
        ----------
        mu_pairs : tuple, list of tuples
            A tuple or list of tuples in the format (mu_hi - mu_lo), where
            P(k, mu=mu_hi) - P(k, mu=mu_lo) will be returned
        flatten : bool, optional    
            If `True`, flatten the return array, which will have a length of 
            `len(self.k_obs) * len(mu)`, or `len(self.k) * len(mu)` (the 
            latter if `hires=True`). Default is `False`
        hires : bool, optional
            If `True`, return the values corresponding to `self.k`, otherwise
            return those corresponding to `self.k`
        """
        mu_hi, mu_lo = map(list, zip(*list(mu_pairs)))
        return self.Pgal(mu_hi, flatten=flatten, hires=hires) - self.Pgal(mu_lo, flatten=flatten, hires=hires)
    
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
    
