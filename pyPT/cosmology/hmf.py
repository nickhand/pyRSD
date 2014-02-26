"""
 hmf.pyx
 pyPT: module for calculation of halo mass function
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/24/2014
"""
import numpy as np
import copy
from .cosmo import Cosmology
from . import hmf_fits, functionator, linear_growth, cosmo_tools

import scipy.integrate as intg
import scipy.optimize
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class HaloMassFunction(object):
    """
    A class to represent a dark matter halo mass function (HMF) and to hold 
    all relevant quantities related to the HMF. These quantities can be 
    accessed as ``@property`` attributes of the class.

    Parameters
    ----------
    M : array_like, optional
        The masses at which to perform analysis 
        [units :math:`\log_{10}M_\odot h^{-1}`]. The default is 
        ``np.linspace(8, 15, 1001)``.

    z : float, optional
        The redshift at which to compute the HMF. Default is 0.

    mf_fit : str, optional
        A string indicating which fitting function to use for the 
        multiplicity function :math:`f(\sigma)`. Default is ``"Tinker_unnorm"``.

        Available options:

        1. ``'PS'``: Press-Schechter form from 1974
        #. ``'SMT'``: Sheth-Mo-Tormen empirical fit from 2001
        #. ``'Jenkins'``: Jenkins empirical fit from 2001
        #. ``'Warren'``: Warren empirical fit from 2006
        #. ``'Tinker_unnorm'``: unnormalized Tinker empirical from 2008 (main text)
        #. ``'Tinker_norm'``: normalized Tinker empirical from 2008 (Appendix C)

    delta_h : float, optional
        The overdensity used for the halo mass definition, with respect to 
        ``delta_wrt``. Default is 200.

    delta_wrt : str, {``"mean"``, ``"crit"``}, optional
        Defines what the overdensity of a halo is with respect to, either
        the mean density of the universe, or critical density. Default is 
        ``"mean"``.
    
    delta_c : float, optional
        The critical overdensity for collapse. Default is 1.686.

    cut_fit : bool, optional
        Whether to set the HMF to np.NaN where :math:`f(\sigma)` lies
        outside the bounds of validity for each fit. If false, use whole 
        range of `M`

    params : {str, dict, cosmo.Cosmology}, optional
        The cosmological parameters to use, specified by the name of a predefined
        cosmology, a parameter dictionary, or a Cosmology class. Default is None.

    force_flat : bool, optional
        Whether to force the cosmological parameters to have a total density
        equal to the critical density. Passed to the cosmo.Cosmology
        initiator. Default is ``False``.

    default : str, optional
        The cosmology to default to for missing parameters. Default is 
        ``"Planck1_lens_WP_highL"``.
    
    powerlaw_M : float, optional
        The maximum mass to use in the power-law fit to the low mass end of the
        mass function [units :math:`M_\odot h^{-1}`]. Default is 1e12.
        
    Attributes
    ----------
    z : float
    mf_fit : str
    delta_h : float
    delta_wrt : float
    delta_c : float
    cut_fit : bool
    powerlaw_M : float

    cosmo : cosmo.Cosmology, read-only
        The class holding the cosmology parameters

    M : array_like
        The masses at which to perform analysis. [units :math:`M_\odot h^{-1}`]

    R : array_like, read-only
        The radii corresponding to the masses, ``len=len(M)`` 
        [units :math:`Mpc h^{-1}`].

    delta_halo : float, read-only
        The overdensity of a halo with respect to mean density

    sigma : array_like, read-only
        The mass variance at `z`, ``len=len(M)``.

    lnsigma : array_like, read-only
        The natural log of inverse mass variance, ``len=len(M)``

    fsigma : array_like, read-only
        The multiplicity function, :math:`f(\sigma)`, for `mf_fit`. ``len=len(M)`
    
    dndm : array_like, read-only
        The differential number density of haloes, ``len=len(M)`` 
        [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]

    dndlnm : array_like, read-only
        The differential mass function in terms of natural log of `M`, 
        ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]

    dndlog10m : array_like, read-only
        The differential mass function in terms of log10 of `M`, 
        ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        
    ngtm : array_like, read-only
        The cumulative mass function above `M`, ``len=len(M)``
        [units :math:`h^3 Mpc^{-3}`]
        
    exp_cutoff_M : float
        The minimum mass to use in the exponential fit to the high mass end of the
        mass function [units :math:`M_\odot h^{-1}`]. Default is the mass
        where :math: ``4 \sigma(M, z) = \delta_c``.
    """
    def __init__(self, M=None, z=0., mf_fit="Tinker_unnorm", delta_h=200.,
                    delta_wrt='mean', delta_c=1.686, cut_fit=True,
                    params=None, force_flat=False, default="Planck1_lens_WP_highL",
                    powerlaw_M=1e12):
        
        if M is None:
            M = np.linspace(8, 15, 1001)

        # store the cosmology parameters
        if params is None:
            params = default

        # cosmology will be read only so cannot update once initialized
        self._cosmo = Cosmology(params, force_flat=force_flat, default=default)

        # Set all given parameters.
        self.mf_fit       = mf_fit
        self.M            = M
        self.delta_h      = delta_h
        self.delta_wrt    = delta_wrt
        self.delta_c      = delta_c
        self.z            = z
        self.cut_fit      = cut_fit
        self.powerlaw_M   = powerlaw_M
        self.exp_cutoff_M = self._find_cutoff_mass()
        
    #end __init__
    #---------------------------------------------------------------------------
    def sample(self):
        """
        Compute the spline fit to the HMF.
        """
        if not hasattr(self, '__spline'):
            self._compute_dndm_spline()
            
    #---------------------------------------------------------------------------
    def _find_cutoff_mass(self):
        """
        Find the approximate exponential cutoff mass, defined as where 
        :math: ``4*sigma(M, z) = delta_c``.
        """
        def objective(M):
            R = cosmo_tools.mass_to_radius(M*1e14, self.cosmo.mean_dens)
            sigma = linear_growth.mass_variance(R, self.z, normed=True, 
                                                    tf='full', params=self.cosmo)
            return 4.*sigma - self.delta_c
            
        M = scipy.optimize.brentq(objective, 1e-4, 1e5)
        return M*1e14
    #---------------------------------------------------------------------------
    @property
    def cut_fit(self):
        return self.__cut_fit

    @cut_fit.setter
    def cut_fit(self, val):
        del self.fsigma
        self.__cut_fit = val
    #---------------------------------------------------------------------------  
    @property
    def M(self):
        return self.__M

    @M.setter
    def M(self, val):
        del self._sigma_0
        self.__M = 10**val
        self.__R = cosmo_tools.mass_to_radius(self.__M, self.cosmo.mean_dens)
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, val):
        self.__z = val
        del self.sigma
        del self.fsigma
        del self.dndm
    #---------------------------------------------------------------------------
    @property
    def delta_c(self):
        return self.__delta_c

    @delta_c.setter
    def delta_c(self, val):
        self.__delta_c = val
        del self.fsigma
    #---------------------------------------------------------------------------
    @property
    def mf_fit(self):
        return self.__mf_fit
       
    @mf_fit.setter
    def mf_fit(self, val):

        if val not in hmf_fits.Multiplicity.available:
           raise ValueError("mf_fit is not in the list of available fitting functions: ", val)
        del self.fsigma
        self.__mf_fit = val
    #---------------------------------------------------------------------------
    @property
    def delta_h(self):
        return self.__delta_h
  
    @delta_h.setter
    def delta_h(self, val):
        
        self.__delta_h = val
        del self.delta_halo
        
    #---------------------------------------------------------------------------
    @property
    def powerlaw_M(self):
        """
        The maximum mass to use in the power-law fit to the low mass end of the
        mass function [units :math:`M_\odot h^{-1}`].
        """
        return self.__powerlaw_M
  
    @powerlaw_M.setter
    def powerlaw_M(self, val):
        
        if hasattr(self, 'exp_cutoff_M'):
            if val > self.exp_cutoff_M:
                raise ValueError("Low-mass power law mass cannot be greater than exponential cutoff mass")
        self.__powerlaw_M = val
        try:
            del self.__spline
        except:
            pass
    
    #---------------------------------------------------------------------------
    @property
    def exp_cutoff_M(self):
        """
        The minimum mass to use in the exponential fit to the high mass end of the
        mass function [units :math:`M_\odot h^{-1}`].
        """
        return self.__exp_cutoff_M
  
    @exp_cutoff_M.setter
    def exp_cutoff_M(self, val):
        
        if hasattr(self, 'powerlaw_M'):
            if val < self.powerlaw_M:
                raise ValueError("Exponential cutoff mass cannot be smaller than low-mass power law mass")
        self.__exp_cutoff_M = val
        try:
            del self.__spline
        except:
            pass
    
    #---------------------------------------------------------------------------
    @property
    def delta_wrt(self):
        return self.__delta_wrt

    @delta_wrt.setter
    def delta_wrt(self, val):
        if val not in ['mean', 'crit']:
            raise ValueError("delta_wrt must be either 'mean' or 'crit' (", val, ")")

        self.__delta_wrt = val
        del self.delta_halo
    #---------------------------------------------------------------------------
    @property
    def R(self):
        return self.__R
    #---------------------------------------------------------------------------
    @property
    def delta_halo(self):
       """ 
       Overdensity of a halo w.r.t mean density
       """
       try:
           return self.__delta_halo
       except:
           if self.delta_wrt == 'mean':
               self.__delta_halo = self.delta_h

           elif self.delta_wrt == 'crit':
               self.__delta_halo = self.delta_h / cosmo_tools.omega_m_z(self.z, params=self.cosmo)
           return self.__delta_halo

    @delta_halo.deleter
    def delta_halo(self):
        try:
           del self.__delta_halo
           del self.fsigma
        except:
           pass
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        """
        Cosmological parameters stored in a cosmo.Cosmology class
        """
        return self._cosmo
    #---------------------------------------------------------------------------
    @property
    def _sigma_0(self):
        """
        The normalised mass variance at z=0 :math:`\sigma`

        Notes
        -----

        .. math:: \sigma^2(R) = \frac{1}{2\pi^2}\int_0^\infty{k^2P(k)W^2(kR)dk}

        """

        try:
           return self.__sigma_0
        except:
           self.__sigma_0 = linear_growth.mass_variance(self.R, 0., normed=True, 
                                                        tf='full', params=self.cosmo)
           return self.__sigma_0

    @_sigma_0.deleter
    def _sigma_0(self):
        try:
           del self.__sigma_0
           del self._dlnsdlnm
           del self.sigma
        except:
           pass

    @property
    def _dlnsdlnm(self):
        """
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln M}\right|`, ``len=len(M)``

        Notes
        -----

        .. math:: frac{d\ln\sigma}{d\ln M} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk

        """
        try:
           return self.__dlnsdlnm
        except:
           self.__dlnsdlnm = linear_growth.dlnsdlnm(self.R, sigma0=self._sigma_0, params=self.cosmo)
           return self.__dlnsdlnm

    @_dlnsdlnm.deleter
    def _dlnsdlnm(self):
        try:
           del self.__dlnsdlnm
           del self.dndm
        except:
           pass

    @property
    def sigma(self):
        """
        The mass variance at `z`, ``len=len(M)``
        """
        try:
           return self.__sigma
        except:
           self.__sigma = self._sigma_0*linear_growth.growth_function(self.z, 
                                                normed=True, params=self.cosmo)
           return self.__sigma

    @sigma.deleter
    def sigma(self):
        try:
           del self.__sigma
           del self.fsigma
           del self.lnsigma
        except:
           pass

    @property
    def lnsigma(self):
        """
        Natural log of inverse mass variance, ``len=len(M)``
        """
        try:
           return self.__lnsigma
        except:
           self.__lnsigma = np.log(1 / self.sigma)
           return self.__lnsigma

    @lnsigma.deleter
    def lnsigma(self):
        try:
           del self.__lnsigma
           del self.fsigma
        except:
           pass

    @property
    def fsigma(self):
        """
        The multiplicity function, :math:`f(\sigma)`, for `mf_fit`. ``len=len(M)``
        """
        try:
           return self.__fsigma
        except:
           fit_class = hmf_fits.Multiplicity(self, cut=self.cut_fit)
           self.__fsigma = fit_class.fsigma()
           return self.__fsigma

    @fsigma.deleter
    def fsigma(self):
        try:
           del self.__fsigma
           del self.dndm
        except:
           pass

    @property
    def dndm(self):
        """
        The number density of haloes, ``len=len(M)`` [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]
        """
        try:
           return self.__dndm
        except:
           self.__dndm = self.fsigma*self.cosmo.mean_dens*np.abs(self._dlnsdlnm) / self.M**2
           return self.__dndm

    @dndm.deleter
    def dndm(self):
        try:
           del self.__dndm
           del self.dndlnm
           del self.dndlog10m
           del self.__spline
        except:
           pass            
            
    @property
    def dndlnm(self):
        """
        The differential mass function in terms of natural log of `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
           return self.__dndlnm
        except:
           self.__dndlnm = self.M * self.dndm
           return self.__dndlnm

    @dndlnm.deleter
    def dndlnm(self):
        try:
           del self.__dndlnm
        except:
           pass

    @property
    def dndlog10m(self):
        """
        The differential mass function in terms of log of `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
           return self.__dndlog10m
        except:
           self.__dndlog10m = self.M * self.dndm * np.log(10)
           return self.__dndlog10m

    @dndlog10m.deleter
    def dndlog10m(self):
        try:
           del self.__dndlog10m
        except:
           pass

    def _ngtm(self):
        """
        Calculate n(>m).
        """
        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        self.sample()

        newM = np.concatenate( (M[:-1], 10**(np.linspace(np.log10(M[0]), 18, 1000))) )
        mass_function = self.dndlnm_spline(newM)

        # Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
        ngtm = intg.cumtrapz(mass_function[::-1], dx=np.log(newM[1]) - np.log(newM[0]))[::-1]

        # We need to set ngtm back in the original length vector with nans where they were originally
        if len(M) < len(self.M):
            ngtm_temp = np.zeros_like(self.dndlnm)
            ngtm_temp[:] = np.nan
            ngtm_temp[np.logical_not(np.isnan(self.dndlnm))] = ngtm[:len(M)]
            ngtm = ngtm_temp
        else:
            ngtm = ngtm[:len(M)]

        return ngtm
    #end _ngtm
    
    #---------------------------------------------------------------------------
    @property
    def ngtm(self):
        """
        The cumulative mass function above `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
           return self.__ngtm
        except:
           self.__ngtm = self._ngtm()
           return self.__ngtm

    @ngtm.deleter
    def ngtm(self):
        try:
           del self.__ngtm
        except:
           pass

    @property
    def mgtm(self):
        """
        Mass in haloes `>M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]
        """
        try:
           return self.__mgtm
        except:
           M = self.M[np.logical_not(np.isnan(self.dndlnm))]
           mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

           # Calculate the mass function (and its integral) from the highest M up to 10**18
           if M[-1] < 10 ** 18:
               m_upper, mf = self._upper_ngtm(M, np.log(mass_function), M[-1] < self.M[-1])
               int_upper = intg.simps(np.exp(mf + m_upper) , dx=m_upper[2] - m_upper[1], even='first')
           else:
               int_upper = 0

           # Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
           self.__mgtm = np.concatenate((intg.cumtrapz(mass_function[::-1] * M[::-1], dx=np.log(M[1]) - np.log(M[0]))[::-1], np.zeros(1))) + int_upper

           # We need to set ngtm back in the original length vector with nans where they were originally
           if len(self.__mgtm) < len(self.M):
               mgtm_temp = np.zeros_like(self.dndlnm)
               mgtm_temp[:] = np.nan
               mgtm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__mgtm
               self.__mgtm = mgtm_temp
           return self.__mgtm
    @mgtm.deleter
    def mgtm(self):
        try:
           del self.__mgtm
        except:
           pass

    @property
    def nltm(self):
        """
        Inverse cumulative mass function, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
           return self.__nltm
        except:
           # set M and mass_function within computed range
           M = self.M[np.logical_not(np.isnan(self.dndlnm))]
           mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

           # Calculate the mass function (and its integral) from 10**3 up to lowest M
           if M[0] > 10 ** 3:
               m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

               int_lower = intg.simps(np.exp(mf), dx=m_lower[2] - m_lower[1], even='first')
           else:
               int_lower = 0

           # Calculate the cumulative integral of mass_function (Adding on the lower integral)
           self.__nltm = np.concatenate((np.zeros(1), intg.cumtrapz(mass_function, dx=np.log(M[1]) - np.log(M[0])))) + int_lower

           # We need to set ngtm back in the original length vector with nans where they were originally
           if len(self.__nltm) < len(self.M):
               nltm_temp = np.zeros_like(self.dndlnm)
               nltm_temp[:] = np.nan
               nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__nltm
               self.__nltm = nltm_temp

           return self.__nltm
    @nltm.deleter
    def nltm(self):
        try:
           del self.__nltm
        except:
           pass

    @property
    def mltm(self):
        """
        Total mass in haloes `<M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]
        """
        try:
           return self.__mltm
        except:
           # Set M within calculated range
           M = self.M[np.logical_not(np.isnan(self.dndlnm))]
           mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

           # Calculate the mass function (and its integral) from 10**3 up to lowest M
           if M[0] > 10 ** 3:
               m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

               int_lower = intg.simps(np.exp(mf + m_lower), dx=m_lower[2] - m_lower[1], even='first')
           else:
               int_lower = 0

           # Calculate the cumulative integral of mass_function (Adding on the lower integral)
           self.__mltm = np.concatenate((np.zeros(1), intg.cumtrapz(mass_function * M, dx=np.log(M[1]) - np.log(M[0])))) + int_lower

           # We need to set ngtm back in the original length vector with nans where they were originally
           if len(self.__mltm) < len(self.M):
               nltm_temp = np.zeros_like(self.dndlnm)
               nltm_temp[:] = np.nan
               nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__mltm
               self.__mltm = nltm_temp

           return self.__mltm
          
    #----------------------------------------------------------------------------
    def _fit_exponential_cutoff(self, M, mf):
        """
        Fit the high-mass end of the HMF with an exponential cutoff in mass 
        and return a functionator.powerLawExtrapolator object. 
        """
        # find if we have high mass values computed already
        inds = np.where(M > self.exp_cutoff_M)[0]
        
        # make sure we have at least 10 points to fit
        if len(inds) >= 10:
            m_upper  = M[inds]
            mf_upper = mf[inds]
        else:
            new_mf         = copy.deepcopy(self)
            new_mf.cut_fit = False
            new_mf.M       = np.linspace(np.log10(self.exp_cutoff_M), np.log10(1e17), 500)
            mf_upper       = new_mf.dndm            
            m_upper        = new_mf.M
            
        # do the exponential fit and initialize the extrapolator function
        exp_decay = lambda x, A, gamma: A * np.exp(x * gamma)
        p0 = [1e-25, -1./1e16]
        (A, gamma), parm_cov = scipy.optimize.curve_fit(exp_decay, m_upper, mf_upper, p0=p0, maxfev=10000)
        exp_extrap = functionator.exponentialExtrapolator(gamma=gamma, A=A, min_x=self.exp_cutoff_M)
        return exp_extrap
    #end _fit_exponential_cutoff
    
    #---------------------------------------------------------------------------
    def _fit_lowmass_powerlaw(self, M, mf):
        """
        Fit the low-mass end of the HMF with a power law and return a
        functionator.powerLawExtrapolator object. 
        """
        # find if we have high mass values computed already
        inds = np.where(M < self.powerlaw_M)[0]
        
        # make sure we have at least 10 points to fit
        if len(inds) >= 10:
            m_lower = M[inds]
            mf_lower = mf[inds]
        else:
            new_mf = copy.deepcopy(self)
            new_mf.cut_fit = False
            new_mf.M = np.linspace(np.log10(1e8), np.log10(self.powerlaw_M), 500)
            mf_lower = new_mf.dndm            
            m_lower = new_mf.M
            
        # do the power law fit and initialize the extrapolator function
        p_fit = np.polyfit(np.log(m_lower), np.log(mf_lower), 1)
        p_gamma, p_amp = p_fit[0], np.exp(p_fit[1])
        power_extrap = functionator.powerLawExtrapolator(gamma=p_gamma, A=p_amp, max_x=self.powerlaw_M)

        return power_extrap
    #end _fit_lowmass_powerlaw
    #---------------------------------------------------------------------------
    def _compute_dndm_spline(self):
        """
        Compute a spline fit to the HMF, where we use a power-law extrapolation
        at low mass and an exponential cutoff at high mass
        """
        #  keep track of where M/mass function are not defined
        M = self.M[np.logical_not(np.isnan(self.dndm))]
        mf = self.dndm[np.logical_not(np.isnan(self.dndm))]

        # fit the exponential cutoff
        hi_extrap = self._fit_exponential_cutoff(M, mf)
       
        # now fit the low mass power law
        lo_extrap = self._fit_lowmass_powerlaw(M, mf)
       
        # now do the middle spline interpolation 
        new_mf = copy.deepcopy(self)
        new_mf.cut_fit = False
        new_mf.M = np.linspace(np.log10(self.powerlaw_M), np.log10(self.exp_cutoff_M), 500)
        mf_middle = new_mf.dndm
        m_middle = new_mf.M
        
        spline_interp = functionator.splineLogInterpolator(m_middle, mf_middle,
                                                            min_x=self.powerlaw_M,
                                                            max_x=self.exp_cutoff_M)
        
        # get the whole combined function
        self.__spline = functionator.functionator(ops=[lo_extrap, spline_interp, hi_extrap])
    #end _compute_dndm_spline
    
    #---------------------------------------------------------------------------
    def dndm_spline(self, mass):
        """
        Return the spline fit for dn/dm at the input mass M, 
        [units :math:`M_\odot h^{-1}`]
        """
        try:
            return self.__spline(mass)
        except:
            self._compute_dndm_spline()
            return self.__spline(mass)
    #end dndm_spline
    
    #---------------------------------------------------------------------------
    def dndlog10m_spline(self, mass):
        """
        Return the spline fit for dn/dlog10m at the input mass M, 
        [units :math:`M_\odot h^{-1}`]
        """
        try:
            return mass*np.log(10)*self.__spline(mass)
        except:
            self._compute_dndm_spline()
            return mass*np.log(10)*self.__spline(mass)
    #end dndlog10m_spline
    
    #---------------------------------------------------------------------------
    def dndlnm_spline(self, mass):
        """
        Return the spline fit for dn/dlnm at the input mass M, 
        [units :math:`M_\odot h^{-1}`]
        """
        try:
            return mass*self.__spline(mass)
        except:
            self._compute_dndm_spline()
            return mass*self.__spline(mass)
    #end dndlnm_spline
    
    #---------------------------------------------------------------------------
