#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
 power_dm.pyx
 pyPT: class implementing the redshift space dark matter power spectrum using
       the PT expansion outlined in Vlah et al. 2012.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
from pyPT.power cimport integralsIJ
from pyPT.cosmology cimport growth, cosmo_tools
from ..cosmology import cosmo, velocity, hmf
from pyPT.power import power_1loop
 
import scipy.integrate as intgr
import scipy.interpolate as interp
import numpy as np
cimport numpy as np
import re

class DMSpectrum(object):
    
    KMAX = 1e3
    KMIN = 1e-7
    CONVERGENCE_FACTOR = 100.
    
    def __init__(self, k=np.logspace(-2, np.log10(0.5), 100),
                       z=0., 
                       kmin=KMIN, 
                       kmax=KMAX, 
                       num_threads=1, 
                       cosmo_params="Planck1_lens_WP_highL", 
                       cosmo_kwargs={'default':"Planck1_lens_WP_highL", 'force_flat':False},
                       mass_function_kwargs={'mf_fit' : 'Tinker'}, 
                       bias_model='Tinker',
                       include_2loop=False,
                       max_mu=6):
        """
        Parameters
        ----------
        k : array_like, optional
            The wavenumbers to compute the power spectrum at [units: h/Mpc]. 
            Must be between 1e-5 h/Mpc and 100 h/Mpc to be valid. 
            
        z : float
            The redshift to compute the power spectrum at.
            
        kmin : float, optional
            The wavenumber in h/Mpc defining the minimum value to compute 
            integrals to. Default is kmin = 1e-3 h/Mpc. Must be greater
            than 1e-5 h/Mpc to be valid.
            
        kmax : float, optional
            The wavenumber in h/Mpc defining the maximum value to compute 
            integrals to. Default is 10 h/Mpc. Must be less than 100 h/Mpc 
            to be valid.
            
        num_threads : int, optional
            The number of threads to use in parallel computation. Default = 1.
            
        cosmo_params : {str, dict, cosmo.Cosmology}
            The cosmological parameters to use. Default is Planck DR1 + lensing
            + WP + high L 2013 parameters.
        
        cosmo_kwargs : dict, optional
            Keyword arguments to pass to the cosmo.Cosmology class. Possible 
            keywords include ``default`` and ``force_flat``.
            
        mass_function_kwargs : dict, optional
            The keyword arguments to pass to the hmf.HaloMassFunction object
            for use in computing small-scale velocity dispersions in the halo
            model.
            
        bias_model : {'Tinker', 'SMT', 'PS'}
            The name of the halo bias model to use for any halo model 
            calculations.
            
        include_2loop : bool, optional
            If ``True``, include 2-loop contributions in the model terms. Default
            is ``False``.
        
        max_mu : {0, 2, 4, 6, 8}, optional
            Only compute angular terms up to mu**(``max_mu``). Default is 4.
        """
        # initialize the cosmology parameters
        self.max_mu          = max_mu
        self.cosmo_kwargs    = cosmo_kwargs
        self.cosmo           = cosmo_params
        self.kmin, self.kmax = kmin, kmax
        self.num_threads     = num_threads
        self.include_2loop   = include_2loop
        self.k               = np.array(k, copy=False, ndmin=1)
        
        # wavenumbers for interpolation purposes
        self.klin_interp = np.logspace(np.log(1e-8), np.log(1e5), 1e4, base=np.e)
                    
        # save useful quantitues for later
        self.z                    = z 
        self.mass_function_kwargs = mass_function_kwargs
        self.bias_model           = bias_model
        
    #end __init__
    #---------------------------------------------------------------------------
    def _delete_all(self):
        """
        Delete all integral and power spectra attributes.
        """
        pattern = re.compile("_[IJP]([0-9]*|lin|_mu[0-9]+)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    # SET ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, val):
        if np.amin(val) < DMSpectrum.CONVERGENCE_FACTOR*self.kmin:
            raise ValueError("With kmin = %.2e h/Mpc, power spectrum cannot be computed " %self.kmin + \
                            "for wavenumbers less than %.2e h/Mpc" 
                            %(self.kmin*DMSpectrum.CONVERGENCE_FACTOR))
                            
        if np.amax(val) > self.kmax/DMSpectrum.CONVERGENCE_FACTOR:
            raise ValueError("With kmax = %.2e h/Mpc, power spectrum cannot be computed " %self.kmax + \
                            "for wavenumbers greater than %.2e h/Mpc" 
                            %(self.kmax/DMSpectrum.CONVERGENCE_FACTOR))
        self._k = val
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, val):
        self._z = val
        if hasattr(self, 'oneloop_interp'):
            self.oneloop_interp.z = val
        if hasattr(self, 'mass_function_kwargs'):
            self.mass_function_kwargs['z'] = val
        
        del self.hmf
        pattern = re.compile("_((f|D|conformalH)|P[0-9_]+[A-Za-z0-9_]+)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        return self._cosmo
    
    @cosmo.setter
    def cosmo(self, val):
        if not isinstance(val, cosmo.Cosmology):
            self._cosmo = cosmo.Cosmology(val, **self.cosmo_kwargs)
        else:
            self._cosmo = val
            
        # basically delete everything
        del self.hmf
        del self.oneloop_interp
        pattern = re.compile("_(f|D|conformalH|Plin_interp|sigma_lin)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
        self._delete_all()
    #--------------------------------------------------------------------------
    @property
    def cosmo_kwargs(self):
        return self._cosmo_kwargs
    
    @cosmo_kwargs.setter
    def cosmo_kwargs(self, val):
        self._cosmo_kwargs = val
        
        # basically delete everything
        del self.hmf
        del self.oneloop_interp
        pattern = re.compile("_(f|D|conformalH|Plin_interp|sigma_lin)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def bias_model(self):
        return self._bias_model
    
    @bias_model.setter
    def bias_model(self, val):
        self._bias_model = val
        
        del self.sigma_bv2
        del self.sigma_bv4
    #---------------------------------------------------------------------------
    @property
    def max_mu(self):
        return self._max_mu
    
    @max_mu.setter
    def max_mu(self, val):
        if val not in [0, 2, 4, 6, 8]:
            raise ValueError("Maximum mu value must be one of [0, 2, 4, 6, 8]")
        self._max_mu = val
        
        # delete all power spectra terms
        pattern = re.compile("_[P]([0-9]{2}[_a-z]*|_mu[0-9]{2})")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def mass_function_kwargs(self):
        return self._mass_function_kwargs
    
    @mass_function_kwargs.setter
    def mass_function_kwargs(self, val):
        self._mass_function_kwargs = val
        self._mass_function_kwargs['z'] = self.z
        
        del self.hmf
    #---------------------------------------------------------------------------
    @property
    def kmin(self):
        return self._kmin

    @kmin.setter
    def kmin(self, val):
        if val < DMSpectrum.KMIN:
            raise ValueError("Minimum wavenumber must be greater than %.2e h/Mpc" %DMSpectrum.KMIN)       
        self._kmin = val
        
        if hasattr(self, 'oneloop_interp'):
            self.oneloop_interp.kmin = val
        
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def kmax(self):
        return self._kmax

    @kmax.setter
    def kmax(self, val):
        if val > DMSpectrum.KMAX:
            raise ValueError("Maximum wavenumber must be less than %.2e h/Mpc" %DMSpectrum.KMAX)
        self._kmax = val
        
        if hasattr(self, 'oneloop_interp'):
            self.oneloop_interp.kmax = val
        
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def include_2loop(self):
        return self._include_2loop
    
    @include_2loop.setter
    def include_2loop(self, val):
        self._include_2loop = val
        
        pattern = re.compile("_P(11|02|12|22|03|13|04|_mu[2468])[_a-z]*")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def sigma_v2(self):
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by the velocity squared. [units: km/s]
        
        .. math:: (sigma_{v2})^2 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} v_{\parallel}^2
        
        Notes
        -----
        The halo mass function used here is determined by the 
        ``halo_mass_function_kwargs`` attribute.
        """
        try:
            return self._sigma_v2
        except:
            self._sigma_v2 = velocity.sigma_v2(self.hmf)
            return self._sigma_v2
    
    @sigma_v2.setter
    def sigma_v2(self, val):
        self._sigma_v2 = val
        
        pattern = re.compile("_P(01|03|_mu[46])")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    @sigma_v2.deleter
    def sigma_v2(self):
        try:
            del self._sigma_v2
        except:
            pass
    #---------------------------------------------------------------------------
    @property 
    def sigma_bv2(self):
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by the bias times velocity squared. [units: km/s]
        
        .. math:: (sigma_{bv2})^2 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} b(M) v_{\parallel}^2
        
        Notes
        -----
        The halo mass function used here is determined by the 
        ``halo_mass_function_kwargs`` attribute. The bias model used is 
        determined by the ``bias_model`` attribute. 
        """
        try:
            return self._sigma_bv2
        except:
            self._sigma_bv2 = velocity.sigma_bv2(self.hmf, self.bias_model)
            return self._sigma_bv2
    
    @sigma_bv2.setter
    def sigma_bv2(self, val):
        self._sigma_bv2 = val
        
        pattern = re.compile("_P(02|12|13|22|_mu[2468])")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    @sigma_bv2.deleter
    def sigma_bv2(self):
        try:
            del self._sigma_bv2
        except:
            pass
    #---------------------------------------------------------------------------
    @property 
    def sigma_bv4(self):
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by bias times the velocity squared. [units: km/s]
        
        .. math:: (sigma_{bv4})^4 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} b(M) v_{\parallel}^4
        
        Notes
        -----
        The halo mass function used here is determined by the 
        ``halo_mass_function_kwargs`` attribute. The bias model used is 
        determined by the ``bias_model`` attribute.
        """
        try:
            return self._sigma_bv4
        except:
            self._sigma_bv4 = velocity.sigma_bv4(self.hmf, self.bias_model)
            return self._sigma_bv4
            
    @sigma_bv4.setter
    def sigma_bv4(self, val):
        self._sigma_bv4 = val
        
        pattern = re.compile("_P(04|_mu[46])")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    @sigma_bv4.deleter
    def sigma_bv4(self):
        try:
            del self._sigma_bv4
        except:
            pass
    #---------------------------------------------------------------------------
    # READ-ONLY ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def D(self):
        try:
            return self._D
        except:
            self._D = growth.growth_function(self.z, normed=True, params=self.cosmo)
            return self._D
    #---------------------------------------------------------------------------
    @property
    def f(self):
        try:
            return self._f
        except:
            self._f = growth.growth_rate(self.z, params=self.cosmo)
            return self._f
    #---------------------------------------------------------------------------
    @property
    def conformalH(self):
        try:
            return self._conformalH
        except:
            self._conformalH = cosmo_tools.H(self.z,  params=self.cosmo)/(1.+self.z)
            return self._conformalH
    #---------------------------------------------------------------------------
    @property
    def hmf(self):
        try:
            return self._hmf
        except:
            self._hmf = hmf.HaloMassFunction(**self.mass_function_kwargs)
            return self._hmf
    
    @hmf.deleter
    def hmf(self):
        try:
            del self._hmf
        except:
            pass
        del self.sigma_v2
        del self.sigma_bv2
        del self.sigma_bv4
    #---------------------------------------------------------------------------
    @property
    def oneloop_interp(self):
        try:
            return self._oneloop_interp
        except:
            self._oneloop_interp = DM1LoopInterp(self.z, self.cosmo, self.kmin, 
                                            self.kmax, k=np.logspace(-4, 1, 500), 
                                            num_threads=self.num_threads)
            return self._oneloop_interp
    
    @oneloop_interp.deleter
    def oneloop_interp(self):
        try:
            del self._oneloop_interp
        except:
            pass
    #---------------------------------------------------------------------------
    @property
    def Plin(self):
        try:
            return self._Plin
        except:
            self._Plin = growth.Pk_lin(self.k, 0., tf='EH', params=self.cosmo) 
            return self._Plin
    #---------------------------------------------------------------------------
    @property
    def Plin_interp(self):
        try:
            return self._Plin_interp
        except:
            self._Plin_interp = growth.Pk_lin(self.klin_interp, 0., tf='EH', params=self.cosmo)
            return self._Plin_interp
    #---------------------------------------------------------------------------        
    @property
    def I(self):
        try:
            return self._I
        except:
            self._I = integralsIJ.I_nm(0, 0, self.klin_interp, self.Plin_interp, k2=None, P2=None)
            return self._I
    #---------------------------------------------------------------------------
    @property
    def J(self):
        try:
            return self._J
        except:
            self._J = integralsIJ.J_nm(0, 0, self.klin_interp, self.Plin_interp)
            return self._J
    #---------------------------------------------------------------------------
    @property
    def sigma_lin(self):
        """
        The dark matter velocity dispersion, as evaluated in linear theory 
        [units: Mpc/h]
        """
        try:
            return self._sigma_lin
        except:
            self._sigma_lin = velocity.sigmav_lin(cosmo_params=self.cosmo)
            return self._sigma_lin 
    #---------------------------------------------------------------------------
    # INTEGRAL ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def I00(self):
        try:
            return self._I00
        except:
            self.I.n, self.I.m = 0, 0
            self._I00 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I00
    #---------------------------------------------------------------------------
    @property
    def I01(self):
        try:
            return self._I01
        except:
            self.I.n, self.I.m = 0, 1
            self._I01 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I01
    #---------------------------------------------------------------------------
    @property
    def I10(self):
        try:
            return self._I10
        except:
            self.I.n, self.I.m = 1, 0
            self._I10 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I10
    #---------------------------------------------------------------------------
    @property
    def I11(self):
        try:
            return self._I11
        except:
            self.I.n, self.I.m = 1, 1
            self._I11 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I11
    #---------------------------------------------------------------------------
    @property
    def I02(self):
        try:
            return self._I02
        except:
            self.I.n, self.I.m = 0, 2
            self._I02 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I02
    #---------------------------------------------------------------------------
    @property
    def I20(self):
        try:
            return self._I20
        except:
            self.I.n, self.I.m = 2, 0
            self._I20 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I20
    #---------------------------------------------------------------------------
    @property
    def I12(self):
        try:
            return self._I12
        except:
            self.I.n, self.I.m = 1, 2
            self._I12 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I12
    #---------------------------------------------------------------------------
    @property
    def I21(self):
        try:
            return self._I21
        except:
            self.I.n, self.I.m = 2, 1
            self._I21 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I21
    #---------------------------------------------------------------------------
    @property
    def I22(self):
        try:
            return self._I22
        except:
            self.I.n, self.I.m = 2, 2
            self._I22 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I22
    #---------------------------------------------------------------------------
    @property
    def I03(self):
        try:
            return self._I03
        except:
            self.I.n, self.I.m = 0, 3
            self._I03 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I03
    #---------------------------------------------------------------------------
    @property
    def I30(self):
        try:
            return self._I30
        except:
            self.I.n, self.I.m = 3, 0
            self._I30 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I30
    #---------------------------------------------------------------------------
    @property
    def I31(self):
        try:
            return self._I31
        except:
            self.I.n, self.I.m = 3, 1
            self._I31 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I31
    #---------------------------------------------------------------------------
    @property
    def I13(self):
        try:
            return self._I13
        except:
            self.I.n, self.I.m = 1, 3
            self._I13 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I13
    #---------------------------------------------------------------------------
    @property
    def I23(self):
        try:
            return self._I23
        except:
            self.I.n, self.I.m = 2, 3
            self._I23 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I23
    #---------------------------------------------------------------------------
    @property
    def I32(self):
        try:
            return self._I32
        except:
            self.I.n, self.I.m = 3, 2
            self._I32 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I32
    #---------------------------------------------------------------------------
    @property
    def I33(self):
        try:
            return self._I33
        except:
            self.I.n, self.I.m = 3, 3
            self._I33 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I33
    #---------------------------------------------------------------------------
    @property
    def J00(self):
        try:
            return self._J00
        except:
            self.J.n, self.J.m = 0, 0
            self._J00 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self._J00
    #---------------------------------------------------------------------------
    @property
    def J01(self):
        try:
            return self._J01
        except:
            self.J.n, self.J.m = 0, 1
            self._J01 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self._J01
    #---------------------------------------------------------------------------
    @property
    def J10(self):
        try:
            return self._J10
        except:
            self.J.n, self.J.m = 1, 0
            self._J10 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self._J10
    #---------------------------------------------------------------------------
    @property
    def J11(self):
        try:
            return self._J11
        except:
            self.J.n, self.J.m = 1, 1
            self._J11 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self._J11
    #---------------------------------------------------------------------------
    @property
    def J20(self):
        try:
            return self._J20
        except:
            self.J.n, self.J.m = 2, 0
            self._J20 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self._J20
    #---------------------------------------------------------------------------
    @property
    def J02(self):
        try:
            return self._J02
        except:
            self.J.n, self.J.m = 0, 2
            self._J02 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self._J02
    #---------------------------------------------------------------------------
    # TWO LOOP INTEGRALS (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def I23_2loop(self):
        try:
            return self._I23_2loop
        except:
            I = integralsIJ.I_nm(2, 3, self.oneloop_interp.k, self.oneloop_interp.Pvv, k2=None, P2=None)
            self._I23_2loop = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I23_2loop
    #---------------------------------------------------------------------------
    @property
    def I32_2loop(self):
        try:
            return self._I32_2loop
        except:
            I = integralsIJ.I_nm(3, 2, self.oneloop_interp.k, self.oneloop_interp.Pvv, k2=None, P2=None)
            self._I32_2loop = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I32_2loop
    #---------------------------------------------------------------------------
    @property
    def I33_2loop(self):
        try:
            return self._I33_2loop
        except:
            I = integralsIJ.I_nm(3, 3, self.oneloop_interp.k, self.oneloop_interp.Pvv, k2=None, P2=None)
            self._I33_2loop = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I33_2loop
    #---------------------------------------------------------------------------
    @property
    def I04(self):
        try:
            return self._I04
        except:
            I = integralsIJ.I_nm(0, 4, self.oneloop_interp.k, self.oneloop_interp.Pvv, k2=self.oneloop_interp.k, P2=self.oneloop_interp.Pdd)
            self._I04 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I04
    #---------------------------------------------------------------------------
    @property
    def I40(self):
        try:
            return self._I40
        except:
            I = integralsIJ.I_nm(4, 0, self.oneloop_interp.k, self.oneloop_interp.Pdv, k2=None, P2=None)
            self._I40 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I40
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def P_mu0(self):
        """
        The full power spectrum term with no angular dependence. Contributions
        from P00
        """
        try:
            return self._P_mu0
        except:
            self._P_mu0 = self.P00.total.mu0
            return self._P_mu0
    #---------------------------------------------------------------------------
    @property
    def P_mu2(self):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        try:
            return self._P_mu2
        except:
            self._P_mu2 = self.P01.total.mu2 + self.P11.total.mu2 + self.P02.total.mu2
            return self._P_mu2
    #---------------------------------------------------------------------------
    @property
    def P_mu4(self):
        """
        The full power spectrum term with mu^4 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13 (2-loop), and P04 (2-loop)
        """
        try:
            return self._P_mu4
        except:
            self._P_mu4 = self.P11.total.mu4 + self.P02.total.mu4 + \
                            self.P12.total.mu4 + self.P22.total.mu4 + \
                            self.P03.total.mu4 
            if self.include_2loop:
                self._P_mu4 += (self.P13.total.mu4 + self.P04.total.mu4)
            return self._P_mu4
    #---------------------------------------------------------------------------
    @property
    def P_mu6(self):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P12, P22, P13, and P04 (2-loop)
        """
        try:
            return self._P_mu6
        except:
            self._P_mu6 = self.P12.total.mu6 + self.P22.total.mu6 + \
                            self.P13.total.mu6
            if self.include_2loop:
                self._P_mu6 += self.P04.total.mu6
            return self._P_mu6
    #---------------------------------------------------------------------------
    @property
    def P_mu8(self):
        """
        The full power spectrum term with mu^8 angular dependence. Contributions
        from P22. 
        """
        try:
            return self._P_mu8
        except:
            if self.include_2loop:
                self._P_mu8 = self.P22.total.mu8
            else:
                self._P_mu8 = np.zeros(len(self.k))
            return self._P_mu8
    #---------------------------------------------------------------------------
    @property
    def P00(self):
        """
        The isotropic, zero-order term in the power expansion, corresponding
        to the density field auto-correlation. No angular dependence.
        """
        try:
            return self._P00
        except:
            self._P00 = PowerTerm()
            P11 = self.Plin
            P22 = 2*self.I00
            P13 = 3*self.k**2 * P11*self.J00
            self._P00.total.mu0 = self.D**2*P11 + self.D**4*(P22 + 2*P13)
            return self._P00
    #---------------------------------------------------------------------------
    @property
    def P01(self):
        """
        The correlation of density and momentum density, which contributes
        mu^2 terms to the power expansion.
        """
        try:
            return self._P01
        except:
            self._P01 = PowerTerm()
            A = 2.*self.f*self.D**2
            self._P01.total.mu2 = A*(self.Plin + 4.*self.D**2*(self.I00 + 3*self.k**2*self.J00*self.Plin))
            return self._P01
    #---------------------------------------------------------------------------
    @property
    def P11(self):
        """
        The auto-correlation of momentum density, which has a scalar portion 
        which contributes mu^4 terms and a vector term which contributes
        mu^2*(1-mu^2) terms to the power expansion. This is the last term to
        contain a linear contribution.
        """
        try:
            return self._P11
        except:
            self._P11 = PowerTerm()
            
            # do mu^2 terms?
            if self.max_mu >= 2:
                
                # do the vector part, contribution mu^2 and mu^4 terms
                if not self.include_2loop:
                    Pvec_mu2 = (self.f*self.D**2)**2 * self.I31
                else:
                    Pvec_mu2 = (self.I04 + self.I40)/self.conformalH**2
                
                # save the mu^2 vector term
                self._P11.vector.mu2 = self._P11.total.mu2 = Pvec_mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                        
                    # compute the scalar mu^4 contribution
                    part2 = self.D**2*(2*self.I11 + 4*self.I22 + self.I13 + 6*self.k**2*(self.J11 + 2*self.J10)*self.Plin)
                    P_scalar = (self.f*self.D)**2*(self.Plin + part2)
                    
                    # save the scalar/vector mu^4 terms
                    self._P11.scalar.mu4 = P_scalar
                    self._P11.total.mu4 = self._P11.scalar.mu4
            
            return self._P11
    #---------------------------------------------------------------------------
    @property
    def P02(self):
        """
        The correlation of density and energy density, which contributes
        mu^2 and mu^4 terms to the power expansion. There are no 
        linear contributions here.
        """
        try:
            return self._P02
        except:
            self._P02 = PowerTerm()
            
            # do mu^2 terms?
            if self.max_mu >= 2:
            
                # the nmu^2 no velocity terms
                self._P02.no_velocity.mu2 = (self.f*self.D**2)**2 * (self.I02 + 2*self.k**2*self.J02*self.Plin)
                
                # the mu^2 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_lin
                sigma_02  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D)
                vel_terms = sigma_lin**2 + sigma_02**2

                if self.include_2loop:
                    self._P02.with_velocity.mu2 = -(self.f*self.D*self.k)**2 * vel_terms * self.P00.total.mu0
                else:
                    self._P02.with_velocity.mu2 = -(self.f*self.D**2*self.k)**2 * vel_terms * self.Plin
            
                # save the total mu^2 term
                self._P02.total.mu2 = self._P02.with_velocity.mu2 + self._P02.no_velocity.mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                    
                    # only a no velocity term for mu^4
                    self._P02.no_velocity.mu4 = (self.f*self.D**2)**2 * (self.I20 + 2*self.k**2*self.J20*self.Plin)
                    self._P02.total.mu4 = self._P02.no_velocity.mu4
                    
            return self._P02
    #---------------------------------------------------------------------------
    @property
    def P12(self):
        """
        The correlation of momentum density and energy density, which contributes
        mu^4 and mu^6 terms to the power expansion. There are no linear 
        contributions here. Two-loop contribution uses the mu^2 contribution
        from the P01 term.
        """
        try:
            return self._P12
        except:
            self._P12 = PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # do the mu^4 terms that don't depend on velocity
                self._P12.no_velocity.mu4 = self.f**3 * self.D**4 * (self.I12 - self.I03 + 2*self.k**2*self.J02*self.Plin)
            
                # now do mu^4 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_lin  
                sigma_12  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                vel_terms = sigma_lin**2 + sigma_12**2
            
                if self.include_2loop:
                    self._P12.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * self.P01.total.mu2
                else:
                    self._P12.with_velocity.mu4 = -self.f**3 * self.D**4 * self.k**2 * vel_terms * self.Plin
            
                # total mu^4 is velocity + no velocity terms
                self._P12.total.mu4 = self._P12.with_velocity.mu4 + self._P12.no_velocity.mu4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    self._P12.no_velocity.mu6 = self.f**3 * self.D**4 * (self.I21 - self.I30 + 2*self.k**2*self.J20*self.Plin)
                    self._P12.total.mu6 = self._P12.no_velocity.mu6
            
            return self._P12
    #---------------------------------------------------------------------------
    @property
    def P22(self):
        """
        The autocorelation of energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear 
        contributions here. 
        """
        try:
            return self._P22
        except:
            self._P22 = PowerTerm()
            
            # velocity terms come in at 2-loop here
            if self.include_2loop:
                
                # velocities in units of Mpc/h
                sigma_lin = self.sigma_lin
                sigma_22  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                vel_terms = sigma_lin**2 + sigma_22**2
                
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # 1-loop or 2-loop terms that don't depend on velocity
                if not self.include_2loop:
                    A = 1./16 * (self.f*self.D)**4
                    self._P22.no_velocity.mu4 = A * self.I23
                else:
                    A = (1./16) / self.conformalH**4
                    self._P22.no_velocity.mu4 = A * self.I23_2loop

                # now add in the extra 2 loop terms, if specified
                if self.include_2loop:
                               
                    # 2 loop terms have P02 and P00 weighted by velocity
                    extra1_mu4 = 0.5*(self.f*self.D*self.k)**2 * vel_terms * self.P02.no_velocity.mu2
                    extra2_mu4 = 0.25*(self.f*self.D*self.k)**4 * vel_terms**2 * self.P00.total.mu0
                    
                    # last term is convolution of P22_bar and P00
                    extra1_no_vel_mu4 = self.oneloop_interp.convolve('P22bar_mu4', 'Pdd', self.k)
                
                    # store the extra two loop terms
                    self._P22.with_velocity.mu4 = extra1_mu4 + extra2_mu4 
                    self._P22.no_velocity.mu4 += extra1_no_vel_mu4 
                
                # save the totals
                if self.include_2loop:
                    self._P22.total.mu4 = self._P22.with_velocity.mu4 + self._P22.no_velocity.mu4
                else:
                    self._P22.total.mu4 = self._P22.no_velocity.mu4
                    
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # 1-loop or 2-loop terms that don't depend on velocity
                    if not self.include_2loop:
                        A = 1./16 * (self.f*self.D)**4
                        self._P22.no_velocity.mu6 = A * 2*self.I32
                    else:
                        A = (1./16) / self.conformalH**4
                        self._P22.no_velocity.mu6 = A * 2*self.I32_2loop
                        
                    # now add in the extra 2 loop terms, if specified
                    if self.include_2loop:

                        # 2 loop terms have P02 and P00 weighted by velocity
                        self._P22.with_velocity.mu6 = 0.5*(self.f*self.D*self.k)**2 * vel_terms * self.P02.no_velocity.mu4
                        
                        # this term is convolution of P22_bar and P00 (no vel dependence)
                        self._P22.no_velocity.mu6 += self.oneloop_interp.convolve('P22bar_mu6', 'Pdd', self.k)
                    
                    
                    # save the totals
                    if self.include_2loop:
                        self._P22.total.mu6 = self._P22.with_velocity.mu6 + self._P22.no_velocity.mu6
                    else:
                        self._P22.total.mu6 = self._P22.no_velocity.mu6

                    # do mu^8 terms?
                    if self.max_mu >= 8:
                        
                        # 1-loop or 2-loop terms that don't depend on velocity
                        if not self.include_2loop:
                            A = 1./16 * (self.f*self.D)**4
                            self._P22.no_velocity.mu8 = A * self.I33
                        else:
                            A = (1./16) / self.conformalH**4
                            self._P22.no_velocity.mu8 = A * self.I33_2loop
                            
                        # now add in the extra 2 loop terms, if specified
                        if self.include_2loop:
                            
                            # this term is convolution of P22_bar and P00
                            extra1_no_vel_mu8 = self.oneloop_interp.convolve('P22bar_mu8', 'Pdd', self.k)
                            
                            # store the extra two loop term
                            self._P22.no_velocity.mu8 += extra1_no_vel_mu8
                        
                        # save the total
                        self._P22.total.mu8 = self._P22.no_velocity.mu8
                        
            return self._P22
    #---------------------------------------------------------------------------
    @property
    def P03(self):
        """
        The cross-corelation of density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^4 terms.
        """
        try:
            return self._P03
        except:
            self._P03 = PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # only terms depending on velocity here (velocities in Mpc/h)
                sigma_lin = self.sigma_lin 
                sigma_03  = self.sigma_v2 * self.cosmo.h / (self.f*self.conformalH*self.D)
                vel_terms = sigma_lin**2 + sigma_03**2

                # either 1 or 2 loop quantities
                if self.include_2loop:
                    self._P03.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * self.P01.total.mu2
                else:
                    self._P03.with_velocity.mu4 = -(self.f**3*self.D**4) * self.k**2 * vel_terms * self.Plin
            
                self._P03.total.mu4 = self._P03.with_velocity.mu4

            return self._P03
    #---------------------------------------------------------------------------
    @property
    def P13(self):
        """
        The cross-correlation of momentum density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^6 terms at 1-loop order and 
        mu^4 terms at 2-loop order.
        """
        try:
            return self._P13
        except:
            self._P13 = PowerTerm()
            
            # compute velocity weighting in Mpc/h
            sigma_lin = self.sigma_lin 
            sigma_13_v  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
            vel_terms_vector = sigma_lin**2 + sigma_13_v**2
            
            if self.include_2loop:
                sigma_13_s  = self.sigma_v2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                vel_terms_scalar = sigma_lin**2 + sigma_13_s**2
                
            # do mu^4 terms?
            if self.max_mu >= 4:
            
                # mu^4 is only 2-loop
                if self.include_2loop:

                    A = -(self.f*self.D*self.k)**2
                    P13_vel_mu4 = A*vel_terms_vector*self.P11.vector.mu2
                    self._P13.total.mu4 = self._P13.with_velocity.mu4 = P13_vel_mu4

                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # mu^6 velocity terms at 1 or 2 loop
                    if self.include_2loop:
                        A = -(self.f*self.D*self.k)**2
                        self._P13.with_velocity.mu6 = A*vel_terms_scalar*(self.P11.scalar.mu4 + self.P11.vector.mu4)
                    else:
                        self._P13.with_velocity.mu6 = -(self.f**4*self.D**4) * self.k**2 * vel_terms_vector * self.Plin
                        
                    self._P13.total.mu6 = self._P13.with_velocity.mu6
            
            return self._P13
    #---------------------------------------------------------------------------
    @property
    def P04(self):
        """
        The cross-correlation of density with the rank four tensor field
        ((1+delta)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        try:
            return self._P04
        except:
            self._P04 = PowerTerm()
            
            # only 2-loop terms here...
            if self.include_2loop:
                
                # compute the relevant small-scale + linear velocities in Mpc/h
                sigma_lin = self.sigma_lin 
                sigma_04  = self.sigma_bv4 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                vel_terms = sigma_lin**2 + sigma_04**2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    # do P04 mu^4 terms depending on velocity
                    P04_vel_mu4_1 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * self.P02.no_velocity.mu2
                    P04_vel_mu4_2 = 0.25*(self.f*self.D*self.k)**4 * vel_terms**2 * self.P00.total.mu0
                    self.P04.with_velocity.mu4 = P04_vel_mu4_1 + P04_vel_mu4_2
                    
                    # do P04 mu^4 terms without vel dependence
                    A = 0.25*(self.k/self.conformalH)**4 / (2*np.pi**2)
                    integral = intgr.simps(self.oneloop_interp.k**3*self.oneloop_interp.P22_bar.total.mu4, x=np.log(self.oneloop_interp.k))
                    self.P04.no_velocity.mu4 = A/12. * self.P00.total.mu0 * integral
                
                    # save the total
                    self.P04.total.mu4 = self.P04.with_velocity.mu4 + self.P04.no_velocity.mu4
                
                    # do mu^6 terms?
                    if self.max_mu >= 6:
                        
                        # only terms depending on velocity
                        self.P04.with_velocity.mu6 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * self.P02.no_velocity.mu4
                        self.P04.total.mu6 = self.P04.with_velocity.mu6
                        
            return self._P04
    #---------------------------------------------------------------------------
    def monopole(self):
        """
        The monopole moment of the power spectrum. Include mu terms up to 
        mu^max_mu.
        """
        if self.max_mu == 0:
            return self.P_mu0
        elif self.max_mu == 2:
            return self.P_mu0 + (1./3)*self.P_mu2
        elif self.max_mu == 4:
            return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4
        elif self.max_mu == 6:
            return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4 + (1./7)*self.P_mu6
        elif self.max_mu == 8:
            return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4 + (1./7)*self.P_mu6 + (1./9)*self.P_mu8

    #end monopole
    #---------------------------------------------------------------------------
    def quadrupole(self):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to 
        mu^max_mu.
        """
        if self.max_mu == 2:
            return (2./3)*self.P_mu2
        elif self.max_mu == 4:
            return (2./3)*self.P_mu2 + (4./7)*self.P_mu4
        elif self.max_mu == 6:
            return (2./3)*self.P_mu2 + (4./7)*self.P_mu4 + (10./21)*self.P_mu6
        elif self.max_mu == 8:
            return (2./3)*self.P_mu2 + (4./7)*self.P_mu4 + (10./21)*self.P_mu6 + (40./99)*self.P_mu8
    #end quadrupole
    #---------------------------------------------------------------------------
    def hexadecapole(self):
        """
        The hexadecapole moment of the power spectrum. Include mu terms up to 
        mu^max_mu.
        """
        if self.max_mu == 4:
            return (8./35)*self.P_mu4
        elif self.max_mu == 6:
            return (8./35)*self.P_mu4 + (24./77)*self.P_mu6
        elif self.max_mu == 8:
            return (8./35)*self.P_mu4 + (24./77)*self.P_mu6 + (48./143)*self.P_mu8
    #end hexadecapole
    #---------------------------------------------------------------------------
    def load(self, k_data, power_data, power_term, mu_term, errs=None):
        """
        Load data into a given power attribute, as specified by power_term
        and mu_term.
        """
        power_name = "_%s" %power_term
        if hasattr(self, power_name):
            power_attr = getattr(self, power_name)
        else:
            self.__dict__[power_name] = PowerTerm()
            power_attr = self.__dict__[power_name]
            
        power_attr = getattr(power_attr, 'total')
        if errs is not None:
            w = 1./np.array(errs)**2
        else:
            w = None
        s = interp.InterpolatedUnivariateSpline(k_data, power_data, w=w)
        power_interp = s(self.k)
        setattr(power_attr, mu_term, power_interp)
    #end load
    #---------------------------------------------------------------------------
    def unload(self, power_term):
        """
        Delete the given power attribute, as specified by power_term.
        """
        power_name = "_%s" %power_term
        if hasattr(self, power_name):
            del self.__dict__[power_name]
    #end load
    #---------------------------------------------------------------------------
#endclass DMSpectrum 

#-------------------------------------------------------------------------------
class DM1LoopInterp(object):
    """
    A class to hold one loop terms for interpolation purposes.
    """
    def __init__(self, z, cosmo, kmin, kmax, k=np.logspace(-4, 1, 500), num_threads=1):
        
        self.k = k
        self.z = z
        self.num_threads = num_threads
        self.cosmo = cosmo
        self.kmin, self.kmax = kmin, kmax
    #end __init__
    
    #---------------------------------------------------------------------------
    @property
    def kmin(self):
        return self._kmin
    
    @kmin.setter
    def kmin(self, val):
        self._kmin = val
        
        pattern = re.compile("_(I23|I32|I33|P22bar.)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def kmax(self):
        return self._kmax
    
    @kmax.setter
    def kmax(self, val):
        self._kmax = val
        
        pattern = re.compile("_(I23|I32|I33|P22bar.)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

        pattern = re.compile("_(D|f|conformalH|I23|I32|I33|P22bar.)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #----------------------------------------------------------------------------
    @property
    def D(self):
        try:
            return self._D
        except:
            self._D = growth.growth_function(self.z, normed=True, params=self.cosmo)
            return self._D
    #---------------------------------------------------------------------------
    @property
    def f(self):
        try:
            return self._f
        except:
            self._f = growth.growth_rate(self.z, params=self.cosmo)
            return self._f
    #---------------------------------------------------------------------------
    @property
    def conformalH(self):
        try:
            return self._conformalH
        except:
            self._conformalH = cosmo_tools.H(self.z,  params=self.cosmo)/(1.+self.z)
            return self._conformalH    
    #---------------------------------------------------------------------------
    @property
    def Plin(self):
        try:
            return self._Plin
        except:
            self._Plin = growth.Pk_lin(self.k, 0., tf='EH', params=self.cosmo) 
            return self._Plin
    #---------------------------------------------------------------------------
    # 1-loop power spectra
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        try:
            A = (self.f*self.conformalH)**2
            return A*(self.D**2 * self.Plin + self.D**4 * (self._Pvv22 + self._Pvv13))
        except:
            self._Pvv22, self._Pvv13 = power_1loop.Pvv_1loop(self.k, self.z, 
                                                            num_threads=self.num_threads, 
                                                            cosmo=self.cosmo,
                                                            return_components=True)            
            A = (self.f*self.conformalH)**2
            return A*(self.D**2 * self.Plin + self.D**4 * (self._Pvv22 + self._Pvv13))       
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        try:
            A = -self.f*self.conformalH
            return A*(self.D**2 * self.Plin + self.D**4 * (self._Pdv22 + self._Pdv13))
        except:
            self._Pdv22, self._Pdv13 = power_1loop.Pdv_1loop(self.k, self.z, 
                                                            num_threads=self.num_threads, 
                                                            cosmo=self.cosmo,
                                                            return_components=True)            
            A = -self.f*self.conformalH
            return A*(self.D**2 * self.Plin + self.D**4 * (self._Pdv22 + self._Pdv13))     
    #---------------------------------------------------------------------------
    @property
    def Pdd(self):
        try:
            return self.D**2 * self.Plin + self.D**4 * (self._Pdd22 + self._Pdd13)
        except:
            self._Pdd22, self._Pdd13 = power_1loop.Pdd_1loop(self.k, self.z, 
                                                            num_threads=self.num_threads, 
                                                            cosmo=self.cosmo,
                                                            return_components=True)            
            return self.D**2 * self.Plin + self.D**4 * (self._Pdd22 + self._Pdd13)
    #---------------------------------------------------------------------------
    # INTEGRAL ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def I23(self):
        try:
            return self._I23
        except:
            I = integralsIJ.I_nm(2, 3, self.k, self.Pvv, k2=None, P2=None)
            self._I23 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I23
    #---------------------------------------------------------------------------
    @property
    def I32(self):
        try:
            return self._I32
        except:
            I = integralsIJ.I_nm(3, 2, self.k, self.Pvv, k2=None, P2=None)
            self._I32 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I32
    #---------------------------------------------------------------------------
    @property
    def I33(self):
        try:
            return self._I33
        except:
            I = integralsIJ.I_nm(3, 3, self.k, self.Pvv, k2=None, P2=None)
            self._I33 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self._I33
    #---------------------------------------------------------------------------
    @property
    def P22bar_mu4(self):
        """
        The mu^4 contribution to the P22 with no velocity terms, at 2 loop order.
        """
        try:
            return self._P22bar_mu4
        except:     
            A = 0.25/self.k**4
            self._P22bar_mu4 = A * self.I23

            return self._P22bar_mu4
    #---------------------------------------------------------------------------
    @property
    def P22bar_mu6(self):
        """
        The mu^6 contribution to the P22 with no velocity terms, at 2 loop order.
        """
        try:
            return self._P22bar_mu6
        except:    
            A = 0.25/self.k**4
            self._P22bar_mu6 = A * 2.*self.I32

            return self._P22bar_mu6
    #---------------------------------------------------------------------------
    @property
    def P22bar_mu8(self):
        """
        The mu^8 contribution to the P22 with no velocity terms, at 2 loop order.
        """
        try:
            return self._P22bar_mu8
        except:
            A = 0.25/self.k**4
            self._P22bar_mu8 = A * self.I33

            return self._P22bar_mu8
    #---------------------------------------------------------------------------
    def convolve(self, power1, power2, k_eval):
        """
        Convolve with the specified power term with P22_bar.
        """
        # make sure the input attributes exist
        if not hasattr(self, power1):
            raise AttributeError("Power attribute '%s' of %s not found." %(power1, self.__class__))
        if not hasattr(self, power1):
            raise AttributeError("Power attribute '%s' of %s not found." %(power1, self.__class__))
        
        power1 = getattr(self, power1)
        power2 = getattr(self, power2)
        
        # make sure the power and k have the same lengths
        if len(power1) != len(self.k):
            raise ValueError("Wavenumber and '%s' attribute of %s are of different lengths" %(power1, self.__class__))
        if len(power2) != len(self.k):
            raise ValueError("Wavenumber and '%s' attribute of %s are of different lengths" %(power2, self.__class__))
        
        A = 0.25*(k_eval/self.conformalH)**4
        I = integralsIJ.I_nm(-1, -1, self.k, power1, k2=self.k, P2=power2)
        return A*I.evaluate(k_eval, self.kmin, self.kmax, self.num_threads)
    #---------------------------------------------------------------------------
#endclass DM1LoopInterp

#-------------------------------------------------------------------------------   
class PowerTerm(object):
    def __init__(self):
        
        # total angular dependences
        self.total = Angular()
        
        # initialize scalar/vector sub terms
        self.scalar = Angular()
        self.vector = Angular()
        
        # initialize with/without velocity dispersion sub terms
        self.no_velocity   = Angular()
        self.with_velocity = Angular()
        
class Angular(object):
    
    def __init__(self):
        self.mu0 = None
        self.mu2 = None
        self.mu4 = None
        self.mu6 = None
        self.mu8 = None
        
        
