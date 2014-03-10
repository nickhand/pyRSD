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
from pyPT.cosmology cimport cosmo_tools, growth 
from ..cosmology import cosmo, velocity, hmf
from pyPT.power import power_1loop

import scipy.integrate as intgr
import numpy as np
cimport numpy as np
import re

class Spectrum(object):
    
    KMAX = 1e3
    KMIN = 1e-7
    CONVERGENCE_FACTOR = 100.
    
    def __init__(self, k=np.linspace(0.01, 0.5, 100),
                       z=0., 
                       kmin=KMIN, 
                       kmax=KMAX, 
                       num_threads=1, 
                       cosmo_params="Planck1_lens_WP_highL", 
                       cosmo_kwargs={'default':"Planck1_lens_WP_highL", 'force_flat':False},
                       mass_function_kwargs={'mf_fit' : 'Tinker'}, 
                       bias_model='Tinker',
                       include_2loop=False):
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
        """
        # initialize the cosmology parameters
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
        pattern = re.compile("_Spectrum__[IJP][0-9_]+[A-Za-z0-9_]+")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    # SET ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, val):
        if np.amin(val) < Spectrum.CONVERGENCE_FACTOR*self.kmin:
            raise ValueError("With kmin = %.2e h/Mpc, power spectrum cannot be computed " %self.kmin + \
                            "for wavenumbers less than %.2e h/Mpc" 
                            %(self.kmin*Spectrum.CONVERGENCE_FACTOR))
                            
        if np.amax(val) > self.kmax/Spectrum.CONVERGENCE_FACTOR:
            raise ValueError("With kmax = %.2e h/Mpc, power spectrum cannot be computed " %self.kmax + \
                            "for wavenumbers greater than %.2e h/Mpc" 
                            %(self.kmax/Spectrum.CONVERGENCE_FACTOR))
        self.__k = val
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self.__z
    
    @z.setter
    def z(self, val):
        self.__z = val
        
        del self.hmf
        del self.two_loop
        pattern = re.compile("_Spectrum__((f|D|conformalH)|[P][0-9_]+[A-Za-z0-9_]+)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        return self.__cosmo
    
    @cosmo.setter
    def cosmo(self, val):
        if not isinstance(val, cosmo.Cosmology):
            self.__cosmo = cosmo.Cosmology(val, **self.cosmo_kwargs)
        else:
            self.__cosmo = val
            
        # basically delete everything
        del self.hmf
        del self.two_loop
        pattern = re.compile("_Spectrum__(f|D|conformalH|Plin_interp|I|J|sigma_lin)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
        self._delete_all()
    #--------------------------------------------------------------------------
    @property
    def cosmo_kwargs(self):
        return self.__cosmo_kwargs
    
    @cosmo_kwargs.setter
    def cosmo_kwargs(self, val):
        self.__cosmo_kwargs = val
        
        # basically delete everything
        del self.hmf
        del self.two_loop
        pattern = re.compile("_Spectrum__(f|D|conformalH|Plin_interp|I|J|sigma_lin)")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def bias_model(self):
        return self.__bias_model
    
    @bias_model.setter
    def bias_model(self, val):
        self.__bias_model = val
        
        del self.sigma_bv2
        del self.sigma_bv4
    #---------------------------------------------------------------------------
    @property
    def mass_function_kwargs(self):
        return self.__mass_function_kwargs
    
    @mass_function_kwargs.setter
    def mass_function_kwargs(self, val):
        self.__mass_function_kwargs = val
        self.__mass_function_kwargs['z'] = self.z
        
        del self.hmf
    #---------------------------------------------------------------------------
    @property
    def kmin(self):
        return self.__kmin

    @kmin.setter
    def kmin(self, val):
        if val < Spectrum.KMIN:
            raise ValueError("Minimum wavenumber must be greater than %.2e h/Mpc" %Spectrum.KMIN)       
        self.__kmin = val
        
        self._delete_all()
        del self.two_loop
    #---------------------------------------------------------------------------
    @property
    def kmax(self):
        return self.__kmax

    @kmax.setter
    def kmax(self, val):
        if val > Spectrum.KMAX:
            raise ValueError("Maximum wavenumber must be less than %.2e h/Mpc" %Spectrum.KMAX)
        self.__kmax = val
        
        self._delete_all()
        del self.two_loop
    #---------------------------------------------------------------------------
    @property
    def include_2loop(self):
        return self.__include_2loop
    
    @include_2loop.setter
    def include_2loop(self, val):
        self.__include_2loop = val
        
        pattern = re.compile("_Spectrum__P(11|02|12|22|03|13|04|_mu[2468])")
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
            return self.__sigma_v2
        except:
            self.__sigma_v2 = velocity.sigma_v2(self.hmf)
            return self.__sigma_v2
    
    @sigma_v2.setter
    def sigma_v2(self, val):
        self.__sigma_v2 = val
        
        pattern = re.compile("_Spectrum__P(01|03|_mu[46])")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    @sigma_v2.deleter
    def sigma_v2(self):
        try:
            del self.__sigma_v2
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
            return self.__sigma_bv2
        except:
            self.__sigma_bv2 = velocity.sigma_bv2(self.hmf, self.bias_model)
            return self.__sigma_bv2
    
    @sigma_bv2.setter
    def sigma_bv2(self, val):
        self.__sigma_bv2 = val
        
        pattern = re.compile("_Spectrum__P(02|12|13|22|_mu[2468])")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    @sigma_bv2.deleter
    def sigma_bv2(self):
        try:
            del self.__sigma_bv2
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
            return self.__sigma_bv4
        except:
            self.__sigma_bv4 = velocity.sigma_bv4(self.hmf, self.bias_model)
            return self.__sigma_bv4
            
    @sigma_bv4.setter
    def sigma_bv4(self, val):
        self.__sigma_bv4 = val
        
        pattern = re.compile("_Spectrum__P(04|_mu[46])")
        for k in self.__dict__.keys():
            if pattern.match(k): del self.__dict__[k]
    
    @sigma_bv4.deleter
    def sigma_bv4(self):
        try:
            del self.__sigma_bv4
        except:
            pass
    #---------------------------------------------------------------------------
    # READ-ONLY ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def D(self):
        try:
            return self.__D
        except:
            self.__D = growth.growth_function(self.z, normed=True, params=self.cosmo)
            return self.__D
    #---------------------------------------------------------------------------
    @property
    def f(self):
        try:
            return self.__f
        except:
            self.__f = growth.growth_rate(self.z, params=self.cosmo)
            return self.__f
    #---------------------------------------------------------------------------
    @property
    def conformalH(self):
        try:
            return self.__conformalH
        except:
            self.__conformalH = cosmo_tools.H(self.z,  params=self.cosmo)/(1.+self.z)
            return self.__conformalH
    #---------------------------------------------------------------------------
    @property
    def hmf(self):
        try:
            return self.__hmf
        except:
            self.__hmf = hmf.HaloMassFunction(**self.mass_function_kwargs)
            return self.__hmf
    
    @hmf.deleter
    def hmf(self):
        try:
            del self.__hmf
        except:
            pass
        del self.sigma_v2
        del self.sigma_bv2
        del self.sigma_bv4
    #---------------------------------------------------------------------------
    @property
    def two_loop(self):
        try:
            return self.__two_loop
        except:
            self.__two_loop = TwoLoopInterp(self.z, self.cosmo, self.kmin, 
                                            self.kmax, k=np.logspace(-4, 1, 500), 
                                            num_threads=self.num_threads)
            return self.__two_loop
    
    @two_loop.deleter
    def two_loop(self):
        try:
            del self.__two_loop
        except:
            pass
    #---------------------------------------------------------------------------
    @property
    def Plin(self):
        try:
            return self.__Plin
        except:
            self.__Plin = growth.Pk_full(self.k, 0., params=self.cosmo) 
            return self.__Plin
    #---------------------------------------------------------------------------
    @property
    def Plin_interp(self):
        try:
            return self.__Plin_interp
        except:
            self.__Plin_interp = growth.Pk_lin(self.klin_interp, 0., tf='EH', params=self.cosmo)
            return self.__Plin_interp
    #---------------------------------------------------------------------------        
    @property
    def I(self):
        try:
            return self.__I
        except:
            self.__I = integralsIJ.I_nm(0, 0, self.klin_interp, self.Plin_interp, k2=None, P2=None)
            return self.__I
    #---------------------------------------------------------------------------
    @property
    def J(self):
        try:
            return self.__J
        except:
            self.__J = integralsIJ.J_nm(0, 0, self.klin_interp, self.Plin_interp)
            return self.__J
    #---------------------------------------------------------------------------
    @property
    def sigma_lin(self):
        """
        The dark matter velocity dispersion, as evaluated in linear theory 
        [units: Mpc/h]
        """
        try:
            return self.__sigma_lin
        except:
            self.__sigma_lin = velocity.sigmav_lin(cosmo_params=self.cosmo)
            return self.__sigma_lin 
    #---------------------------------------------------------------------------
    # INTEGRAL ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def I00(self):
        try:
            return self.__I00
        except:
            self.I.n, self.I.m = 0, 0
            self.__I00 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I00
    #---------------------------------------------------------------------------
    @property
    def I01(self):
        try:
            return self.__I01
        except:
            self.I.n, self.I.m = 0, 1
            self.__I01 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I01
    #---------------------------------------------------------------------------
    @property
    def I10(self):
        try:
            return self.__I10
        except:
            self.I.n, self.I.m = 1, 0
            self.__I10 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I10
    #---------------------------------------------------------------------------
    @property
    def I11(self):
        try:
            return self.__I11
        except:
            self.I.n, self.I.m = 1, 1
            self.__I11 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I11
    #---------------------------------------------------------------------------
    @property
    def I02(self):
        try:
            return self.__I02
        except:
            self.I.n, self.I.m = 0, 2
            self.__I02 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I02
    #---------------------------------------------------------------------------
    @property
    def I20(self):
        try:
            return self.__I20
        except:
            self.I.n, self.I.m = 2, 0
            self.__I20 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I20
    #---------------------------------------------------------------------------
    @property
    def I12(self):
        try:
            return self.__I12
        except:
            self.I.n, self.I.m = 1, 2
            self.__I12 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I12
    #---------------------------------------------------------------------------
    @property
    def I21(self):
        try:
            return self.__I21
        except:
            self.I.n, self.I.m = 2, 1
            self.__I21 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I21
    #---------------------------------------------------------------------------
    @property
    def I22(self):
        try:
            return self.__I22
        except:
            self.I.n, self.I.m = 2, 2
            self.__I22 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I22
    #---------------------------------------------------------------------------
    @property
    def I03(self):
        try:
            return self.__I03
        except:
            self.I.n, self.I.m = 0, 3
            self.__I03 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I03
    #---------------------------------------------------------------------------
    @property
    def I30(self):
        try:
            return self.__I30
        except:
            self.I.n, self.I.m = 3, 0
            self.__I30 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I30
    #---------------------------------------------------------------------------
    @property
    def I31(self):
        try:
            return self.__I31
        except:
            self.I.n, self.I.m = 3, 1
            self.__I31 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I31
    #---------------------------------------------------------------------------
    @property
    def I13(self):
        try:
            return self.__I13
        except:
            self.I.n, self.I.m = 1, 3
            self.__I13 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I13
    #---------------------------------------------------------------------------
    @property
    def I23(self):
        try:
            return self.__I23
        except:
            self.I.n, self.I.m = 2, 3
            self.__I23 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I23
    #---------------------------------------------------------------------------
    @property
    def I32(self):
        try:
            return self.__I32
        except:
            self.I.n, self.I.m = 3, 2
            self.__I32 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I32
    #---------------------------------------------------------------------------
    @property
    def I33(self):
        try:
            return self.__I33
        except:
            self.I.n, self.I.m = 3, 3
            self.__I33 = self.I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I33
    #---------------------------------------------------------------------------
    @property
    def J00(self):
        try:
            return self.__J00
        except:
            self.J.n, self.J.m = 0, 0
            self.__J00 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self.__J00
    #---------------------------------------------------------------------------
    @property
    def J01(self):
        try:
            return self.__J01
        except:
            self.J.n, self.J.m = 0, 1
            self.__J01 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self.__J01
    #---------------------------------------------------------------------------
    @property
    def J10(self):
        try:
            return self.__J10
        except:
            self.J.n, self.J.m = 1, 0
            self.__J10 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self.__J10
    #---------------------------------------------------------------------------
    @property
    def J11(self):
        try:
            return self.__J11
        except:
            self.J.n, self.J.m = 1, 1
            self.__J11 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self.__J11
    #---------------------------------------------------------------------------
    @property
    def J20(self):
        try:
            return self.__J20
        except:
            self.J.n, self.J.m = 2, 0
            self.__J20 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self.__J20
    #---------------------------------------------------------------------------
    @property
    def J02(self):
        try:
            return self.__J02
        except:
            self.J.n, self.J.m = 0, 2
            self.__J02 = self.J.evaluate(self.k, self.kmin, self.kmax)
            return self.__J02
    #---------------------------------------------------------------------------
    # TWO LOOP INTEGRALS (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def I23_2loop(self):
        try:
            return self.__I23_2loop
        except:
            I = integralsIJ.I_nm(2, 3, self.two_loop.k, self.two_loop.Pvv, k2=None, P2=None)
            self.__I23_2loop = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I23_2loop
    #---------------------------------------------------------------------------
    @property
    def I32_2loop(self):
        try:
            return self.__I32_2loop
        except:
            I = integralsIJ.I_nm(3, 2, self.two_loop.k, self.two_loop.Pvv, k2=None, P2=None)
            self.__I32_2loop = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I32_2loop
    #---------------------------------------------------------------------------
    @property
    def I33_2loop(self):
        try:
            return self.__I33_2loop
        except:
            I = integralsIJ.I_nm(3, 3, self.two_loop.k, self.two_loop.Pvv, k2=None, P2=None)
            self.__I33_2loop = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I33_2loop
    #---------------------------------------------------------------------------
    @property
    def I04(self):
        try:
            return self.__I04
        except:
            I = integralsIJ.I_nm(0, 4, self.two_loop.k, self.two_loop.Pvv, k2=self.two_loop.k, P2=self.two_loop.Pdd)
            self.__I04 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I04
    #---------------------------------------------------------------------------
    @property
    def I40(self):
        try:
            return self.__I40
        except:
            I = integralsIJ.I_nm(4, 0, self.two_loop.k, self.two_loop.Pdv, k2=None, P2=None)
            self.__I40 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I40
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
            return self.__P_mu0
        except:
            self.__P_mu0 = self.P00.total.mu0
            return self.__P_mu0
    #---------------------------------------------------------------------------
    @property
    def P_mu2(self):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        try:
            return self.__P_mu2
        except:
            self.__P_mu2 = self.P01.total.mu2 + self.P11.total.mu2 + self.P02.total.mu2
            return self.__P_mu2
    #---------------------------------------------------------------------------
    @property
    def P_mu4(self):
        """
        The full power spectrum term with mu^4 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13 (2-loop), and P04 (2-loop)
        """
        try:
            return self.__P_mu4
        except:
            self.__P_mu4 = self.P11.total.mu4 + self.P02.total.mu4 + \
                            self.P12.total.mu4 + self.P22.total.mu4 + \
                            self.P03.total.mu4 
            if self.include_2loop:
                self.__P_mu4 += (self.P13.total.mu4 + self.P04.total.mu4)
            return self.__P_mu4
    #---------------------------------------------------------------------------
    @property
    def P_mu6(self):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13, and P04 (2-loop).
        """
        try:
            return self.__P_mu6
        except:
            self.__P_mu6 = self.P12.total.mu6 + self.P22.total.mu6 + \
                            self.P13.total.mu6
            if self.include_2loop:
                self.__P_mu6 += self.P04.total.mu6
            return self.__P_mu6
    #---------------------------------------------------------------------------
    @property
    def P_mu8(self):
        """
        The full power spectrum term with mu^8 angular dependence. Contributions
        from P22. 
        """
        try:
            return self.__P_mu8
        except:
            if self.include_2loop:
                self.__P_mu8 = self.P22.total.mu8
            else:
                self.__P_mu8 = np.zeros(len(self.k))
            return self.__P_mu8
    #---------------------------------------------------------------------------
    @property
    def P00(self):
        """
        The isotropic, zero-order term in the power expansion, corresponding
        to the density field auto-correlation. No angular dependence.
        """
        try:
            return self.__P00
        except:
            self.__P00 = PowerTerm()
            P11 = self.Plin
            P22 = 2*self.I00
            P13 = 3*self.k**2 * P11*self.J00
            self.__P00.total.mu0 = self.D**2*P11 + self.D**4*(P22 + 2*P13)
            return self.__P00
    #---------------------------------------------------------------------------
    @property
    def P01(self):
        """
        The correlation of density and momentum density, which contributes
        mu^2 terms to the power expansion.
        """
        try:
            return self.__P01
        except:
            self.__P01 = PowerTerm()
            A = 2.*self.f*self.D**2
            self.__P01.total.mu2 = A*(self.Plin + 4.*self.D**2*(self.I00 + 3*self.k**2*self.J00*self.Plin))
            return self.__P01
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
            return self.__P11
        except:
            self.__P11 = PowerTerm()
            
            # first do the scalar part, contributing only mu^4 terms 
            P_scalar = (self.f*self.D)**2*(self.Plin + self.D**2*(8.*self.I00 + 18.*self.J00*self.k**2*self.Plin))
            self.__P11.scalar.mu4 = P_scalar
            
            # now do the vector part, contribution mu^2 and mu^4 terms
            if not self.include_2loop:
                P_vector = (self.f*self.D**2)**2 * self.I31
            else:
                P_vector = (self.I04 + self.I40)/self.conformalH**2
            
            self.__P11.vector.mu2 = P_vector
            self.__P11.vector.mu4 = P_vector

            # now save the total angular dependences
            self.__P11.total.mu2 = self.__P11.vector.mu2
            self.__P11.total.mu4 = self.__P11.scalar.mu4 + self.__P11.vector.mu4
            
            return self.__P11
    #---------------------------------------------------------------------------
    @property
    def P02(self):
        """
        The correlation of density and energy density, which contributes
        mu^2 and mu^4 terms to the power expansion. There are no 
        linear contributions here.
        """
        try:
            return self.__P02
        except:
            self.__P02 = PowerTerm()
            
            # first do the no velocity terms (mu^2, mu^4 dependences)
            P02_no_vel_mu2 = (self.f*self.D**2)**2 * (self.I02 + 2*self.k**2*self.J02*self.Plin)
            P02_no_vel_mu4 = (self.f*self.D**2)**2 * (self.I20 + 2*self.k**2*self.J20*self.Plin)
            self.__P02.no_velocity.mu2 = P02_no_vel_mu2
            self.__P02.no_velocity.mu4 = P02_no_vel_mu4
            
            # now do the terms depending on velocity (mu^2 dependence)
            sigma_lin = self.sigma_lin # units are Mpc/h
            sigma_02  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) # units are now Mpc/h
            vel_terms = sigma_lin**2 + sigma_02**2

            if self.include_2loop:
                P00 = self.P00
                P02_vel_mu2 = -(self.f*self.D*self.k)**2 * vel_terms * P00.total.mu0
            else:
                P02_vel_mu2 = -(self.f*self.D**2*self.k)**2 * vel_terms * self.Plin
            self.__P02.with_velocity.mu2 = P02_vel_mu2
            
            # now save the total angular dependences
            self.__P02.total.mu2 = self.__P02.with_velocity.mu2 + self.__P02.no_velocity.mu2
            self.__P02.total.mu4 = self.__P02.no_velocity.mu4
            
            self.__P02.anisotropic = 2./3*self.__P02.total.mu4
            fac = 1./3.*(self.f*self.D**2)**2
            self.__P02.isotropic = fac*(3*self.I02 + self.I20 + 2*self.k**2*(3*self.J02 + self.J20)*self.Plin) + self.__P02.with_velocity.mu2
            
            return self.__P02
    #---------------------------------------------------------------------------
    @property
    def P12(self):
        """
        The correlation of momentum density and energy density, which contributes
        mu^4 and mu^6 terms to the power expansion. There are no 
        linear contributions here. Two-loop contribution uses the mu^2 
        contribution from the P01 term.
        """
        try:
            return self.__P12
        except:
            self.__P12 = PowerTerm()
            
            # first do the no velocity terms (mu^4, mu^6 dependences)
            P12_no_vel_mu4 = self.f**3 * self.D**4 * (self.I12 - self.I03 + 2*self.k**2*self.J02*self.Plin)
            P12_no_vel_mu6 = self.f**3 * self.D**4 * (self.I21 - self.I30 + 2*self.k**2*self.J20*self.Plin)
            self.__P12.no_velocity.mu4 = P12_no_vel_mu4
            self.__P12.no_velocity.mu6 = P12_no_vel_mu6
            
            # now do the terms depending on velocity (mu^4 dependence)
            sigma_lin = self.sigma_lin   # units are Mpc/h
            sigma_12  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) # units are now Mpc/h
            vel_terms = sigma_lin**2 + sigma_12**2
            
            if self.include_2loop:
                P01 = self.P01
                P12_vel_mu4 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * P01.total.mu2
            else:
                P12_vel_mu4 = -self.f**3 * self.D**4 * self.k**2 * vel_terms * self.Plin
            self.__P12.with_velocity.mu4 = P12_vel_mu4
            
            # now save the total angular dependences
            self.__P12.total.mu4 = self.__P12.with_velocity.mu4 + self.__P12.no_velocity.mu4
            self.__P12.total.mu6 = self.__P12.no_velocity.mu6
            
            return self.__P12
    #---------------------------------------------------------------------------
    @property
    def P22(self):
        """
        The autocorelation of energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no 
        linear contributions here. 
        """
        try:
            return self.__P22
        except:
            self.__P22 = PowerTerm()
            
            # do (either 1 loop or 2 loop) no velocity terms 
            # (mu^4, mu^6, mu^8 dependences)
            if not self.include_2loop:
                A = 1./16 * (self.f*self.D)**4
                P22_no_vel_mu4 = A * self.I23
                P22_no_vel_mu6 = A * 2*self.I32
                P22_no_vel_mu8 = A * self.I33
            else:
                A = (1./16) / self.conformalH**4
                P22_no_vel_mu4 = A * self.I23_2loop
                P22_no_vel_mu6 = A * 2*self.I32_2loop
                P22_no_vel_mu8 = A * self.I33_2loop

            self.__P22.no_velocity.mu4 = P22_no_vel_mu4
            self.__P22.no_velocity.mu6 = P22_no_vel_mu6
            self.__P22.no_velocity.mu8 = P22_no_vel_mu8
            
            self.__P22.total.mu4 = self.__P22.no_velocity.mu4
            self.__P22.total.mu6 = self.__P22.no_velocity.mu6
            self.__P22.total.mu8 = self.__P22.no_velocity.mu8
            
            # now add in the extra 2 loop terms, if specified
            if self.include_2loop:
                               
                sigma_lin = self.sigma_lin   # units are Mpc/h
                sigma_22  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) # units are now Mpc/h
                vel_terms = sigma_lin**2 + sigma_22**2
                P02 = self.P02
                P00 = self.P00
                
                extra1_mu4 = 0.5*(self.f*self.D*self.k)**2 * vel_terms * P02.no_velocity.mu2
                extra1_mu6 = 0.5*(self.f*self.D*self.k)**2 * vel_terms * P02.no_velocity.mu4
                extra2_mu4 = 0.25*(self.f*self.D*self.k)**4 * vel_terms**2 * P00.total.mu0
                
                A = 0.25*(self.k/self.conformalH)**4
                I = integralsIJ.I_nm(-1, -1, self.two_loop.k, self.two_loop.P22_bar.total.mu4, k2=self.two_loop.k, P2=self.two_loop.Pdd)
                self.extra3_mu4 = A*I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
                I = integralsIJ.I_nm(-1, -1, self.two_loop.k, self.two_loop.P22_bar.total.mu6, k2=self.two_loop.k, P2=self.two_loop.Pdd)
                self.extra3_mu6 = A*I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
                I = integralsIJ.I_nm(-1, -1, self.two_loop.k, self.two_loop.P22_bar.total.mu8, k2=self.two_loop.k, P2=self.two_loop.Pdd)
                self.extra3_mu8 = A*I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
                
                self.__P22.with_velocity.mu4 = extra1_mu4 + extra2_mu4 
                self.__P22.with_velocity.mu6 = extra1_mu6
                self.__P22.no_velocity.mu4 += self.extra3_mu4
                self.__P22.no_velocity.mu6 += self.extra3_mu6
                self.__P22.no_velocity.mu8 += self.extra3_mu8
                
                self.__P22.total.mu4 = (self.__P22.with_velocity.mu4 + self.__P22.no_velocity.mu4)
                self.__P22.total.mu6 = (self.__P22.with_velocity.mu6 + self.__P22.no_velocity.mu6)
                self.__P22.total.mu8 = self.__P22.no_velocity.mu8
            
            return self.__P22
    #---------------------------------------------------------------------------
    @property
    def P03(self):
        """
        The cross-corelation of density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^4 terms.
        """
        try:
            return self.__P03
        except:
            self.__P03 = PowerTerm()
            
            sigma_lin = self.sigma_lin # units are Mpc/h
            sigma_03  = self.sigma_v2 * self.cosmo.h / (self.f*self.conformalH*self.D) # units are now Mpc/h
            vel_terms = sigma_lin**2 + sigma_03**2

            if self.include_2loop:
                P01 = self.P01
                P03_vel_mu4 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * P01.total.mu2
            else:
                P03_vel_mu4 = -(self.f**3*self.D**4) * self.k**2 * vel_terms * self.Plin
            
            self.__P03.total.mu4 = self.__P03.with_velocity.mu4 = P03_vel_mu4

            return self.__P03
    #---------------------------------------------------------------------------
    @property
    def P13(self):
        """
        The cross-correlation of momentum density with the rank three tensor field
        ((1+delta)v)^3, which contributes mu^6 terms at 1-loop order and 
        mu^4 terms at 2-loop order.
        """
        try:
            return self.__P13
        except:
            self.__P13 = PowerTerm()
            
            sigma_lin = self.sigma_lin # units are Mpc/h
            sigma_13_v  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
            vel_terms_vector = sigma_lin**2 + sigma_13_v**2
    
            if self.include_2loop:
                P11 = self.P11
                sigma_13_s  = self.sigma_v2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                vel_terms_scalar = sigma_lin**2 + sigma_13_s**2
                A = -(self.f*self.D*self.k)**2
                P13_vel_mu4 = A*vel_terms_vector*P11.vector.mu2
                P13_vel_mu6 = A*vel_terms_scalar*(P11.scalar.mu4 + P11.vector.mu4)
                self.__P13.total.mu4 = self.__P13.with_velocity.mu4 = P13_vel_mu4
            else:
                P13_vel_mu6 = -(self.f**4*self.D**4) * self.k**2 * vel_terms_vector * self.Plin

            self.__P13.total.mu6 = self.__P13.with_velocity.mu6 = P13_vel_mu6
            return self.__P13
    #---------------------------------------------------------------------------
    @property
    def P04(self):
        """
        The cross-correlation of density with the rank four tensor field
        ((1+delta)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        try:
            return self.__P04
        except:
            self.__P04 = PowerTerm()
            
            if self.include_2loop:
                
                sigma_lin = self.sigma_lin 
                sigma_04  = self.sigma_bv4 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                vel_terms = sigma_lin**2 + sigma_04**2
                
                P02_bar = self.P02.no_velocity
                P04_mu4_1 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * P02_bar.mu2
                P04_mu4_2 = 0.25*(self.f*self.D*self.k)**4 * vel_terms**2 * self.P00.total.mu0
                
                A = 0.25*(self.k/self.conformalH)**4
                P04_mu4_3 = A/12 * self.P00.total.mu0 * intgr.simps(self.two_loop.k**3*self.two_loop.P22_bar.total.mu4, x=np.log(self.two_loop.k))/(2*np.pi**2)
                
                P04_mu6 = -0.5*(self.f*self.D*self.k)**2 * vel_terms * P02_bar.mu4
                
                self.P04.with_velocity.mu4 = P04_mu4_1 + P04_mu4_2
                self.P04.no_velocity.mu4   = P04_mu4_3
                self.P04.with_velocity.mu6 = P04_mu6
                
                self.P04.total.mu4 = self.P04.with_velocity.mu4 + self.P04.no_velocity.mu4
                self.P04.total.mu6 = self.P04.with_velocity.mu6
            return self.__P04
    #---------------------------------------------------------------------------
    def monopole(self, max_mu=4):
        """
        The monopole moment of the power spectrum. Include mu terms up to 
        mu^max_mu.
        """
        if max_mu == 0:
            return self.P_mu0
        elif max_mu == 2:
            return self.P_mu0 + (1./3)*self.P_mu2
        elif max_mu == 4:
            return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4
        elif max_mu == 6:
            return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4 + (1./7)*self.P_mu6
        elif max_mu == 8:
            return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4 + (1./7)*self.P_mu6 + (1./9)*self.P_mu8

    #end monopole
    #---------------------------------------------------------------------------
    def quadrupole(self, max_mu=6):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to 
        mu^max_mu.
        """
        if max_mu == 2:
            return (2./3)*self.P_mu2
        elif max_mu == 4:
            return (2./3)*self.P_mu2 + (4./7)*self.P_mu4
        elif max_mu == 6:
            return (2./3)*self.P_mu2 + (4./7)*self.P_mu4 + (10./21)*self.P_mu6
        elif max_mu == 8:
            return (2./3)*self.P_mu2 + (4./7)*self.P_mu4 + (10./21)*self.P_mu6 + (40./99)*self.P_mu8
    #end quadrupole
    #---------------------------------------------------------------------------
    def hexadecapole(self, max_mu=6):
        """
        The hexadecapole moment of the power spectrum. Include mu terms up to 
        mu^max_mu.
        """
        if max_mu == 4:
            return (8./35)*self.P_mu4
        elif max_mu == 6:
            return (8./35)*self.P_mu4 + (24./77)*self.P_mu6
        elif max_mu == 8:
            return (8./35)*self.P_mu4 + (24./77)*self.P_mu6 + (48./143)*self.P_mu8
    #end hexadecapole
#endclass Spectrum 

#-------------------------------------------------------------------------------
class TwoLoopInterp(object):
    """
    A class to hold two loop terms for interpolation purposes.
    """
    def __init__(self, z, cosmo, kmin, kmax, k=np.logspace(-4, 1, 500), num_threads=1):
        self.k = k
        self.z = z
        self.num_threads = num_threads
        self.cosmo = cosmo
        self.kmin, self.kmax = kmin, kmax
    #---------------------------------------------------------------------------
    # 1-loop power spectra
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        try:
            return self.__Pvv
        except:
            self.__Pvv = power_1loop.Pvv_1loop(self.k, self.z, num_threads=self.num_threads, cosmo=self.cosmo)
            return self.__Pvv        
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        try:
            return self.__Pdv
        except:
            self.__Pdv = power_1loop.Pdv_1loop(self.k, self.z, num_threads=self.num_threads, cosmo=self.cosmo)
            return self.__Pdv        
    #---------------------------------------------------------------------------
    @property
    def Pdd(self):
        try:
            return self.__Pdd
        except:
            self.__Pdd = power_1loop.Pdd_1loop(self.k, self.z, num_threads=self.num_threads, cosmo=self.cosmo)
            return self.__Pdd            
    #---------------------------------------------------------------------------
    # INTEGRAL ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def I23(self):
        try:
            return self.__I23
        except:
            I = integralsIJ.I_nm(2, 3, self.k, self.Pvv, k2=None, P2=None)
            self.__I23 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I23
    #---------------------------------------------------------------------------
    @property
    def I32(self):
        try:
            return self.__I32
        except:
            I = integralsIJ.I_nm(3, 2, self.k, self.Pvv, k2=None, P2=None)
            self.__I32 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I32
    #---------------------------------------------------------------------------
    @property
    def I33(self):
        try:
            return self.__I33
        except:
            I = integralsIJ.I_nm(3, 3, self.k, self.Pvv, k2=None, P2=None)
            self.__I33 = I.evaluate(self.k, self.kmin, self.kmax, self.num_threads)
            return self.__I33
    #---------------------------------------------------------------------------
    @property
    def P22_bar(self):
        """
        The contribution to the P22 with no velocity terms, at 2 loop order.
        """
        
        try:
            return self.__P22_bar
        except:
            self.__P22_bar = PowerTerm()        
            A = 0.25/self.k**4
            P22_no_vel_mu4 = A * self.I23
            P22_no_vel_mu6 = A * 2*self.I32
            P22_no_vel_mu8 = A * self.I33

            self.__P22_bar.no_velocity.mu4 = P22_no_vel_mu4
            self.__P22_bar.no_velocity.mu6 = P22_no_vel_mu6
            self.__P22_bar.no_velocity.mu8 = P22_no_vel_mu8
            
            # now save the total angular dependences
            self.__P22_bar.total.mu4 = self.__P22_bar.no_velocity.mu4
            self.__P22_bar.total.mu6 = self.__P22_bar.no_velocity.mu6
            self.__P22_bar.total.mu8 = self.__P22_bar.no_velocity.mu8
            return self.__P22_bar
    #---------------------------------------------------------------------------
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
        
        # isotropic/anisotropic
        self.isotropic   = None
        self.anisotropic = None
        
class Angular(object):
    
    def __init__(self):
        self.mu0 = None
        self.mu2 = None
        self.mu4 = None
        self.mu6 = None
        self.mu8 = None
        
        
