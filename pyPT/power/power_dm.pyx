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
import copy

class DMSpectrum(object):
    
    CONVERGENCE_FACTOR = 100.
    
    def __init__(self, k=np.logspace(-2, np.log10(0.5), 100),
                       z=0., 
                       num_threads=1, 
                       cosmo_params="Planck1_lens_WP_highL", 
                       cosmo_kwargs={'default':"Planck1_lens_WP_highL", 'force_flat':False},
                       mass_function_kwargs={'mf_fit' : 'Tinker'}, 
                       bias_model='Tinker',
                       include_2loop=False,
                       linear_power=None,
                       max_mu=4):
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
            
        linear_power : str, optional
            The name of file name holding the linear power spectrum to use. If
            ``None``, analytic Eisenstein + Hu fit is used. 
        
        max_mu : {0, 2, 4, 6, 8}, optional
            Only compute angular terms up to mu**(``max_mu``). Default is 4.
        """
        # initialize the cosmology parameters
        self.max_mu          = max_mu
        self.cosmo_kwargs    = cosmo_kwargs
        self.cosmo           = cosmo_params
        self.num_threads     = num_threads
        self.include_2loop   = include_2loop
        self.k               = np.array(k, copy=False, ndmin=1)
        self.linear_power    = linear_power
                    
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
        self._k = val
        self._delete_all()
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, val):
        self._z = val
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
        del self.base_spectra
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
        del self.base_spectra
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
    def _kmin(self):
        return np.amin(self.k)

    #---------------------------------------------------------------------------
    @property
    def _kmax(self):
        return np.amax(self.k)
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
    def Plin(self):
        try:
            return self._Plin
        except:
            if self.linear_power is None:
                self._Plin = growth.Pk_lin(self.k, 0., tf='EH', params=self.cosmo) 
            else:
                Plin_data = np.loadtxt(self.linear_power)
                spline = interp.InterpolatedUnivariateSpline(Plin_data[:,0], Plin_data[:,1])
                self._Plin = spline(self.k)
            return self._Plin
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
    def base_spectra(self):
        try:
            return self._base_spectra
        except:
            self._base_spectra = DMBaseSpectra(self.z, self._kmin, self._kmax, self.cosmo, self.num_threads)
            return self._base_spectra
            
    @base_spectra.deleter
    def base_spectra(self):
        try:
            del self._base_spectra
        except:
            pass
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
            self.base_spectra.I.n, self.base_spectra.I.m = 0, 0
            self._I00 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I00
    #---------------------------------------------------------------------------
    @property
    def I01(self):
        try:
            return self._I01
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 0, 1
            self._I01 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I01
    #---------------------------------------------------------------------------
    @property
    def I10(self):
        try:
            return self._I10
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 1, 0
            self._I10 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I10
    #---------------------------------------------------------------------------
    @property
    def I11(self):
        try:
            return self._I11
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 1, 1
            self._I11 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I11
    #---------------------------------------------------------------------------
    @property
    def I02(self):
        try:
            return self._I02
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 0, 2
            self._I02 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I02
    #---------------------------------------------------------------------------
    @property
    def I20(self):
        try:
            return self._I20
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 2, 0
            self._I20 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I20
    #---------------------------------------------------------------------------
    @property
    def I12(self):
        try:
            return self._I12
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 1, 2
            self._I12 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I12
    #---------------------------------------------------------------------------
    @property
    def I21(self):
        try:
            return self._I21
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 2, 1
            self._I21 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I21
    #---------------------------------------------------------------------------
    @property
    def I22(self):
        try:
            return self._I22
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 2, 2
            self._I22 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I22
    #---------------------------------------------------------------------------
    @property
    def I03(self):
        try:
            return self._I03
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 0, 3
            self._I03 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I03
    #---------------------------------------------------------------------------
    @property
    def I30(self):
        try:
            return self._I30
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 3, 0
            self._I30 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I30
    #---------------------------------------------------------------------------
    @property
    def I31(self):
        try:
            return self._I31
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 3, 1
            self._I31 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I31
    #---------------------------------------------------------------------------
    @property
    def I13(self):
        try:
            return self._I13
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 1, 3
            self._I13 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I13
    #---------------------------------------------------------------------------
    @property
    def I23(self):
        try:
            return self._I23
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 2, 3
            self._I23 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I23
    #---------------------------------------------------------------------------
    @property
    def I32(self):
        try:
            return self._I32
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 3, 2
            self._I32 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I32
    #---------------------------------------------------------------------------
    @property
    def I33(self):
        try:
            return self._I33
        except:
            self.base_spectra.I.n, self.base_spectra.I.m = 3, 3
            self._I33 = self.base_spectra.I.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin, self.num_threads)
            return self._I33
    #---------------------------------------------------------------------------
    @property
    def J00(self):
        try:
            return self._J00
        except:
            self.base_spectra.J.n, self.base_spectra.J.m = 0, 0
            self._J00 = self.base_spectra.J.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin)
            return self._J00
    #---------------------------------------------------------------------------
    @property
    def J01(self):
        try:
            return self._J01
        except:
            self.base_spectra.J.n, self.base_spectra.J.m = 0, 1
            self._J01 = self.base_spectra.J.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin)
            return self._J01
    #---------------------------------------------------------------------------
    @property
    def J10(self):
        try:
            return self._J10
        except:
            self.base_spectra.J.n, self.base_spectra.J.m = 1, 0
            self._J10 = self.base_spectra.J.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin)
            return self._J10
    #---------------------------------------------------------------------------
    @property
    def J11(self):
        try:
            return self._J11
        except:
            self.base_spectra.J.n, self.base_spectra.J.m = 1, 1
            self._J11 = self.base_spectra.J.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin)
            return self._J11
    #---------------------------------------------------------------------------
    @property
    def J20(self):
        try:
            return self._J20
        except:
            self.base_spectra.J.n, self.base_spectra.J.m = 2, 0
            self._J20 = self.base_spectra.J.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin)
            return self._J20
    #---------------------------------------------------------------------------
    @property
    def J02(self):
        try:
            return self._J02
        except:
            self.base_spectra.J.n, self.base_spectra.J.m = 0, 2
            self._J02 = self.base_spectra.J.evaluate(self.k, self.base_spectra.kmin_lin, self.base_spectra.kmax_lin)
            return self._J02
    #---------------------------------------------------------------------------
    # TWO LOOP INTEGRALS (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def I23_2loop(self):
        try:
            return self._I23_2loop
        except:
            I = integralsIJ.I_nm(2, 3, self.base_spectra.k1loop, self.base_spectra.Pvv, k2=None, P2=None)
            self._I23_2loop = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I23_2loop
    #---------------------------------------------------------------------------
    @property
    def I32_2loop(self):
        try:
            return self._I32_2loop
        except:
            I = integralsIJ.I_nm(3, 2, self.base_spectra.k1loop, self.base_spectra.Pvv, k2=None, P2=None)
            self._I32_2loop = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I32_2loop
    #---------------------------------------------------------------------------
    @property
    def I33_2loop(self):
        try:
            return self._I33_2loop
        except:
            I = integralsIJ.I_nm(3, 3, self.base_spectra.k1loop, self.base_spectra.Pvv, k2=None, P2=None)
            self._I33_2loop = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I33_2loop
    #---------------------------------------------------------------------------
    @property
    def I04(self):
        try:
            return self._I04
        except:
            I = integralsIJ.I_nm(0, 4, self.base_spectra.k1loop, self.base_spectra.Pvv, k2=self.base_spectra.k1loop, P2=self.base_spectra.Pdd)
            self._I04 = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I04
    #---------------------------------------------------------------------------
    @property
    def I40(self):
        try:
            return self._I40
        except:
            I = integralsIJ.I_nm(4, 0, self.base_spectra.k1loop, self.base_spectra.Pdv, k2=None, P2=None)
            self._I40 = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I40
    #---------------------------------------------------------------------------
    @property
    def I14(self):
        try:
            return self._I14
        except:
            I = integralsIJ.I_nm(1, 4, self.base_spectra.k1loop, self.base_spectra.Pvv, k2=self.base_spectra.k1loop, P2=self.base_spectra.Pdd)
            self._I14 = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I14
    #---------------------------------------------------------------------------
    @property
    def I41(self):
        try:
            return self._I41
        except:
            I = integralsIJ.I_nm(4, 1, self.base_spectra.k1loop, self.base_spectra.Pdv, k2=None, P2=None)
            self._I41 = I.evaluate(self.k, self.base_spectra.kmin_1loop, self.base_spectra.kmax_1loop, self.num_threads)
            return self._I41
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
    def Pdd(self):
        """
        The 1-loop auto-correlation of density.
        """
        return self.P00
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        """
        The 1-loop cross-correlation between dark matter density and velocity 
        divergence.
        """
        try:
            return self._Pdv
        except:
            P11  = self.Plin
            P22  = 2.*self.I01
            P13  = 6.*self.k**2*self.Plin*self.J01

            self._Pdv = -self.f*self.D**2 * (P11 + self.D**2*(P22 + P13))
            return self._Pdv
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        """
        The 1-loop auto-correlation of velocity divergence.
        """
        try:
            return self._Pvv
        except:
            P11  = self.Plin
            P22  = 2.*self.I11
            P13  = 6.*self.k**2*self.Plin*self.J11

            self._Pvv = (self.f*self.D)**2 * (P11 + self.D**2*(P22 + P13))
            return self._Pvv
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
                    Pvec = (self.f*self.D**2)**2 * self.I31
                else:
                    Pvec = self.I04 + self.I40
                
                # save the mu^2 vector term
                self._P11.vector.mu2 = self._P11.total.mu2 = Pvec
                self._P11.vector.mu4 = self._P11.vector.mu4 = -Pvec
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                        
                    # compute the scalar mu^4 contribution
                    if self.include_2loop:
                        C11_contrib = self.I14 + self.I41
                    else:
                        C11_contrib = (self.f*self.D**2)**2 * self.I13
                        
                    part2 = self.D**2 * (2*self.I11 + 4*self.I22 + 6*self.k**2 * (self.J11 + 2*self.J10)*self.Plin)
                    P_scalar = (self.f*self.D)**2*(self.Plin + part2) + C11_contrib - self._P11.vector.mu4
                    
                    # save the scalar/vector mu^4 terms
                    self._P11.scalar.mu4 = P_scalar
                    self._P11.total.mu4 = self._P11.scalar.mu4 + self._P11.vector.mu4
            
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
                sigsq_eff = sigma_lin**2 + sigma_02**2

                if self.include_2loop:
                    self._P02.with_velocity.mu2 = -(self.f*self.D*self.k)**2 * sigsq_eff * self.P00.total.mu0
                else:
                    self._P02.with_velocity.mu2 = -(self.f*self.D**2*self.k)**2 * sigsq_eff * self.Plin
            
                # save the total mu^2 term
                self._P02.total.mu2 = self._P02.with_velocity.mu2 + self._P02.no_velocity.mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                    
                    # only a no velocity term for mu^4
                    self._P02.total.mu4 = self._P02.no_velocity.mu4 = (self.f*self.D**2)**2 * (self.I20 + 2*self.k**2*self.J20*self.Plin)
                    
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
                sigsq_eff = sigma_lin**2 + sigma_12**2
            
                if self.include_2loop:
                    self._P12.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01.total.mu2
                else:
                    self._P12.with_velocity.mu4 = -self.f**3*self.D**4*self.k**2 * sigsq_eff * self.Plin
            
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
                sigsq_eff = sigma_lin**2 + sigma_22**2
                
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                # 1-loop or 2-loop terms that don't depend on velocity
                if not self.include_2loop:
                    self._P22.no_velocity.mu4 = (1./16)*(self.f*self.D)**4 * self.I23
                else:
                    self._P22.no_velocity.mu4 = (1./16)*self.I23_2loop

                # now add in the extra 2 loop terms, if specified
                if self.include_2loop:
                               
                    # 2 loop terms have P02 and P00 weighted by velocity
                    extra1_vel_mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                    extra2_vel_mu4 = 0.25*(self.f*self.D*self.k)**4 * sigsq_eff**2 * self.P00.total.mu0
                                        
                    # last term is convolution of P22_bar and P00
                    A = 0.25 * self.k**4
                    convolve_mu0 = self.base_spectra.convolve('k1loop', 'P22bar_mu0', 'k1loop', 'Pdd', self.k)
                    convolve_mu2 = self.base_spectra.convolve('k1loop', 'P22bar_mu2', 'k1loop', 'Pdd', self.k)
                    convolve_mu4 = self.base_spectra.convolve('k1loop', 'P22bar_mu4', 'k1loop', 'Pdd', self.k)
                    convolve = A*(convolve_mu0 + (1./3)*convolve_mu2 + (1./5)*convolve_mu4)
                    
                    # extra 2 loop term from modeling <v^2|v^2>
                    extra1_bar_mu4 = (self.f*self.k*self.D)**4 * self.D**2 * self.Plin * self.J02**2
                    
                    # store the extra two loop terms
                    self._P22.with_velocity.mu4 = extra1_vel_mu4 + extra2_vel_mu4 + convolve
                    self._P22.no_velocity.mu4 += extra1_bar_mu4 
                    self._P22.total.mu4 = self._P22.with_velocity.mu4 + self._P22.no_velocity.mu4
                    
                else:
                    self._P22.total.mu4 = self._P22.no_velocity.mu4
                    
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # 1-loop or 2-loop terms that don't depend on velocity
                    if not self.include_2loop:
                        self._P22.no_velocity.mu6 = 2.*(1./16)*(self.f*self.D)**4 * self.I32
                    else:
                        self._P22.no_velocity.mu6 = 2.*(1./16)*self.I32_2loop
                        
                    # now add in the extra 2 loop terms, if specified
                    if self.include_2loop:

                        # 2 loop terms have P02 weighted by velocity
                        extra1_vel_mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                        
                        # extra 2 loop term from modeling <v^2|v^2>
                        extra1_bar_mu6 = 2*(self.f*self.k*self.D)**4 * self.D**2 * self.Plin * self.J02*self.J20
                        
                        # save the totals
                        self._P22.no_velocity.mu6 += extra1_bar_mu6
                        self._P22.with_velocity.mu6 = extra1_vel_mu6
                        self._P22.total.mu6 = self._P22.no_velocity.mu6 + self._P22.with_velocity.mu6
                        
                    else:
                        self._P22.total.mu6 = self._P22.no_velocity.mu6

                    # do mu^8 terms?
                    if self.max_mu >= 8:
                        
                        # 1-loop or 2-loop terms that don't depend on velocity
                        if not self.include_2loop:
                            self._P22.no_velocity.mu8 = (1./16)*(self.f*self.D)**4 * self.I33
                        else:
                            self._P22.no_velocity.mu8 = (1./16)*self.I33_2loop
                            
                            # extra 2 loop term from modeling <v^2|v^2>
                            self._P22.no_velocity.mu8 += (self.f*self.k*self.D)**4 * self.D**2 * self.Plin*self.J20**2
                            
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
                sigsq_eff = sigma_lin**2 + sigma_03**2

                # either 1 or 2 loop quantities
                if self.include_2loop:
                    self._P03.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01.total.mu2
                else:
                    self._P03.with_velocity.mu4 = -(self.f**3*self.D**4) * self.k**2 * sigsq_eff * self.Plin
            
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
            sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
            
            if self.include_2loop:
                sigma_13_s  = self.sigma_v2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                sigsq_eff_scalar = sigma_lin**2 + sigma_13_s**2
                
            # do mu^4 terms?
            if self.max_mu >= 4:
            
                # mu^4 is only 2-loop
                if self.include_2loop:

                    A = -(self.f*self.D*self.k)**2
                    P13_vel_mu4 = A*sigsq_eff_vector*self.P11.total.mu2
                    self._P13.total.mu4 = self._P13.with_velocity.mu4 = P13_vel_mu4

                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # mu^6 velocity terms at 1 or 2 loop
                    if self.include_2loop:
                        A = -(self.f*self.D*self.k)**2
                        self._P13.with_velocity.mu6 = A*sigsq_eff_scalar*self.P11.total.mu4
                    else:
                        self._P13.with_velocity.mu6 = -(self.f*self.D)**4 *self.k**2 * sigsq_eff_scalar * self.Plin
                        
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
                sigsq_eff = sigma_lin**2 + sigma_04**2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    # do P04 mu^4 terms depending on velocity
                    P04_vel_mu4_1 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                    P04_vel_mu4_2 = 0.25*(self.f*self.D*self.k)**4 * sigsq_eff**2 * self.P00.total.mu0
                    self.P04.with_velocity.mu4 = P04_vel_mu4_1 + P04_vel_mu4_2
                    
                    # do P04 mu^4 terms without vel dependence
                    self.P04.no_velocity.mu4 = (1/12.)*self.k**4 * self.P00.total.mu0*self.base_spectra.vel_kurtosis
                
                    # save the total
                    self.P04.total.mu4 = self.P04.with_velocity.mu4 + self.P04.no_velocity.mu4
                
                    # do mu^6 terms?
                    if self.max_mu >= 6:
                        
                        # only terms depending on velocity
                        self.P04.with_velocity.mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
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
        kmin, kmax = np.amin(k_data), np.amax(k_data)
        if np.amin(self.k) < kmin or np.amax(self.k) > kmax:
            raise ValueError("Input data to load not defined over entire k range. Interpolation errors will occur.")
        
        power_name = "_%s" %power_term
        if hasattr(self, power_name):
            power_attr = getattr(self, power_name)
        else:
            if mu_term is not None:
                
                # initialize the object so other mu terms are not zero
                getattr(self, power_term)
                power_attr = self.__dict__[power_name]
        
        if errs is not None:
            w = 1./np.array(errs)**2
        else:
            w = None
        s = interp.InterpolatedUnivariateSpline(k_data, power_data, w=w)
        power_interp = s(self.k)
        
        if mu_term is not None:
            power_attr = getattr(power_attr, 'total')
            setattr(power_attr, mu_term, power_interp)
        else:
            setattr(self, power_name, power_interp)
            
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
class PowerTerm(object):
    """
    Class to hold the data for each term in the power expansion.
    """
    def __init__(self):
        
        # total angular dependences
        self.total = Angular()
        
        # initialize scalar/vector sub terms
        self.scalar = Angular()
        self.vector = Angular()
        
        # initialize with/without velocity dispersion sub terms
        self.no_velocity   = Angular()
        self.with_velocity = Angular()
        
#-------------------------------------------------------------------------------  
class Angular(object):
    """
    Class to keep track of the different angular terms for each power term.
    """
    def __init__(self):
        self.mu0 = None
        self.mu2 = None
        self.mu4 = None
        self.mu6 = None
        self.mu8 = None

#-------------------------------------------------------------------------------
class DMBaseSpectra(object):
    """
    Class to holding base power spectra to be used for splines in the 
    computation of PT integrals.
    """
    def __init__(self, z, kmin, kmax, cosmo, num_threads):
        self.z           = z
        self.kmin        = kmin
        self.kmax        = kmax
        self.cosmo       = cosmo
        self.num_threads = num_threads
    
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
    def I(self):
        try:
            return self._I
        except:
            self._I = integralsIJ.I_nm(0, 0, self.klin, self.Plin, k2=None, P2=None)
            return self._I
    #----------------------------------------------------------------------------
    @property
    def J(self):
        try:
            return self._J
        except:
            self._J = integralsIJ.J_nm(0, 0, self.klin, self.Plin)
            return self._J
    #---------------------------------------------------------------------------
    @property
    def klin(self):
        try:
            return self._klin
        except:
            # must have wide enough k region to converge
            kmin = self.kmin/(DMSpectrum.CONVERGENCE_FACTOR)**2
            kmax = self.kmax*(DMSpectrum.CONVERGENCE_FACTOR)**2
            
            self._klin = np.logspace(np.log10(kmin), np.log10(kmax), 1000)
            return self._klin
    #---------------------------------------------------------------------------
    @property
    def k1loop(self):
        try:
            return self._k1loop
        except:
            # must have wide enough k region to converge
            kmin = self.kmin/(DMSpectrum.CONVERGENCE_FACTOR)
            kmax = self.kmax*(DMSpectrum.CONVERGENCE_FACTOR)
            
            self._k1loop = np.logspace(np.log10(kmin), np.log10(kmax), 200)
            return self._k1loop
    #----------------------------------------------------------------------------
    @property
    def kmin_lin(self):
        return np.amin(self.klin)
        
    @property
    def kmax_lin(self):
        return np.amax(self.klin)
    #---------------------------------------------------------------------------
    @property
    def kmin_1loop(self):
        return np.amin(self.k1loop)
        
    @property
    def kmax_1loop(self):
        return np.amax(self.k1loop)
    #---------------------------------------------------------------------------
    @property
    def Plin(self):
        try:
            return self._Plin
        except:

            self._Plin = growth.Pk_lin(self.klin, 0., tf='EH', params=self.cosmo)
            return self._Plin 
    #---------------------------------------------------------------------------
    @property
    def _P11(self):
        try:
            return self.__P11
        except:

            self.__P11 = growth.Pk_lin(self.k1loop, 0., tf='EH', params=self.cosmo)
            return self.__P11
    #---------------------------------------------------------------------------
    @property
    def Pdd(self):
        """
        Auto-correlation of density at 1-loop.
        """
        try:
            return self._Pdd
        except:
            # compute I00
            self.I.n, self.I.m = 0, 0
            I00 = self.I.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)

            # compute J00
            self.J.n, self.J.m = 0, 0
            J00 = self.J.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin)

            P22 = 2.*I00
            P13 = 6.*self.k1loop**2*J00*self._P11
            self._Pdd = self.D**2*(self._P11 + self.D**2*(P22 + P13))
            
            return self._Pdd
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        """
        Cross-correlation of density and velocity divergence at 1-loop.
        """
        try:
            return self._Pdv
        except:       
            # compute I01
            self.I.n, self.I.m = 0, 1
            I01 = self.I.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)

            # compute J01
            self.J.n, self.J.m = 0, 1
            J01 = self.J.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin)

            P22  = 2.*I01
            P13  = 6.*self.k1loop**2*J01*self._P11
            self._Pdv = -self.f*self.D**2*(self._P11 + self.D**2*(P22 + P13))
            
            return self._Pdv
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        """
        Auto-correlation of velocity divergence at 1-loop.
        """
        try:
            return self._Pvv
        except:     
            # compute I11
            self.I.n, self.I.m = 1, 1
            I11 = self.I.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)

            # compute J11
            self.J.n, self.J.m = 1, 1
            J11 = self.J.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin)

            P22  = 2.*I11
            P13  = 6.*self.k1loop**2*J11*self._P11
            self._Pvv = (self.f*self.D)**2 * (self._P11 + self.D**2*(P22 + P13))
            
            return self._Pvv
    #---------------------------------------------------------------------------
    @property
    def P22bar_mu0(self):
        """
        The mu^0 contribution to the P22 with no velocity terms, at 1-loop.
        """
        try:
            return self._P22bar_mu0
        except:     
            # compute I23
            self.I.n, self.I.m = 2, 3
            I23 = self.I.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)
            
            A = 0.25*(self.f*self.D/self.k1loop)**4
            self._P22bar_mu0 = A*I23

            return self._P22bar_mu0
    #---------------------------------------------------------------------------
    @property
    def P22bar_mu2(self):
        """
        The mu^2 contribution to the P22 with no velocity terms, at 1-loop.
        """
        try:
            return self._P22bar_mu2
        except:    
            # compute I32
            self.I.n, self.I.m = 3, 2
            I32 = self.I.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)
            
            A = 0.5*(self.f*self.D/self.k1loop)**4
            self._P22bar_mu2 = A*I32

            return self._P22bar_mu2
    #---------------------------------------------------------------------------
    @property
    def P22bar_mu4(self):
        """
        The mu^4 contribution to the P22 with no velocity terms, at 1-loop.
        """
        try:
            return self._P22bar_mu4
        except:
            # compute I33
            self.I.n, self.I.m = 3, 3
            I33 = self.I.evaluate(self.k1loop, self.kmin_lin, self.kmax_lin, self.num_threads)
            
            A = 0.25*(self.f*self.D/self.k1loop)**4
            self._P22bar_mu4 = A*I33

            return self._P22bar_mu4
    #---------------------------------------------------------------------------
    @property
    def vel_kurtosis(self):
        """
        Compute the velocity kurtosis <v_parallel^4>, using the 1-loop divergence
        auto spectra Pvv.
        """
        try:
            return self._vel_kurtosis
        except:
            # evaluate the integral
            integrand = self.P22bar_mu0 + (1./3)*self.P22bar_mu2 + (1./5)*self.P22bar_mu4
            self._vel_kurtosis = intgr.simps(self.k1loop**3 * integrand, x=np.log(self.k1loop)) / (2*np.pi**2)
            return self._vel_kurtosis
    #---------------------------------------------------------------------------
    def convolve(self, k1, power1, k2, power2, k_eval):
        """
        Evaluate the convolution of two power spectra at the specified wavenumbers.
        """
        # get the actual data arrays from attribute names
        k1 = getattr(self, k1)
        k2 = getattr(self, k2)
        power1 = getattr(self, power1)
        power2 = getattr(self, power2)

        # determine the min, max k values
        kmin = max(np.min(k1), np.min(k2))
        kmax = min(np.max(k1), np.max(k2))

        I = integralsIJ.I_nm(-1, -1, k1, power1, k2=k2, P2=power2)
        return I.evaluate(k_eval, kmin, kmax, self.num_threads)
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------
        
        
