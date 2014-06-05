"""
 power_dm.pyx
 pyRSD: class implementing the redshift space dark matter power spectrum using
        the PT expansion outlined in Vlah et al. 2012.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
from pyRSD.cosmology cimport growth, cosmo_tools
from ..cosmology import velocity, hmf, power
from ..cosmology.cosmo import Cosmology
from . import _pt_integrals
 
import scipy.interpolate as interp
from scipy.integrate import quad
import numpy as np

class DMSpectrum(object):
    
    _power_atts = ['_P00', '_P01', '_P11', '_P02', '_P12', 'P22', '_P03', 
                    '_P13', '_P04', '_Pdd', '_Pdv', '_Pvv']
    
    def __init__(self, k=np.logspace(-2, np.log10(0.5), 100),
                       z=0., 
                       num_threads=1, 
                       cosmo={'default':"Planck1_lens_WP_highL", 'flat': True}, 
                       mass_function_kwargs={'mf_fit' : 'Tinker'}, 
                       bias_model='Tinker',
                       include_2loop=False,
                       transfer_fit="CAMB",
                       camb_kwargs={},
                       max_mu=4):
        """
        Parameters
        ----------
        k : array_like, optional
            The wavenumbers to compute the power spectrum at [units: h/Mpc]. 
            Must be between 1e-5 h/Mpc and 100 h/Mpc to be valid. 
            
        z : float
            The redshift to compute the power spectrum at.
            
        num_threads : int, optional
            The number of threads to use in parallel computation. Default = 1.
            
        cosmo : {dict, Cosmology}
            The cosmological parameters to use. Default is Planck DR1 + lensing
            + WP + high L 2013 parameters.
            
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
            
        transfer_fit : str, optional
            The name of the transfer function fit to use. Default is ``CAMB``
            and the options are {``CAMB``, ``EH``, ``EH_no_wiggles``, 
            ``EH_no_baryons``, ``BBKS``, ``Bond_Efs``}
        
        camb_kwargs : dict, optional
            Keyword arguments to pass to to ``Power`` object. 
        
        max_mu : {0, 2, 4, 6, 8}, optional
            Only compute angular terms up to mu**(``max_mu``). Default is 4.
        """
        if isinstance(cosmo, Cosmology):
            self._cosmo = cosmo
        else:
            self._cosmo     = Cosmology(**cosmo)
        self._max_mu        = max_mu
        self._num_threads   = num_threads
        self._include_2loop = include_2loop
        self.__k            = np.array(k, copy=False, ndmin=1)
        self._transfer_fit  = transfer_fit
        self._camb_kwargs   = camb_kwargs
        self._z             = z 
        
        # save useful quantitues for later
        self.mass_function_kwargs = mass_function_kwargs
        self.bias_model           = bias_model
        
    #end __init__
    #---------------------------------------------------------------------------
    def _delete_power(self):
        """
        Delete all power spectra attributes.
        """
        atts = [a for a in DMSpectrum._power_atts if a in self.__dict__]
        for a in atts:
            del self.__dict__[a]
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Update the parameters in an optimal manner.
        """
        cpdict = self._cosmo.dict()

        # first update the cosmology
        cp = {k:v for k, v in kwargs.iteritems() if k in Cosmology._cp}
        if cp:
            true_cp = {}
            for k, v in cp.iteritems():
                if k not in cpdict:
                    true_cp[k] = v
                elif k in cpdict:
                    if v != cpdict[k]:
                        true_cp[k] = v
                        
            # delete the entries we've used from kwargs
            for k in cp:
                del kwargs[k]
                
            # now actually update the classes depending on cosmology
            cpdict.update(true_cp)
            self._cosmo = Cosmology(**cpdict)
            self.integrals.update(**cpdict)
            self.power_lin.update(**cpdict)
            
            # delete everything 
            del self.f, self.D, self.conformalH, self.hmf
            self._delete_power()
                
        # now do any other parameters
        for key, val in kwargs.iteritems():
            if "_" + key not in self.__dict__:
                print "WARNING: %s is not a valid parameter for the %s class" %(str(key), self.__class__.__name__)
            else:
                if np.any(getattr(self, key) != val):
                    try:
                        setattr(self, key, val)
                    except:
                        setattr(self, '_' + key, val)
                
            # keywords affecting the transfer function
            if key in ['transfer_fit', 'camb_kwargs']:
                d = {key: val}
                if key == 'camb_kwargs':
                    d = val
                self.power_lin.update(**d)
                self.integrals.update(**d)
                
            # do redshift-dependent quantites
            elif key == 'z':
                del self.f, self.D, self.conformalH, self.hmf
                self.integrals.update(**{key:val})
                if hasattr(self, 'mass_function_kwargs'):
                    self.mass_function_kwargs['z'] = val
            elif key == 'num_threads':
                self._integrals.update(**{key:val})

        # delete the power attributes, if any of these changed
        attrbs = ['transfer_fit', 'camb_kwargs', 'z', 'max_mu', 'include_2loop']
        if any(key in attrbs for key in kwargs):
            self._delete_power()
    #---------------------------------------------------------------------------
    # READ-ONLY ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def k(self):
        return self.__k

    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self._z
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        return self._cosmo
    #---------------------------------------------------------------------------
    @property
    def transfer_fit(self):
        return self._transfer_fit

    #---------------------------------------------------------------------------
    @property
    def camb_kwargs(self):
        return self._camb_kwargs
    #---------------------------------------------------------------------------
    @property
    def max_mu(self):
        return self._max_mu
    #---------------------------------------------------------------------------
    @property
    def include_2loop(self):
        return self._include_2loop
    #---------------------------------------------------------------------------
    @property
    def D(self):
        try:
            return self._D
        except AttributeError:
            self._D = growth.growth_function(self.z, normed=True, params=self.cosmo)
            return self._D

    @D.deleter
    def D(self):
        try:
            del self._D
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def f(self):
        try:
            return self._f
        except AttributeError:
            self._f = growth.growth_rate(self.z, params=self.cosmo)
            return self._f

    @f.deleter
    def f(self):
        try:
            del self._f
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def conformalH(self):
        try:
            return self._conformalH
        except AttributeError:
            self._conformalH = cosmo_tools.H(self.z,  params=self.cosmo)/(1.+self.z)
            return self._conformalH

    @conformalH.deleter
    def conformalH(self):
        try:
            del self._conformalH
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def hmf(self):
        try:
            return self._hmf
        except AttributeError:
            self._hmf = hmf.HaloMassFunction(cosmo=self.cosmo, **self.mass_function_kwargs)
            return self._hmf

    @hmf.deleter
    def hmf(self):
        try:
            del self._hmf
            del self.sigma_v2
            del self.sigma_bv2
            del self.sigma_bv4
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def integrals(self):
        try:
            return self._integrals
        except AttributeError:
            self._integrals = _pt_integrals.Integrals(self.k, self.z, self.cosmo, 
                                                    self._num_threads, self.transfer_fit, 
                                                    self.camb_kwargs)
            return self._integrals
    #---------------------------------------------------------------------------
    @property
    def power_lin(self):
        try:
            return self._power_lin
        except AttributeError:
            self._power_lin = power.Power(k=self.k, z=0., transfer_fit=self.transfer_fit, 
                                            cosmo=self.cosmo, **self.camb_kwargs)
            return self._power_lin
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
            return np.sqrt(self.integrals.sigmasq)
        
    @sigma_lin.setter
    def sigma_lin(self, val):
        self._sigma_lin = val
    #---------------------------------------------------------------------------
    # SET ATTRIBUTES
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
    def mass_function_kwargs(self):
        return self._mass_function_kwargs
    
    @mass_function_kwargs.setter
    def mass_function_kwargs(self, val):
        self._mass_function_kwargs = val
        self._mass_function_kwargs['z'] = self.z
        
        del self.hmf
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
        except AttributeError:
            self._sigma_v2 = velocity.sigma_v2(self.hmf)
            return self._sigma_v2
    
    @sigma_v2.setter
    def sigma_v2(self, val):
        self._sigma_v2 = val
        
        atts = ['_P01', '_P03']
        for a in atts:
            if a in self.__dict__: del self.__dict__[a]
    
    @sigma_v2.deleter
    def sigma_v2(self):
        try:
            del self._sigma_v2
        except AttributeError:
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
        except AttributeError:
            self._sigma_bv2 = velocity.sigma_bv2(self.hmf, self.bias_model)
            return self._sigma_bv2
    
    @sigma_bv2.setter
    def sigma_bv2(self, val):
        self._sigma_bv2 = val
        
        atts = ['_P02', '_P12', '_P13', '_P22']
        for a in atts:
            if a in self.__dict__: del self.__dict__[a]
    
    @sigma_bv2.deleter
    def sigma_bv2(self):
        try:
            del self._sigma_bv2
        except AttributeError:
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
        except AttributeError:
            self._sigma_bv4 = velocity.sigma_bv4(self.hmf, self.bias_model)
            return self._sigma_bv4
            
    @sigma_bv4.setter
    def sigma_bv4(self, val):
        self._sigma_bv4 = val
        
        if hasattr(self, '_P04'): del self.__dict__['_P04']
    
    @sigma_bv4.deleter
    def sigma_bv4(self):
        try:
            del self._sigma_bv4
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    @property
    def P_mu0(self):
        """
        The full power spectrum term with no angular dependence. Contributions
        from P00.
        """
        return self.P00.total.mu0
    #---------------------------------------------------------------------------
    @property
    def P_mu2(self):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        return self.P01.total.mu2 + self.P11.total.mu2 + self.P02.total.mu2
    #---------------------------------------------------------------------------
    @property
    def P_mu4(self):
        """
        The full power spectrum term with mu^4 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13 (2-loop), and P04 (2-loop).
        """
        P_mu4 = self.P11.total.mu4 + self.P02.total.mu4 + self.P12.total.mu4 + \
                    self.P22.total.mu4 + self.P03.total.mu4 
        if self.include_2loop:
            P_mu4 += self.P13.total.mu4 + self.P04.total.mu4
        return P_mu4
    #---------------------------------------------------------------------------
    @property
    def P_mu6(self):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P12, P22, P13, and P04 (2-loop).
        """
        P_mu6 = self.P12.total.mu6 + self.P22.total.mu6 + self.P13.total.mu6
        if self.include_2loop:
            P_mu6 += self.P04.total.mu6
        return P_mu6
    #---------------------------------------------------------------------------
    @property
    def _Pdd_1loop(self):
        """
        The 1-loop component of the auto-correlation of density
        """
        I00 = self.integrals.I('f00', 0)
        J00 = self.integrals.J('g00')
        
        P22  = 2.*I00
        P13  = 6.*self.k**2*J00*(self.D**2 * self.power_lin.power)
        return P22 + P13
    #---------------------------------------------------------------------------
    @property
    def _Pdv_1loop(self):
        """
        The 1-loop component of the cross-correlation of density and velocity 
        divergence.
        """
        I01 = self.integrals.I('f01', 0)
        J01 = self.integrals.J('g01')
        
        P22  = 2.*I01
        P13  = 6.*self.k**2*J01*(self.D**2 * self.power_lin.power)
        return P22 + P13
    #---------------------------------------------------------------------------
    @property
    def _Pvv_1loop(self):
        """
        The 1-loop component of the auto-correlation of velocity divergence.
        """
        I11 = self.integrals.I('f11', 0)
        J11 = self.integrals.J('g11')
        
        P22  = 2.*I11
        P13  = 6.*self.k**2*J11*(self.D**2 * self.power_lin.power)
        return P22 + P13
    #---------------------------------------------------------------------------
    @property
    def Pdd(self):
        """
        The 1-loop auto-correlation of density.
        """
        try:
            return self._Pdd
        except AttributeError:
            self._Pdd = (self.D**2*self.power_lin.power + self._Pdd_1loop)
            return self._Pdd
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        """
        The 1-loop cross-correlation between dark matter density and velocity 
        divergence.
        """
        try:
            return self._Pdv
        except AttributeError:
            
            # check for any user-loaded values
            if hasattr(self, '_Pdv_loaded'):
                self._Pdv = self._Pdv_loaded
            else:
                self._Pdv = (-self.f) * (self.D**2*self.power_lin.power + self._Pdv_1loop)
            return self._Pdv
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        """
        The 1-loop auto-correlation of velocity divergence.
        """
        try:
            return self._Pvv
        except AttributeError:
            self._Pvv = (self.f**2) * (self.D**2*self.power_lin.power + self._Pvv_1loop)
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
        except AttributeError:
            self._P00 = PowerTerm()
            
            # check and return any user-loaded values
            if hasattr(self, '_P00_mu0_loaded'):
                self._P00.total.mu0 = self._P00_mu0_loaded
            else:
                # the necessary integrals 
                I00 = self.integrals.I('f00', 0)
                J00 = self.integrals.J('g00')
            
                P11 = self.D**2 * self.power_lin.power
                P22 = 2*I00
                P13 = 6*self.k**2*J00*P11
                self._P00.total.mu0 = P11 + P22 + P13
            
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
        except AttributeError:
            self._P01 = PowerTerm()
            
            # check and return any user-loaded values
            if hasattr(self, '_P01_mu2_loaded'):
                self._P01.total.mu2 = self._P01_mu2_loaded
            else:
                
                # the necessary integrals 
                I00 = self.integrals.I('f00', 0)
                J00 = self.integrals.J('g00')
            
                Plin = self.D**2 * self.power_lin.power
                self._P01.total.mu2 = 2*self.f*(Plin + 4.*(I00 + 3*self.k**2*J00*Plin))
            
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
        except AttributeError:
            self._P11 = PowerTerm()
            
            # do mu^2 terms?
            if self.max_mu >= 2:
                
                # check and return any user-loaded values
                if hasattr(self, '_P11_mu2_loaded'):
                    Pvec = self._P11_mu2_loaded
                    self._P11.vector.mu2 = self._P11.total.mu2 = Pvec
                    self._P11.vector.mu4 = self._P11.vector.mu4 = -Pvec
                else:
                    
                    # do the vector part, contributing mu^2 and mu^4 terms
                    if not self.include_2loop:
                        Pvec = self.f**2 * self.integrals.I('f31', 0)
                    else:
                        I1 = self.integrals.I('h01', 1, ('dd', 'vv'))
                        I2 = self.integrals.I('h03', 1, ('dv', 'dv'))
                        Pvec = self.f**2 * (I1 + I2)
                
                    # save the mu^2 vector term
                    self._P11.vector.mu2 = self._P11.total.mu2 = Pvec
                    self._P11.vector.mu4 = self._P11.vector.mu4 = -Pvec
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                      
                    # check and return any user-loaded values
                    if hasattr(self, '_P11_mu4_loaded'):
                        self._P11.total.mu4 = self._P11_mu4_loaded
                    else:
                          
                        # compute the scalar mu^4 contribution
                        if self.include_2loop:
                            I1 = self.integrals.I('h02', 1, ('dd', 'vv'))
                            I2 = self.integrals.I('h04', 1, ('dv', 'dv'))
                            C11_contrib = I1 + I2
                        else:
                            C11_contrib = self.integrals.I('f13', 0)
                    
                        # the necessary integrals 
                        I11 = self.integrals.I('f11', 0)
                        I22 = self.integrals.I('f22', 0)
                        J11 = self.integrals.J('g11')
                        J10 = self.integrals.J('g10')
                    
                        Plin = self.D**2 * self.power_lin.power
                        part2 = 2*I11 + 4*I22 + 6*self.k**2 * (J11 + 2*J10)*Plin
                        P_scalar = self.f**2 * (Plin + part2 + C11_contrib) - self._P11.vector.mu4
                    
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
        except AttributeError:
            self._P02 = PowerTerm()
            
            # do mu^2 terms?
            if self.max_mu >= 2:        
                Plin = self.D**2 * self.power_lin.power
                
                # the necessary integrals 
                I02 = self.integrals.I('f02', 0)
                J02 = self.integrals.J('g02')
    
                # the nmu^2 no velocity terms
                self._P02.no_velocity.mu2 = self.f**2 * (I02 + 2*self.k**2*J02*Plin)
                
                # the mu^2 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_lin
                sigma_02  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D)
                sigsq_eff = sigma_lin**2 + sigma_02**2

                if self.include_2loop:
                    self._P02.with_velocity.mu2 = -(self.f*self.D*self.k)**2 * sigsq_eff*self.P00.total.mu0
                else:
                    self._P02.with_velocity.mu2 = -(self.f*self.D*self.k)**2 * sigsq_eff*Plin
            
                # save the total mu^2 term
                self._P02.total.mu2 = self._P02.with_velocity.mu2 + self._P02.no_velocity.mu2
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                    
                    # the necessary integrals 
                    I20 = self.integrals.I('f20', 0)
                    J20 = self.integrals.J('g20')
                    self._P02.total.mu4 = self._P02.no_velocity.mu4 = self.f**2 * (I20 + 2*self.k**2*J20*Plin)
                    
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
        except AttributeError:
            self._P12 = PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                Plin = self.D**2 * self.power_lin.power
                
                # the necessary integrals 
                I12 = self.integrals.I('f12', 0)
                I03 = self.integrals.I('f03', 0)
                J02 = self.integrals.J('g02')
                
                # do the mu^4 terms that don't depend on velocity
                self._P12.no_velocity.mu4 = self.f**3 * (I12 - I03 + 2*self.k**2*J02*Plin)
            
                # now do mu^4 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_lin  
                sigma_12  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_12**2
            
                if self.include_2loop:
                    self._P12.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff*self.P01.total.mu2
                else:
                    self._P12.with_velocity.mu4 = -self.f*(self.f*self.D*self.k)**2 * sigsq_eff*Plin
            
                # total mu^4 is velocity + no velocity terms
                self._P12.total.mu4 = self._P12.with_velocity.mu4 + self._P12.no_velocity.mu4
                
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # the necessary integrals 
                    I21 = self.integrals.I('f21', 0)
                    I30 = self.integrals.I('f30', 0)
                    J20 = self.integrals.J('g20')
                    
                    self._P12.no_velocity.mu6 = self.f**3 * (I21 - I30 + 2*self.k**2*J20*Plin)
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
        except AttributeError:
            self._P22 = PowerTerm()
            
            # velocity terms come in at 2-loop here
            if self.include_2loop:
                
                # velocities in units of Mpc/h
                sigma_lin = self.sigma_lin
                sigma_22  = self.sigma_bv2 * self.cosmo.h / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_22**2
                
                J02 = self.integrals.J('g02')
                J20 = self.integrals.J('g20')
                
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                Plin = self.D**2 * self.power_lin.power
                
                # 1-loop or 2-loop terms from <v^2 | v^2 > 
                if not self.include_2loop:
                    self._P22.no_velocity.mu4 = 1./16*self.f**4 * self.integrals.I('f23', 0)
                else:
                    I23_2loop = self.integrals.I('f23', 1, ('vv', 'vv'))
                    self._P22.no_velocity.mu4 = 1./16*self.f**4 * I23_2loop

                # now add in the extra 2 loop terms, if specified
                if self.include_2loop:
                               
                    # one more 2-loop term for <v^2 | v^2>
                    extra_vv_mu4 = (self.f*self.k)**4 * Plin*J02**2
                    
                    # term from <v^2 | d v^2>
                    extra_vdv_mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                    
                    # 1st term coming from <dv^2 | dv^2>
                    extra1_dvdv_mu4 = 0.25*(self.f*self.D*self.k)**4 * sigsq_eff**2 * self.P00.total.mu0
                                        
                    # 2nd term from <dv^2 | dv^2> is convolution of P22_bar and P00
                    extra2_dvdv_mu4 = 0.5*(self.f*self.k)**4 * self.P00.total.mu0*self.integrals.sigmasq_k**2
                    
                    # store the extra two loop terms
                    extra = extra_vv_mu4 + extra_vdv_mu4 + extra1_dvdv_mu4 + extra2_dvdv_mu4
                    self._P22.total.mu4 = self._P22.no_velocity.mu4 + extra
                    
                else:
                    self._P22.total.mu4 = self._P22.no_velocity.mu4
                    
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # 1-loop or 2-loop terms that don't depend on velocity
                    if not self.include_2loop:
                        self._P22.no_velocity.mu6 = 1./8*self.f**4 * self.integrals.I('f32', 0)
                    else:
                        I32_2loop = self.integrals.I('f32', 1, ('vv', 'vv'))
                        self._P22.no_velocity.mu6 = 1./8*self.f**4 * I32_2loop
                        
                    # now add in the extra 2 loop terms, if specified
                    if self.include_2loop:

                        # term from <v^2 | d v^2>
                        extra_vdv_mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                        
                        # one more 2-loop term for <v^2 | v^2>
                        extra_vv_mu6  = 2*(self.f*self.k)**4 * Plin*J02*J20
                        
                        # save the totals
                        extra = extra_vv_mu6 + extra_vdv_mu6 
                        self._P22.total.mu6 = self._P22.no_velocity.mu6 + extra
                        
                    else:
                        self._P22.total.mu6 = self._P22.no_velocity.mu6

                    # do mu^8 terms?
                    if self.max_mu >= 8:
                        
                        # 1-loop or 2-loop terms that don't depend on velocity
                        if not self.include_2loop:
                            self._P22.no_velocity.mu8 = 1./16*self.f**4 * self.integrals.I('f33', 0)
                        else:
                            I33_2loop = self.integrals.I('f33', 1, ('vv', 'vv'))
                            self._P22.no_velocity.mu8 = 1./16*self.f**4 * I33_2loop
                            
                            # extra 2 loop term from modeling <v^2|v^2>
                            self._P22.no_velocity.mu8 += (self.f*self.k)**4 * Plin*J20**2
                            
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
        except AttributeError:
            self._P03 = PowerTerm()
            
            # do mu^4 terms?
            if self.max_mu >= 4:
                Plin = self.D**2 * self.power_lin.power
                
                # only terms depending on velocity here (velocities in Mpc/h)
                sigma_lin = self.sigma_lin 
                sigma_03  = self.sigma_v2 * self.cosmo.h / (self.f*self.conformalH*self.D)
                sigsq_eff = sigma_lin**2 + sigma_03**2

                # either 1 or 2 loop quantities
                if self.include_2loop:
                    self._P03.with_velocity.mu4 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P01.total.mu2
                else:
                    self._P03.with_velocity.mu4 = -self.f*(self.f*self.D*self.k)**2 *sigsq_eff*Plin
            
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
        except AttributeError:
            self._P13 = PowerTerm()
            Plin = self.D**2 * self.power_lin.power
            
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
                        self._P13.with_velocity.mu6 = -self.f**2 *(self.f*self.D*self.k)**2 * sigsq_eff_scalar*Plin
                        
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
        except AttributeError:
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
                    P04_vel_mu4_2 = 0.25*(self.f*self.k)**4 * (self.D**2*sigsq_eff)**2 * self.P00.total.mu0
                    self.P04.with_velocity.mu4 = P04_vel_mu4_1 + P04_vel_mu4_2
                    
                    # do P04 mu^4 terms without vel dependence
                    self.P04.no_velocity.mu4 = 1./12.*(self.f*self.k)**4 * self.P00.total.mu0*self.integrals.vel_kurtosis
                
                    # save the total
                    self.P04.total.mu4 = self.P04.with_velocity.mu4 + self.P04.no_velocity.mu4
                
                    # do mu^6 terms?
                    if self.max_mu >= 6:
                        
                        # only terms depending on velocity
                        self.P04.with_velocity.mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                        self.P04.total.mu6 = self.P04.with_velocity.mu6
                        
            return self._P04
    #---------------------------------------------------------------------------
    def power(self, mu, mu_hi=None):
        """
        Return the redshift space power spectrum at the specified value of mu, 
        including terms up to mu**max_mu
        """
        if mu_hi is not None:
            assert(mu_hi > mu)
        
        if self.max_mu == 0:
            return self.P_mu0
            
        elif self.max_mu == 2:
            mu2 = mu**2
            if mu_hi is not None:
                mu2 = self._mu_avg(mu, mu_hi, 2)
            return self.P_mu0 + mu2 * self.P_mu2
            
        elif self.max_mu == 4:
            mu2, mu4 = mu**2, mu**4
            if mu_hi is not None:
                mu2 = self._mu_avg(mu, mu_hi, 2)
                mu4 = self._mu_avg(mu, mu_hi, 4)
            return self.P_mu0 + mu2 * self.P_mu2 + mu4 * self.P_mu4
            
        elif self.max_mu == 6:
            mu2, mu4, mu6 = mu**2, mu**4, mu**6
            if mu_hi is not None:
                mu2 = self._mu_avg(mu, mu_hi, 2)
                mu4 = self._mu_avg(mu, mu_hi, 4)
                mu4 = self._mu_avg(mu, mu_hi, 6)
            return self.P_mu0 + mu2 * self.P_mu2 + mu4 * self.P_mu4 + mu6 * self.P_mu6 
            
        elif self.max_mu == 8:
            raise NotImplementedError("Cannot compute power spectrum including terms with order higher than mu^6")
    #end power
    #---------------------------------------------------------------------------
    def _mu_avg(self, mu_lo, mu_hi, power):
        """
        Compute the mean value of ``mu**power`` over the specified mu range
        """
        return quad(lambda x: x**power, mu_lo, mu_hi)[0] / (mu_hi - mu_lo)
    #---------------------------------------------------------------------------
    def monopole(self, linear=False):
        """
        The monopole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        if linear:
            beta = self.f/self.b1
            return (1. + 2./3*beta + 1/5*beta**2) * (self.b1*self.D)**2 * self.power_lin.power
        else:
            if self.max_mu == 0:
                return self.P_mu0
            elif self.max_mu == 2:
                return self.P_mu0 + (1./3)*self.P_mu2
            elif self.max_mu == 4:
                return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4
            elif self.max_mu == 6:
                return self.P_mu0 + (1./3)*self.P_mu2 + (1./5)*self.P_mu4 + (1./7)*self.P_mu6
            elif self.max_mu == 8:
                raise NotImplementedError("Cannot compute monopole including terms with order higher than mu^6")
    #end monopole
    #---------------------------------------------------------------------------
    def quadrupole(self, linear=False):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        if linear:
            beta = self.f/self.b1
            return (4./3*beta + 4./7*beta**2) * (self.b1*self.D)**2 * self.power_lin.power
        else:
            if self.max_mu == 2:
                return (2./3)*self.P_mu2
            elif self.max_mu == 4:
                return (2./3)*self.P_mu2 + (4./7)*self.P_mu4
            elif self.max_mu == 6:
                return (2./3)*self.P_mu2 + (4./7)*self.P_mu4 + (10./21)*self.P_mu6
            elif self.max_mu == 8:
                raise NotImplementedError("Cannot compute monopole including terms with order higher than mu^6")
    #end quadrupole
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
            del self.__dict__[power_name]
        
        if errs is not None:
            w = 1./np.array(errs)
        else:
            w = None
        s = interp.InterpolatedUnivariateSpline(k_data, power_data, w=w)
        power_interp = s(self.k)
        
        if mu_term is not None:
            setattr(self, "%s_%s_loaded" %(power_name, mu_term), power_interp)
        else:
            setattr(self, "%s_loaded" %power_name, power_interp)
            
    #end load
    #---------------------------------------------------------------------------
    def unload(self, power_term, mu_term):
        """
        Delete the given power attribute, as specified by power_term.
        """
        power_name = "_%s" %power_term
        
        # first delete the power term
        if hasattr(self, power_name):
            del self.__dict__[power_name]
            
        # also delete the z = 0 stored variabels
        if mu_term is not None:
            try:
                del self.__dict__["%s_%s" %(power_name, mu_term)]
            except AttributeError:
                pass
        else:
            try:
                del self.__dict__["%s" %power_name]
            except AttributeError:
                pass
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
        self.mu0 = 0.
        self.mu2 = 0.
        self.mu4 = 0.
        self.mu6 = 0.
        self.mu8 = 0.

#-------------------------------------------------------------------------------

        
        
