"""
 power_dm.py
 pyRSD: class implementing the redshift space dark matter power spectrum using
        the PT expansion outlined in Vlah et al. 2012.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/17/2014
"""
from .. import pygcl, data_dir, os, numpy as np
from . import integrals, dm_power_moments, tools
 
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import quad

class DMSpectrum(object):
    
    allowable_kwargs = ['k', 'z', 'cosmo', 'include_2loop', 'transfer_fit', 'max_mu', 'DM_model']
    _power_atts = ['_P00', '_P01', '_P11', '_P02', '_P12', '_P22', '_P03', '_P13', '_P04']
    
    def __init__(self, k=np.logspace(-2, np.log10(0.5), 100),
                       z=0., 
                       cosmo="planck1_WP.ini",
                       include_2loop=False,
                       transfer_fit="CLASS",
                       max_mu=4, 
                       DM_model='A'):
        """
        Parameters
        ----------
        k : array_like, optional
            The wavenumbers to compute power spectrum at [units: `h/Mpc`]
            
        z : float, optional
            The redshift to compute the power spectrum at. Default = 0.
            
        cosmo : {str, pygcl.Cosmology}
            The cosmological parameters to use, specified as either the name
            of the file holding the `CLASS` parameter file, or a `pygcl.Cosmology`
            object. Default is `planck1_WP.ini`.
            
        include_2loop : bool, optional
            If `True`, include 2-loop contributions in the model terms. Default
            is `False`.
            
        transfer_fit : str, optional
            The name of the transfer function fit to use. Default is `CLASS`
            and the options are {`CLASS`, `EH`, `EH_NoWiggle`, `BBKS`}, 
            or the name of a data file holding (k, T(k))
        
        max_mu : {0, 2, 4, 6, 8}, optional
            Only compute angular terms up to mu**(``max_mu``). Default is 4.
        
        DM_model : {`A`, `B`, None}, optional
            Use the specified dark matter model for P00, P01 terms
        """
        # determine the type of transfer fit
        self.transfer_fit = transfer_fit
        
        # initialize the pygcl.Cosmology object
        if isinstance(cosmo, pygcl.Cosmology):
            self._cosmo = cosmo
        else:
            if self.transfer_file is None:
                self._cosmo = pygcl.Cosmology(cosmo, self.transfer_fit)
            else:
                self._cosmo = pygcl.Cosmology(cosmo, self.transfer_fit, self.transfer_file)
        
        # store the input parameters
        self._max_mu        = max_mu
        self._include_2loop = include_2loop
        self._transfer_fit  = transfer_fit
        self._z             = z 
        self._k_obs         = k
        
        # set sigma8 to its initial value for this cosmology
        self.sigma8 = self.cosmo.sigma8()

        # set the DM model term
        self.DM_model = DM_model
        
    #end __init__
    
    #---------------------------------------------------------------------------
    # UTILITY FUNCTIONS
    #---------------------------------------------------------------------------
    def k_true(self, k_obs, mu_obs):
        """
        Return the `true` k values, given an observed (k, mu)
        """
        F = self.alpha_par / self.alpha_perp
        if (F != 1.):
            return (k_obs/self.alpha_perp)*(1 + mu_obs**2*(1./F**2 - 1))**0.5
        else:
            return k_obs/self.alpha_perp
            
    #end k_true
    
    #---------------------------------------------------------------------------
    def mu_true(self, mu_obs):
        """
        Return the `true` mu values, given an observed mu
        """
        F = self.alpha_par / self.alpha_perp
        return (mu_obs/F) * (1 + mu_obs**2*(1./F**2 - 1))**(-0.5)
            
    #end mu_true
    #-------------------------------------------------------------------------------
    def update(self, **kwargs):
        for k, v in kwargs.iteritems():
            if hasattr(self, k): 
                if getattr(self, k) != v: setattr(self, k, v)
    
    #end update
    #---------------------------------------------------------------------------
    def _delete_power(self):
        """
        Delete all power spectra attributes.
        """
        # delete the power attributes
        for a in DMSpectrum._power_atts:
            if hasattr(self, a): delattr(self, a)
            
        # also delete the P_mu* splines
        for a in ['_P_mu0_spline', '_P_mu2_spline', '_P_mu4_spline', '_P_mu6_spline']:
            if hasattr(self, a): delattr(self, a)
            
    #end _delete_power
    #---------------------------------------------------------------------------
    # INPUT ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def k_obs(self):
        """
        The "observed" wavenumbers to compute the power spectrum at. This should
        be read-only.
        """
        return self._k_obs
    
    #---------------------------------------------------------------------------
    @property
    def mu_obs(self):
        """
        The "observed" mu to compute the power spectrum at. This can be set, and
        note that the "true" wavenumbers depend on this value
        """
        try:
            return self._mu_obs           
        except AttributeError:
            raise AttributeError("need to set `mu_obs` first")
        
    @mu_obs.setter
    def mu_obs(self, val):
        self._mu_obs = val
        
        # delete dependencies
        del self.mu
        
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        """
        The `pygcl.Cosmology` object holding the cosmology parameters. This is
        really a wrapper for a `CLASS` parameter file
        """
        return self._cosmo
        
    #---------------------------------------------------------------------------
    @property
    def max_mu(self):
        """
        Only compute the power terms up to and including `max_mu`
        """
        return self._max_mu
        
    @max_mu.setter
    def max_mu(self, val):        
        self._max_mu = val
        self._delete_power()
        
    #---------------------------------------------------------------------------
    @property
    def include_2loop(self):
        """
        Whether to include 2-loop terms in the power spectrum calculation
        """
        return self._include_2loop
        
    @include_2loop.setter
    def include_2loop(self, val):        
        self._include_2loop = val
        self._delete_power()
        
    #---------------------------------------------------------------------------
    @property
    def DM_model(self):
        """
        The type of dark matter model to use for P00/P01
        """
        return self._DM_model
        
    @DM_model.setter
    def DM_model(self, val):
                
        # check input value
        if val is not None and val not in ['A', 'B']:
            raise ValueError("`DM_model` must be `None` or one of ['A', 'B']")
        
        self._DM_model = val
        if hasattr(self, '_P00_model'): self.P00_model.model_type = val
        if hasattr(self, '_P01_model'): self.P01_model.model_type = val
        
        # delete old power attributes
        if hasattr(self, '_P00'): delattr(self, '_P00')
        if hasattr(self, '_P01'): delattr(self, '_P01')
        
    #---------------------------------------------------------------------------
    @property
    def transfer_fit(self):
        """
        The transfer function fitting method
        """
        return self._transfer_fit
        
    @transfer_fit.setter
    def transfer_fit(self, val):

        # delete old transfer file
        del self.transfer_file
        
        # set the new transfer fit
        if val in ['CLASS', 'EH', 'EH_NoWiggle', 'BBKS']:
            self._transfer_fit = getattr(pygcl.Cosmology, val)
        else:
            self._transfer_fit = 'FromFile'
            self.transfer_file = val
        
    @property
    def transfer_file(self):
        """
        The name of the data file holding the transfer function
        """
        try:
            return self._transfer_file
        except AttributeError:
            return None

    @transfer_file.setter
    def transfer_file(self, val):
        
        if os.path.exists(val):
            self._transfer_file = val
        elif (os.path.exists("%s/%s" %(data_dir, val))):
            self._transfer_file = "%s/%s" %(data_dir, val)
        else:
            raise ValueError("Could not find transfer data file '%s', tried ./%s and %s/%s" %(val, val, data_dir, val))
        
    @transfer_file.deleter
    def transfer_file(self):
        try:
            del self._transfer_fit
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    # SET PARAMETERS
    #---------------------------------------------------------------------------
    @property
    def k(self):
        """
        Return the true k values, given the current AP rescaling factors and
        the observed (k, mu) values, stored in `self.k_obs` and `self.mu_obs`
        """
        try:
            return self._k
        except AttributeError:
            k_mu0 = self.k_true(self.k_obs, 0.)
            k_mu1 = self.k_true(self.k_obs, 1.)
            kmin = min(np.amin(k_mu0), np.amin(k_mu1))
            kmax = max(np.amax(k_mu0), np.amax(k_mu1))
            
            self._k = np.logspace(np.log10(kmin), np.log10(kmax), 500)
            return self._k
            
    @k.deleter
    def k(self):
        try:
            del self._k
        except AttributeError:
            pass
        self._delete_power()
        
    #---------------------------------------------------------------------------
    @property
    def z(self):
        """
        Redshift to evaluate power spectrum at
        """
        return self._z
    
    @z.setter
    def z(self, val):
        if hasattr(self, '_z') and val == self._z: 
            return
            
        self._z = val
        if hasattr(self, '_integrals'): self.integrals.z = val
        if hasattr(self, '_P00_model'): self.P00_model.z = val
        if hasattr(self, '_P01_model'): self.P01_model.z = val
        del self.D, self.conformalH
        self._delete_power()

    #---------------------------------------------------------------------------
    @property
    def sigma8(self):
        """
        Sigma_8 to compute the power spectrum at, which gives the normalization 
        of the linear power spectrum
        """
        return self._sigma8

    @sigma8.setter
    def sigma8(self, val):
        if hasattr(self, '_sigma8') and val == self._sigma8: 
            return
            
        self._sigma8 = val
        if hasattr(self, '_integrals'): self.integrals.sigma8 = val
        if hasattr(self, '_P00_model'): self.P00_model.sigma8 = val
        if hasattr(self, '_P01_model'): self.P01_model.sigma8 = val
        self._delete_power()

    #---------------------------------------------------------------------------
    @property
    def f(self):
        """
        The growth rate, defined as the `dlnD/dlna`. 
        
        If the parameter has not been explicity set, it defaults to the value
        at `self.z`
        """
        try:
            return self._f
        except AttributeError:
            return self.cosmo.f_z(self.z)

    @f.setter
    def f(self, val):
        if hasattr(self, '_f') and val == self._f: 
            return
            
        self._f = val
        if hasattr(self, '_P01_model'): self.P01_model.f = val
        self._delete_power()
    
    #---------------------------------------------------------------------------
    @property
    def alpha_perp(self):
        """
        The perpendicular Alcock-Paczynski effect scaling parameter, where
        :math: `k_{perp, true} = k_{perp, true} / alpha_{perp}`
        
        If the parameter has not been explicity set, it defaults to unity
        """
        try: 
            return self._alpha_perp
        except AttributeError:
            return 1.  
        
    @alpha_perp.setter
    def alpha_perp(self, val):
        if hasattr(self, '_alpha_perp') and val == self._alpha_perp: 
            return
            
        self._alpha_perp = val
        del self.k
        
    #---------------------------------------------------------------------------
    @property
    def alpha_par(self):
        """
        The parallel Alcock-Paczynski effect scaling parameter, where
        :math: `k_{par, true} = k_{par, true} / alpha_{par}`
        
        If the parameter has not been explicity set, it defaults to unity
        """
        try: 
            return self._alpha_par
        except AttributeError:
            return 1.  
        
    @alpha_par.setter
    def alpha_par(self, val):
        if hasattr(self, '_alpha_par') and val == self._alpha_par: 
            return
            
        self._alpha_par = val
        del self.k
    
    #---------------------------------------------------------------------------
    # DERIVED ATTRIBUTES
    #---------------------------------------------------------------------------    
    @property
    def D(self):
        """
        The growth function, normalized to 1 at z = 0
        """
        try:
            return self._D
        except AttributeError:
            self._D = self.cosmo.D_z(self.z)
            return self._D

    @D.deleter
    def D(self):
        try:
            del self._D
        except AttributeError:
            pass
        
    #---------------------------------------------------------------------------
    @property
    def conformalH(self):
        """
        The conformal Hubble parameter, defined as `H(z) / (1 + z)`
        """
        try:
            return self._conformalH
        except AttributeError:
            self._conformalH = self.cosmo.H_z(self.z) / (1. + self.z)
            return self._conformalH

    @conformalH.deleter
    def conformalH(self):
        try:
            del self._conformalH
        except AttributeError:
            pass
    
    #---------------------------------------------------------------------------
    def normed_power_lin(self, k):
        """
        The linear power evaluated at `self.k` and at `self.z`, normalized
        to `self.sigma8`
        """
        return self._power_norm * self.D**2 * self.power_lin(k)
    
    #---------------------------------------------------------------------------
    @property
    def _power_norm(self):
        """
        The factor needed to normalize the linear power spectrum to the 
        desired sigma_8, as specified by `self.sigma8`
        """
        return (self.sigma8 / self.cosmo.sigma8())**2
        
    #---------------------------------------------------------------------------
    @property
    def power_lin(self):
        """
        A 'pygcl.LinearPS' object holding the linear power spectrum at z = 0
        """
        try:
            return self._power_lin
        except AttributeError:
            self._power_lin = pygcl.LinearPS(self.cosmo, 0.)
            return self._power_lin
    
    #---------------------------------------------------------------------------
    @property
    def sigma_lin(self):
        """
        The dark matter velocity dispersion at z = 0, as evaluated in 
        linear theory [units: Mpc/h]
        """
        try:
            return self._sigma_lin
        except:
            self._sigma_lin = np.sqrt(self.power_lin.VelocityDispersion())
            return self._sigma_lin
    
    #---------------------------------------------------------------------------
    @property
    def sigma_v(self):
        """
        The velocity dispersion at z = 0. If not provided, defaults to the 
        linear theory prediction (as given by `self.sigma_lin`) [units: Mpc/h]
        """
        try: 
            return self._sigma_v
        except AttributeError:
            return self.sigma_lin
        
    @sigma_v.setter
    def sigma_v(self, val):
        self._sigma_v = val
        
        # delete power terms that dependend on sigma_v
        for a in ['_P02', '_P12', '_P22', '_P03', '_P13', '_P04']:
            if hasattr(self, a): delattr(self, a)
        
    @sigma_v.deleter
    def sigma_v(self):
        try:
            del self._sigma_v
        except AttributeError:
            pass

    #---------------------------------------------------------------------------
    @property
    def sigma_v2(self):
        """
        The additional, small-scale velocity dispersion, evaluated using the 
        halo model and weighted by the velocity squared. [units: km/s]

        .. math:: (sigma_{v2})^2 = (1/\bar{rho}) * \int dM M \frac{dn}{dM} v_{\parallel}^2

        Returns 0 if not defined
        """
        try:
            return self._sigma_v2
        except AttributeError:
            return 0.

    @sigma_v2.setter
    def sigma_v2(self, val):
        self._sigma_v2 = val
 
        for a in ['_P01', '_P03']:
            if hasattr(self, a): delattr(self, a)

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

        Returns 0 if not defined
        """
        try:
            return self._sigma_bv2
        except AttributeError:
            return 0.
            

    @sigma_bv2.setter
    def sigma_bv2(self, val):
        self._sigma_bv2 = val

        for a in ['_P02', '_P12', '_P13', '_P22']:
            if hasattr(self, a): delattr(self, a)

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

        Returns 0 if not defined
        """
        try:
            return self._sigma_bv4
        except AttributeError:
            return 0.

    @sigma_bv4.setter
    def sigma_bv4(self, val):
        self._sigma_bv4 = val
        if hasattr(self, '_P04'): delattr(self, '_P04')

    @sigma_bv4.deleter
    def sigma_bv4(self):
        try:
            del self._sigma_bv4
        except AttributeError:
            pass
   
    #---------------------------------------------------------------------------
    @property
    def integrals(self):
        try:
            return self._integrals
        except AttributeError:
            self._integrals = integrals.Integrals(self.power_lin, self.z, self.sigma8)
            return self._integrals
    
    #---------------------------------------------------------------------------
    @property
    def P00_model(self):
        """
        The class holding the model for the P00 dark matter term
        """
        try:
            return self._P00_model
        except AttributeError:
            self._P00_model = dm_power_moments.DarkMatterP00(self.power_lin, self.z, self.sigma8, model_type=self.DM_model)
            return self._P00_model
    
    #---------------------------------------------------------------------------
    @property
    def P01_model(self):
        """
        The class holding the model for the P01 dark matter term
        """
        try:
            return self._P01_model
        except AttributeError:
            self._P01_model = dm_power_moments.DarkMatterP01(self.power_lin, self.z, self.sigma8, model_type=self.DM_model)
            return self._P01_model
            
    #---------------------------------------------------------------------------
    # POWER TERM ATTRIBUTES (READ-ONLY)
    #---------------------------------------------------------------------------
    def P_mu0(self, k):
        """
        The full power spectrum term with no angular dependence. Contributions
        from P00.
        """
        try:
            return self._P_mu0_spline(k)
        except AttributeError:
            Pk = self.P00.total.mu0
            self._P_mu0_spline = spline(self.k, Pk)
            return self._P_mu0_spline(k)
        
    #end P_mu0
    #---------------------------------------------------------------------------
    def P_mu2(self, k):
        """
        The full power spectrum term with mu^2 angular dependence. Contributions
        from P01, P11, and P02.
        """
        try:
            return self._P_mu2_spline(k)
        except AttributeError:
            Pk = self.P01.total.mu2 + self.P11.total.mu2 + self.P02.total.mu2
            self._P_mu2_spline = spline(self.k, Pk)
            return self._P_mu2_spline(k)
        
    #end P_mu2
    #---------------------------------------------------------------------------
    def P_mu4(self, k):
        """
        The full power spectrum term with mu^4 angular dependence. Contributions
        from P11, P02, P12, P22, P03, P13 (2-loop), and P04 (2-loop).
        """
        try:
            return self._P_mu4_spline(k)
        except AttributeError:
            Pk = self.P11.total.mu4 + self.P02.total.mu4 + self.P12.total.mu4 + self.P22.total.mu4 + self.P03.total.mu4
            if self.include_2loop: Pk += self.P13.total.mu4 + self.P04.total.mu4
            self._P_mu4_spline = spline(self.k, Pk)
            return self._P_mu4_spline(k)

    #end P_mu4
    #---------------------------------------------------------------------------
    def P_mu6(self, k):
        """
        The full power spectrum term with mu^6 angular dependence. Contributions
        from P12, P22, P13, and P04 (2-loop).
        """
        try:
            return self._P_mu6_spline(k)
        except AttributeError:
            Pk = self.P12.total.mu6 + self.P22.total.mu6 + self.P13.total.mu6
            if self.include_2loop: Pk += self.P04.total.mu6
            self._P_mu6_spline = spline(self.k, Pk)
            return self._P_mu6_spline(k)
            
    #end P_mu6
    #---------------------------------------------------------------------------
    @property
    def Pdd(self):
        """
        The 1-loop auto-correlation of density.
        """
        norm = self._power_norm*self.D**2
        return norm*(self.power_lin(self.k) + norm*self.integrals.Pdd(self.k))
        
    #---------------------------------------------------------------------------
    @property
    def Pdv(self):
        """
        The 1-loop cross-correlation between dark matter density and velocity 
        divergence.
        """
        # check for any user-loaded values
        if hasattr(self, '_Pdv_loaded'):
            return self._Pdv_loaded(self.k)
        else:
            norm = self._power_norm*self.D**2
            return (-self.f)*norm*(self.power_lin(self.k) + norm*self.integrals.Pdv(self.k))
          
    #---------------------------------------------------------------------------
    @property
    def Pvv(self):
        """
        The 1-loop auto-correlation of velocity divergence.
        """
        norm = self._power_norm*self.D**2
        return self.f**2 * norm*(self.power_lin(self.k) + norm*self.integrals.Pvv(self.k))
    
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
                self._P00.total.mu0 = self._P00_mu0_loaded(self.k)
            else:
                
                # use the DM model
                if self.DM_model is not None:
                    self._P00.total.mu0 = self.P00_model.power(self.k)
                # use pure PT
                else:
                    # the necessary integrals 
                    I00 = self.integrals.I00(self.k)
                    J00 = self.integrals.J00(self.k)
            
                    P11 = self.normed_power_lin(self.k)
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
                self._P01.total.mu2 = self._P01_mu2_loaded(self.k)
            else:
                
                # use the DM model
                if self.DM_model is not None:
                    self._P01.total.mu2 = self.P01_model.power(self.k)
                # use pure PT
                else:                
                    # the necessary integrals 
                    I00 = self.integrals.I00(self.k)
                    J00 = self.integrals.J00(self.k)
            
                    Plin = self.normed_power_lin(self.k)
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
                    Pvec = self._P11_mu2_loaded(self.k)
                    self._P11.vector.mu2 = self._P11.total.mu2 = Pvec
                    self._P11.vector.mu4 = self._P11.vector.mu4 = -Pvec
                else:
                    
                    # do the vector part, contributing mu^2 and mu^4 terms
                    if not self.include_2loop:
                        Pvec = self.f**2 * self.integrals.I31(self.k)
                    else:
                        I1 = self.integrals.Ivvdd_h01(self.k)
                        I2 = self.integrals.Idvdv_h03(self.k)
                        Pvec = self.f**2 * (I1 + I2)
                
                    # save the mu^2 vector term
                    self._P11.vector.mu2 = self._P11.total.mu2 = Pvec
                    self._P11.vector.mu4 = self._P11.vector.mu4 = -Pvec
                
                # do mu^4 terms?
                if self.max_mu >= 4: 
                      
                    # check and return any user-loaded values
                    if hasattr(self, '_P11_mu4_loaded'):
                        self._P11.total.mu4 = self._P11_mu4_loaded(self.k)
                    else:
                          
                        # compute the scalar mu^4 contribution
                        if self.include_2loop:
                            I1 = self.integrals.Ivvdd_h02(self.k)
                            I2 = self.integrals.Idvdv_h04(self.k)
                            C11_contrib = I1 + I2
                        else:
                            C11_contrib = self.integrals.I13(self.k)
                    
                        # the necessary integrals 
                        I11 = self.integrals.I11(self.k)
                        I22 = self.integrals.I22(self.k)
                        J11 = self.integrals.J11(self.k)
                        J10 = self.integrals.J10(self.k)
                    
                        Plin = self.normed_power_lin(self.k)
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
                Plin = self.normed_power_lin(self.k)
                
                # the necessary integrals 
                I02 = self.integrals.I02(self.k)
                J02 = self.integrals.J02(self.k)
    
                # the mu^2 no velocity terms
                self._P02.no_velocity.mu2 = self.f**2 * (I02 + 2.*self.k**2*J02*Plin)
                
                # the mu^2 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_v
                sigma_02  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
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
                    I20 = self.integrals.I20(self.k)
                    J20 = self.integrals.J20(self.k)
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
                Plin = self.normed_power_lin(self.k)
                
                # the necessary integrals 
                I12 = self.integrals.I12(self.k)
                I03 = self.integrals.I03(self.k)
                J02 = self.integrals.J02(self.k)
                
                # do the mu^4 terms that don't depend on velocity
                self._P12.no_velocity.mu4 = self.f**3 * (I12 - I03 + 2*self.k**2*J02*Plin)
            
                # now do mu^4 terms depending on velocity (velocities in Mpc/h)
                sigma_lin = self.sigma_v  
                sigma_12  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
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
                    I21 = self.integrals.I21(self.k)
                    I30 = self.integrals.I30(self.k)
                    J20 = self.integrals.J20(self.k)
                    
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
                sigma_lin = self.sigma_v
                sigma_22  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_22**2
                
                J02 = self.integrals.J02(self.k)
                J20 = self.integrals.J20(self.k)
                
            # do mu^4 terms?
            if self.max_mu >= 4:
                
                Plin = self.normed_power_lin(self.k)
                
                # 1-loop or 2-loop terms from <v^2 | v^2 > 
                if not self.include_2loop:
                    self._P22.no_velocity.mu4 = 1./16*self.f**4 * self.integrals.I23(self.k)
                else:
                    I23_2loop = self.integrals.Ivvvv_f23(self.k)
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
                    extra2_dvdv_mu4 = 0.5*(self.f*self.k)**4 * self.P00.total.mu0*self.integrals.sigmasq_k(self.k)**2
                    
                    # store the extra two loop terms
                    extra = extra_vv_mu4 + extra_vdv_mu4 + extra1_dvdv_mu4 + extra2_dvdv_mu4
                    self._P22.total.mu4 = self._P22.no_velocity.mu4 + extra
                    
                else:
                    self._P22.total.mu4 = self._P22.no_velocity.mu4
                    
                # do mu^6 terms?
                if self.max_mu >= 6:
                    
                    # 1-loop or 2-loop terms that don't depend on velocity
                    if not self.include_2loop:
                        self._P22.no_velocity.mu6 = 1./8*self.f**4 * self.integrals.I32(self.k)
                    else:
                        I32_2loop = self.integrals.Ivvvv_f32(self.k)
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
                            self._P22.no_velocity.mu8 = 1./16*self.f**4 * self.integrals.I33(self.k)
                        else:
                            I33_2loop = self.integrals.Ivvvv_f33(self.k)
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
                Plin = self.normed_power_lin(self.k)
                
                # only terms depending on velocity here (velocities in Mpc/h)
                sigma_lin = self.sigma_v 
                sigma_03  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D)
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
            Plin = self.D**2 * self.normed_power_lin(self.k)
            
            # compute velocity weighting in Mpc/h
            sigma_lin = self.sigma_v 
            sigma_13_v  = self.sigma_bv2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
            sigsq_eff_vector = sigma_lin**2 + sigma_13_v**2
            
            if self.include_2loop:
                sigma_13_s  = self.sigma_v2 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
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
                sigma_lin = self.sigma_v 
                sigma_04  = self.sigma_bv4 * self.cosmo.h() / (self.f*self.conformalH*self.D) 
                sigsq_eff = sigma_lin**2 + sigma_04**2
                
                # do mu^4 terms?
                if self.max_mu >= 4:
                    
                    # do P04 mu^4 terms depending on velocity
                    P04_vel_mu4_1 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu2
                    P04_vel_mu4_2 = 0.25*(self.f*self.k)**4 * (self.D**2*sigsq_eff)**2 * self.P00.total.mu0
                    self.P04.with_velocity.mu4 = P04_vel_mu4_1 + P04_vel_mu4_2
                    
                    # do P04 mu^4 terms without vel dependence
                    self.P04.no_velocity.mu4 = 1./12.*(self.f*self.k)**4 * self.P00.total.mu0*self.integrals.velocity_kurtosis
                
                    # save the total
                    self.P04.total.mu4 = self.P04.with_velocity.mu4 + self.P04.no_velocity.mu4
                
                    # do mu^6 terms?
                    if self.max_mu >= 6:
                        
                        # only terms depending on velocity
                        self.P04.with_velocity.mu6 = -0.5*(self.f*self.D*self.k)**2 * sigsq_eff * self.P02.no_velocity.mu4
                        self.P04.total.mu6 = self.P04.with_velocity.mu6
                        
            return self._P04
    #---------------------------------------------------------------------------
    def _power_one_mu(self, mu_obs):
        """
        Internal function to evaluate P(k, mu) at a scalar mu value
        """
        # set the observed mu value
        mu = self.mu_true(mu_obs)
        k = self.k_true(self.k_obs, mu_obs)
        vol_scaling = 1./(self.alpha_perp**2 * self.alpha_par)
                
        if self.max_mu == 0:
            P_out = self.P_mu0(k)
        elif self.max_mu == 2:
            P_out = self.P_mu0(k) + mu**2*self.P_mu2(k)
        elif self.max_mu == 4:
            P_out = self.P_mu0(k) + mu**2*self.P_mu2(k) + mu**4*self.P_mu4(k)
        elif self.max_mu == 6:
            P_out = self.P_mu0(k) + mu**2*self.P_mu2(k) + mu**4*self.P_mu4(k) + mu**6*self.P_mu6(k)
        elif self.max_mu == 8:
            raise NotImplementedError("Cannot compute power spectrum including terms with order higher than mu^6")
            
        return vol_scaling*P_out
    #end power
    
    #---------------------------------------------------------------------------
    def power(self, mu, flatten=False):
        """
        Return the redshift space power spectrum at the specified value of mu, 
        including terms up to ``mu**self.max_mu``.
        
        Parameters
        ----------
        mu : float, array_like
            The mu values to evaluate the power at.
        
        Returns
        -------
        Pkmu : float, array_like
            The power model P(k, mu). If `mu` is a scalar, return dimensions
            are `(len(self.k), )`. If `mu` has dimensions (N, ), the return
            dimensions are `(len(k), N)`, i.e., each column corresponds is the
            model evaluated at different `mu` values. If `flatten = True`, then
            the returned array is raveled, with dimensions of `(N*len(self.k), )`
        """
        if np.isscalar(mu):
            return self._power_one_mu(mu)
        else:
            toret = np.vstack([self._power_one_mu(imu) for imu in mu]).T
            if flatten: toret = np.ravel(toret, order='F')
            return toret
    #end power
    
    #---------------------------------------------------------------------------
    @tools.monopole
    def monopole(self, mu):
        """
        The monopole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(mu)
        
    #end monopole
    
    #---------------------------------------------------------------------------
    @tools.quadrupole
    def quadrupole(self, mu):
        """
        The quadrupole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(mu)
        
    #end quadrupole
    
    #---------------------------------------------------------------------------
    @tools.hexadecapole
    def hexadecapole(self, mu):
        """
        The hexadecapole moment of the power spectrum. Include mu terms up to 
        mu**max_mu.
        """
        return self.power(mu)
    
    #end hexadecapole
    
    #---------------------------------------------------------------------------
    def load(self, k_data, power_data, power_term, mu_term, errs=None):
        """
        Load data into a given power attribute, as specified by power_term
        and mu_term.
        """        
        power_name = "_%s" %power_term
        if hasattr(self, power_name):
            del self.__dict__[power_name]
        
        if errs is not None:
            w = 1./np.array(errs)
        else:
            w = None
        s = spline(k_data, power_data, w=w)
        
        if mu_term is not None:
            setattr(self, "%s_%s_loaded" %(power_name, mu_term), s)
        else:
            setattr(self, "%s_loaded" %power_name, s)
            
    #end load
    #---------------------------------------------------------------------------
    def unload(self, power_term, mu_term):
        """
        Delete the given power attribute, as specified by power_term.
        """            
        # delete the loaded data
        if mu_term is not None:
            name = "%s_%s_loaded" %(power_name, mu_term)
            if hasattr(name): del self.__dict__[name]
        else:
            name = "%s_loaded" %power_name
            if hasattr(name): del self.__dict__[name]
        
        # delete all power attributes
        self._delete_power() 
        
    #end unload
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


        
        
