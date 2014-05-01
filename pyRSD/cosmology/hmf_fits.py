"""
 hmf_fits.py
 pyPT: the multiplicity fitting functions for the halo mass function are 
       defined here
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/24/2014
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class Multiplicity(object):
    
    available = {'PS' : '_fsigma_PS', 
                 'SMT': '_fsigma_SMT',
                 'Jenkins': '_fsigma_Jenkins',
                 'Warren': '_fsigma_Warren',
                 'Tinker': '_fsigma_Tinker',
                 'Tinker_norm': '_fsigma_Tinker_norm'}
    
    def __init__(self, hmf, cut=True):
        self.hmf = hmf
        self.cut = cut
    #end __init__
    
    #----------------------------------------------------------------------------
    def fsigma(self):
        if self.hmf.mf_fit in Multiplicity.available:
             return getattr(self, Multiplicity.available[self.hmf.mf_fit])()
    #end fsigma
    
    #---------------------------------------------------------------------------
    def _fsigma_PS(self):
        """
        Calculate f(sigma) for Press-Schechter form.

        Press, W. H., Schechter, P., 1974. ApJ 187, 425-438.
        http://adsabs.harvard.edu/full/1974ApJ...187..425P
        """
        nu = self.hmf.cosmo.delta_c / self.hmf.sigma
        return np.sqrt(2./np.pi)*nu*np.exp(-0.5*nu*nu)
    #end _fsigma_PS

    #---------------------------------------------------------------------------
    def _fsigma_SMT(self):
        """
        Calculate f(sigma) for Sheth-Mo-Tormen form.

        Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
        http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x
        """
        nu = self.hmf.cosmo.delta_c / self.hmf.sigma
        a = 0.707
        return 0.3222*np.sqrt(2.*a/np.pi)*nu*np.exp(-0.5*(a*nu*nu)) \
                 * (1 + (1./(a*nu*nu))**0.3)
    #end _fsigma_SMT

    #---------------------------------------------------------------------------
    def _fsigma_Jenkins(self):
        """
        Calculate f(sigma) for Jenkins form.

        Jenkins, A. R., et al., Feb. 2001. MNRAS 321 (2), 372-384.
        http://doi.wiley.com/10.1046/j.1365-8711.2001.04029.x
    
        Notes
        ------
        Only valid for :math: -1.2 < \ln \sigma^{-1} < 1.05
        """
        fsigma = 0.315*np.exp(-np.abs(self.hmf.lnsigma + 0.61)**3.8)

        # set to NaN outside the relevant bounds
        if self.cut:
            fsigma[np.logical_or(self.hmf.lnsigma < -1.2, self.hmf.lnsigma > 1.05)] = np.NaN

        return fsigma
    #end _fsigma_Jenkins

    #---------------------------------------------------------------------------
    def _fsigma_Warren(self):
        """
        Calculate f(sigma) for Warren form.

        Warren, M. S., et al., Aug. 2006. ApJ 646 (2), 881-885.
        http://adsabs.harvard.edu/abs/2006ApJ...646..881W
    
        Notes
        ------
        Only valid for :math: `10^{10}M_\odot < M <10^{15}M_\odot`
        """
        fsigma = 0.7234 * ((1./self.hmf.sigma)**1.625 + 0.2538) * \
                np.exp(-1.1982/(self.hmf.sigma**2))

        if self.cut:
            fsigma[np.logical_or(M < 10**10, self.hmf.M > 10**15)] = np.NaN
        return fsigma
    #end _fsigma_Warren
    
    #---------------------------------------------------------------------------
    def _fsigma_Tinker(self):
        """
        Calculate f(sigma) for the unnormalized (divergent at low mass)
        Tinker form. This is f(sigma) from Tinker et al. 2008.

        Tinker, J., et al., 2008. ApJ 688, 709-728.
        http://iopscience.iop.org/0004-637X/688/2/709

        Notes
        -----
        Only valid for :math:`-0.6<\log_{10}\sigma^{-1}<0.4`
        """
        # The Tinker function is a bit tricky - we use the code from
        # http://cosmo.nyu.edu/~tinker/massfunction/MF_code.tar to aid us.
        delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        A_array = np.array([ 1.858659e-01,
                            1.995973e-01,
                            2.115659e-01,
                            2.184113e-01,
                            2.480968e-01,
                            2.546053e-01,
                            2.600000e-01,
                            2.600000e-01,
                            2.600000e-01])

        a_array = np.array([1.466904e+00,
                            1.521782e+00,
                            1.559186e+00,
                            1.614585e+00,
                            1.869936e+00,
                            2.128056e+00,
                            2.301275e+00,
                            2.529241e+00,
                            2.661983e+00])

        b_array = np.array([2.571104e+00 ,
                            2.254217e+00,
                            2.048674e+00,
                            1.869559e+00,
                            1.588649e+00,
                            1.507134e+00,
                            1.464374e+00,
                            1.436827e+00,
                            1.405210e+00])

        c_array = np.array([1.193958e+00,
                            1.270316e+00,
                            1.335191e+00,
                            1.446266e+00,
                            1.581345e+00,
                            1.795050e+00,
                            1.965613e+00,
                            2.237466e+00,
                            2.439729e+00])
        
        # interpolate between the delta_virs
        # to get the correct value
        A_func = spline(delta_virs, A_array)
        a_func = spline(delta_virs, a_array)
        b_func = spline(delta_virs, b_array)
        c_func = spline(delta_virs, c_array)

        A_0 = A_func(self.hmf.delta_halo)
        a_0 = a_func(self.hmf.delta_halo)
        b_0 = b_func(self.hmf.delta_halo)
        c_0 = c_func(self.hmf.delta_halo)

        A     = A_0*(1 + self.hmf.z)**(-0.14)
        a     = a_0*(1 + self.hmf.z)**(-0.06)
        alpha = 10**(-(0.75 / np.log10(self.hmf.delta_halo / 75))**1.2)
        b     = b_0*(1 + self.hmf.z) ** (-alpha)
        c     = c_0

        fsigma = A*((self.hmf.sigma/b)**(-a) + 1)*np.exp(-c/self.hmf.sigma**2)

        if self.cut:
            if self.hmf.z == 0.:
                fsigma[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.6 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.NaN
            else:
                fsigma[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.2 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.NaN
        return fsigma
    #end _fsigma_Tinker
    
    #---------------------------------------------------------------------------
    def _fsigma_Tinker_norm(self):
        """
        Calculate f(sigma) for the non-divergent Tinker form. This has been
        normalized such that the dark matter has an average bias of unity, 
        integrated over the mass function (from Tinker et al. 2010)

        Tinker, J., et al., 2010. ApJ 724, 878-886.
        http://iopscience.iop.org/0004-637X/724/2/878

        Notes
        -----
        Only valid for :math:`-0.6<\log_{10}\sigma^{-1}<0.4`
        """
        # The Tinker function is a bit tricky - we use the code from
        # http://cosmo.nyu.edu/~tinker/massfunction/MF_code.tar to aid us.
        delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        
        if self.hmf.delta_halo in delta_virs:
            index = np.where(delta_virs == self.hmf.delta_halo)[0]
        else:
            raise ValueError("delta_halo = %s is invalid; must be one of %s" %(self.hmf.delta_halo, delta_virs))
        alpha_array = np.array([ 0.368,
                                 0.363,
                                 0.385,
                                 0.389,
                                 0.393,
                                 0.365,
                                 0.379,
                                 0.355,
                                 0.327])

        beta_array = np.array([ 0.589,
                                0.585,
                                0.544,
                                0.543,
                                0.564,
                                0.623,
                                0.637,
                                0.673,
                                0.702])

        gamma_array = np.array([ 0.864,
                                 0.922,
                                 0.987,
                                 1.09,
                                 1.20,
                                 1.34,
                                 1.50,
                                 1.68,
                                 1.81])

        phi_array = np.array([ -0.729,
                               -0.789,
                               -0.910,
                               -1.050,
                               -1.200,
                               -1.260,
                               -1.450,
                               -1.500,
                               -1.490])
        
        eta_array = np.array([ -0.243,
                               -0.261,
                               -0.261,
                               -0.273,
                               -0.278,
                               -0.301,
                               -0.301,
                               -0.319,
                               -0.336 ])
                               
        # interpolate between the delta_virs
        # to get the correct value
        alpha_func = spline(delta_virs, alpha_array)
        beta_func  = spline(delta_virs, beta_array)
        gamma_func = spline(delta_virs, gamma_array)
        phi_func   = spline(delta_virs, phi_array)
        eta_func   = spline(delta_virs, eta_array)
        
        alpha_0 = alpha_func(self.hmf.delta_halo)
        beta_0  = beta_func(self.hmf.delta_halo)
        gamma_0 = gamma_func(self.hmf.delta_halo)
        phi_0   = phi_func(self.hmf.delta_halo)
        eta_0   = eta_func(self.hmf.delta_halo)

        z     = self.hmf.z if self.hmf.z < 3 else 3.
        alpha = alpha_0
        beta  = beta_0*(1 + z)**(0.20)
        phi   = phi_0*(1 + z)**(-0.08)
        gamma = gamma_0*(1 + z)**(-0.01)
        eta   = eta_0*(1 + z)**(0.27)
        
        
        nu = self.hmf.cosmo.delta_c / self.hmf.sigma
        fnu = alpha * (1 + (beta*nu)**(-2*phi)) * nu**(2*eta) * np.exp(-0.5*gamma*nu**2)

        if self.cut:
            if self.hmf.z == 0.:
                fnu[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.6 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.NaN
            else:
                fnu[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.2 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.NaN
        return nu*fnu
    #end _fsigma_Tinker_norm
    #---------------------------------------------------------------------------
#endclass Multiplicity


