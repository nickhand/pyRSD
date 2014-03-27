from pyPT.power cimport integral_base
from pyPT.cosmology import growth
import numpy as np
cimport numpy as np

CONVERGENCE_FACTOR = 100.
    
class Integrals(object):
    """
    Class to compute and store the necessary PT integrals for the dark 
    matter redshift space power spectrum, in a redshift-independent manner.
    """
    def __init__(self, k_eval, z, cosmo, num_threads):
        
        self.k_eval      = k_eval
        self.z           = z
        self.kmin        = np.amin(k_eval)
        self.kmax        = np.amax(k_eval)
        self.cosmo       = cosmo
        self.num_threads = num_threads
    #end __init__
    #---------------------------------------------------------------------------
    def J(self, kernel_name):
        """
        Return the value of the integral J_nm(k), at the specified order, 
        with the correct redshift dependence.
        """
        name = '_J_%s' %kernel_name

        # linear order 
        try:
            J1 = getattr(self, "%s_1" %name)
        except:
            self.J_lin.kernel_name = kernel_name
            J1 = self.J_lin.evaluate(self.k_eval)
            self.__dict__["%s_1" %name] = J1
            
        return self.D**2 * J1
    
    #---------------------------------------------------------------------------
    def I(self, kernel_name, order, variables=None):
        """
        Return the value of the integral I_nm(k), at the specified order, 
        with the correct redshift dependence.
        """
        name = '_I_%s' %kernel_name

        # linear order 
        try:
            I1 = getattr(self, "%s_1" %name)
        except:
            self.I_lin.kernel_name = kernel_name
            I1 = self.I_lin.evaluate(self.k_eval)
            self.__dict__["%s_1" %name] = I1
            
        # return I1 if we are doing linear order
        if order == 0:
            return self.D**4 * I1
            
        # 1-loop order
        else:
            
            # now do the first linear-1loop integrals
            try:
                I2 = getattr(self, "%s_2" %name)
            except:
                thisI = getattr(self, 'I_lin_%s1loop' %variables[0])
                thisI.kernel_name = kernel_name
                I2 = thisI.evaluate(self.k_eval)
                self.__dict__["%s_2" %name] = I2
            
            # and do the second, if variables are different
            if variables[0] == variables[1]:
                I3 = I2
            else:
                try:
                    I3 = getattr(self, "%s_3" %name)
                except:
                    thisI = getattr(self, 'I_lin_%s1loop' %variables[1])
                    thisI.kernel_name = kernel_name
                    I3 = thisI.evaluate(self.k_eval) 
                    self.__dict__["%s_3" %name] = I3
            
            # now do the 1loop-1loop integral
            try:
                I4 = getattr(self, "%s_4" %name)
            except:
                if variables[0] == variables[1]:
                    thisI = getattr(self, 'I_%s1loop' %variables[0])
                    thisI.kernel_name = kernel_name
                    I4 = thisI.evaluate(self.k_eval)
                else:
                    self.I_dd1loop_vv1loop.kernel_name = kernel_name
                    I4 = self.I_dd1loop_vv1loop.evaluate(self.k_eval)
                self.__dict__["%s_4" %name] = I4
        
            return self.D**4 * (I1 + self.D**2 * (I2 + I3) + self.D**4 * I4)
    #---------------------------------------------------------------------------
    @property
    def vel_kurtosis(self):
        """
        Compute the velocity kurtosis <v_parallel^4>, using the 1-loop divergence
        auto spectra Pvv.
        """
        # compute P22_bar first (using linear power spectra)
        I23_1, I32_1, I33_1 = self._P22bar
        
        I1 = I23_1 + (2./3)*I32_1 + (1./5)*I33_1
        try:
            kurtosis = self._kurtosis
        except:
            J = integral_base.integral1D('g22', self.kmin_1loop, self.kmax_1loop, 
                                          self.k1loop, I1)
            
            # evaluate at any k (should all be same)  
            k = np.array([0.1])
            self._kurtosis = kurtosis = J.evaluate(k)/k**2
        
        return self.D**4 * 0.25 * kurtosis[0]
    #---------------------------------------------------------------------------
    @property
    def sigmasq_k(self):
        """
        This is sigma^2 as a function of k. 
        """
        try:
            return self.D**2 * self._sigmasq_k
        except:
            J = integral_base.integral1D('unity', self.kmin_1loop, self.kmax_1loop, 
                                          self.klin, self.Plin)
            
            sigmasq_k = []
            # evaluate at any k (should all be same)  
            for k in self.k_eval:
                J.kmax = 0.5*k # integrate up to 0.5*k
                sigmasq_k.append(J.evaluate(np.array([k]))[0])
            self._sigmasq_k = 1./3*np.array(sigmasq_k)
            
            return self.D**2 * self._sigmasq_k
    
    #---------------------------------------------------------------------------
    # INTEGRAL BASE CLASSES FOR EACH SPECTRA COMBO
    #---------------------------------------------------------------------------
    @property
    def I_lin(self):
        try:
            return self._I_lin
        except:
            self._I_lin = integral_base.integral2D('f00', self.kmin_lin, self.kmax_lin, 
                                                    self.num_threads, 
                                                    self.klin, self.Plin,
                                                    k2=self.klin, P2=self.Plin)
            return self._I_lin
    #----------------------------------------------------------------------------
    @property
    def I_lin_vv1loop(self):
        try:
            return self._I_lin_vv1loop
        except:
            self._I_lin_vv1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.klin, self.Plin,
                                                            k2=self.k1loop, P2=self.Pvv_1loop)
            return self._I_lin_vv1loop
    #----------------------------------------------------------------------------
    @property
    def I_vv1loop(self):
        try:
            return self._I_vv1loop
        except:
            self._I_vv1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                        self.num_threads, 
                                                        self.k1loop, self.Pvv_1loop,
                                                        k2=self.k1loop, P2=self.Pvv_1loop)
            return self._I_vv1loop
    #---------------------------------------------------------------------------
    @property
    def I_lin_dd1loop(self):
        try:
            return self._I_lin_dd1loop
        except:
            self._I_lin_dd1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.klin, self.Plin,
                                                            k2=self.k1loop, P2=self.Pdd_1loop)
            return self._I_lin_dd1loop
    #---------------------------------------------------------------------------
    @property
    def I_dd1loop(self):
        try:
            return self._I_lin_dd1loop
        except:
            self._I_lin_dd1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.k1loop, self.Pdd_1loop,
                                                            k2=self.k1loop, P2=self.Pdd_1loop)
            return self._I_lin_dd1loop
    #---------------------------------------------------------------------------
    @property
    def I_dd1loop_vv1loop(self):
        try:
            return self._I_dd1loop_vv1loop
        except:
            self._I_dd1loop_vv1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                                self.num_threads, 
                                                                self.k1loop, self.Pdd_1loop,
                                                                k2=self.k1loop, P2=self.Pvv_1loop)
            return self._I_dd1loop_vv1loop
    #---------------------------------------------------------------------------
    @property
    def I_lin_dv1loop(self):
        try:
            return self._I_lin_dv1loop
        except:
            self._I_lin_dv1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, self.klin, self.Plin,
                                                            k2=self.k1loop, P2=self.Pdv_1loop)
            return self._I_lin_dv1loop
    #---------------------------------------------------------------------------
    @property
    def I_dv1loop(self):
        try:
            return self._I_dv1loop
        except:
            self._I_dv1loop = integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                        self.num_threads, 
                                                        self.k1loop, self.Pdv_1loop,
                                                        k2=self.k1loop, P2=self.Pdv_1loop)
            return self._I_dv1loop
    #---------------------------------------------------------------------------
    @property
    def J_lin(self):
        try:
            return self._J_lin
        except:
            self._J_lin = integral_base.integral1D('g00', self.kmin_lin, self.kmax_lin, 
                                                    self.klin, self.Plin)
            return self._J_lin
    
    #---------------------------------------------------------------------------
    # WAVENUMBERS FOR LINEAR/1-LOOP ORDER
    #---------------------------------------------------------------------------
    @property
    def klin(self):
        try:
            return self._klin
        except:
            # must have wide enough k region to converge
            kmin = self.kmin/(CONVERGENCE_FACTOR)**2
            kmax = self.kmax*(CONVERGENCE_FACTOR)**2
            
            self._klin = np.logspace(np.log10(kmin), np.log10(kmax), 1000)
            return self._klin
    #---------------------------------------------------------------------------
    @property
    def k1loop(self):
        try:
            return self._k1loop
        except:
            # must have wide enough k region to converge
            kmin = self.kmin/(CONVERGENCE_FACTOR)
            kmax = self.kmax*(CONVERGENCE_FACTOR)
            
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
    # THE REDSHIFT DEPENDENT QUANTITIES
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
    # THE POWER SPECTRA
    #---------------------------------------------------------------------------
    @property
    def Plin(self):
        """
        Linear power spectrum.
        """
        try:
            return self._Plin
        except:

            self._Plin = growth.Pk_lin(self.klin, 0., tf='EH', params=self.cosmo)
            return self._Plin 
    #---------------------------------------------------------------------------
    @property
    def _P11(self):
        """
        For internal use.
        """
        try:
            return self.__P11
        except:

            self.__P11 = growth.Pk_lin(self.k1loop, 0., tf='EH', params=self.cosmo)
            return self.__P11
    #---------------------------------------------------------------------------
    @property
    def Pdd_1loop(self):
        """
        The 1-loop component of the auto-correlation of density.
        """
        try:
            return self._Pdd_1loop
        except:
            # compute I00
            self.I_lin.kernel_name = 'f00'
            I00 = self.I_lin.evaluate(self.k1loop)

            # compute J00
            self.J_lin.kernel_name = 'g00'
            J00 = self.J_lin.evaluate(self.k1loop)

            P22 = 2.*I00
            P13 = 6.*self.k1loop**2*J00*self._P11
            self._Pdd_1loop = P22 + P13
            
            return self._Pdd_1loop
    #---------------------------------------------------------------------------
    @property
    def Pdv_1loop(self):
        """
        The 1-loop component of the cross-correlation of density and velocity 
        divergence.
        """
        try:
            return self._Pdv_1loop
        except:       
            # compute I01
            self.I_lin.kernel_name = 'f01'
            I01 = self.I_lin.evaluate(self.k1loop)

            # compute J01
            self.J_lin.kernel_name = 'g01'
            J01 = self.J_lin.evaluate(self.k1loop)

            P22  = 2.*I01
            P13  = 6.*self.k1loop**2*J01*self._P11
            self._Pdv_1loop = P22 + P13
            
            return self._Pdv_1loop
    #---------------------------------------------------------------------------
    @property
    def Pvv_1loop(self):
        """
        The 1-loop component of the auto-correlation of velocity divergence.
        """
        try:
            return self._Pvv_1loop
        except:     
            # compute I11
            self.I_lin.kernel_name = 'f11'
            I11 = self.I_lin.evaluate(self.k1loop)

            # compute J11
            self.J_lin.kernel_name = 'g11'
            J11 = self.J_lin.evaluate(self.k1loop)

            P22  = 2.*I11
            P13  = 6.*self.k1loop**2*J11*self._P11
            self._Pvv_1loop = P22 + P13
            
            return self._Pvv_1loop
    #---------------------------------------------------------------------------
    @property
    def _P22bar(self):
        """
        The linear order P22 bar components
        """
        try:
            I1 = self._P22bar1_mu0
        except:
            self.I_lin.kernel_name = 'f23'
            self._P22bar1_mu0 = I1 = self.I_lin.evaluate(self.k1loop)
        
        try:
            I2 = self._P22bar1_mu2
        except:
            self.I_lin.kernel_name = 'f32'
            self._P22bar1_mu2 = I2 = self.I_lin.evaluate(self.k1loop) 
            
        try:
            I3 = self._P22bar1_mu4
        except:
            self.I_lin.kernel_name = 'f33'
            self._P22bar1_mu4 = I3 = self.I_lin.evaluate(self.k1loop)
            
        return I1, I2, I3          
    #---------------------------------------------------------------------------
    @property
    def K00(self):
        """
        The nonlinear biasing term showing up in P00hh
        """
        try:
            return self._K00
        except:
            self.I_lin.kernel_name = 'k00'
            self._K00 = self.I_lin.evaluate(self.k1loop)
            return self._K00
    #---------------------------------------------------------------------------
    @property
    def K00s(self):
        """
        The tidal nonlinear biasing term showing up in P00hh
        """
        try:
            return self._K00s
        except:
            self.I_lin.kernel_name = 'k00s'
            self._K00s = self.I_lin.evaluate(self.k1loop)
            return self._K00s
    #---------------------------------------------------------------------------
