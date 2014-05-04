#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True

from pyRSD.rsd cimport _integral_base
from pyRSD.cosmology cimport growth
from ..cosmology.power import Power
from ..cosmology.cosmo import Cosmology
import numpy as np

CONVERGENCE_FACTOR = 100.
NORM_FACTOR = 1e6

class Integrals(object):
    """
    Class to compute and store the necessary PT integrals for the dark 
    matter redshift space power spectrum, in a redshift-independent manner.
    """
           
    def __init__(self, k_eval, z, cosmo, num_threads, transfer_fit, camb_kwargs):
        
        self._k_eval      = k_eval
        self._z           = z
        self._kmin        = np.amin(k_eval)
        self._kmax        = np.amax(k_eval)
        self._cosmo       = cosmo
        self._num_threads = num_threads
        
        # initialize the power classes
        self._power = Power(k=self.klin, z=0., transfer_fit=transfer_fit, 
                            cosmo=self._cosmo, **camb_kwargs)
        
    #end __init__
    #---------------------------------------------------------------------------
    # READ-ONLY ATTRIBUTES
    #---------------------------------------------------------------------------
    @property
    def k_eval(self):
        return self._k_eval
    
    @property
    def z(self):
        return self._z
    
    @property
    def kmin(self):
        return self._kmin

    @property
    def kmax(self):
        return self._kmax

    @property
    def num_threads(self):
        return self._num_threads
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Optimally update the cosmology, basically rescaling integrals if a 
        new sigma_8 is given and deleting otherwise
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
                
            # now actually update the Cosmology class and Power class
            cpdict.update(true_cp)
            self._cosmo = Cosmology(**cpdict)
            self._power.update(**cpdict)
            
            # delete everything if anything other than sigma_8 changed
            ckeys = true_cp.keys()
            if len(ckeys) > 1 or (len(ckeys) == 1 and ckeys[0] != 'sigma_8'):
                for k in self.__dict__.keys():
                    if k.startswith("_Integrals"): del self.__dict__[k]
            
        # now do any other parameters
        for key, val in kwargs.iteritems():
            if "_" + key not in self.__dict__:
                if not hasattr(self._power, key):
                    print "WARNING: %s is not a valid parameter for the %s class" %(str(key), self.__class__.__name__)
                else:
                    self._power.update(**{key:val})
            else:
                if np.any(getattr(self, key) != val):
                    setattr(self, '_' + key, val)
                            
            # now do the deletions
            if key == 'z':
                del self.D
            else:
                for k in self.__dict__.keys():
                    if k.startswith("_Integrals"): del self.__dict__[k]
    #end update
    #---------------------------------------------------------------------------
    def J(self, kernel_name):
        """
        Return the value of the integral J_nm(k), at the specified order, 
        with the correct redshift dependence.
        """
        name = '_Integrals__J_%s' %kernel_name

        # linear order 
        try:
            J1 = getattr(self, "%s_1" %name)
        except:
            self.J_lin.kernel_name = kernel_name
            J1 = self.J_lin.evaluate(self.k_eval)
            self.__dict__["%s_1" %name] = J1
            
        power_norm = self._power.power_norm/NORM_FACTOR
        return (self.D**2 * power_norm)*J1
    
    #---------------------------------------------------------------------------
    def I(self, kernel_name, order, variables=None):
        """
        Return the value of the integral I_nm(k), at the specified order, 
        with the correct redshift dependence.
        """
        name = '_Integrals__I_%s' %kernel_name
        power_norm = self._power.power_norm/NORM_FACTOR
        
        # linear order 
        try:
            I1 = getattr(self, "%s_1" %name)
        except:
            self.I_lin.kernel_name = kernel_name
            I1 = self.I_lin.evaluate(self.k_eval)
            self.__dict__["%s_1" %name] = I1
            
        # return I1 if we are doing linear order
        if order == 0:
            return (self.D**2 * power_norm)**2 * I1
            
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
        
            fac = (self.D**2 * power_norm)
            return  fac**2 * (I1 + fac*(I2 + I3) +  fac**2*I4)
    #---------------------------------------------------------------------------
    @property
    def vel_kurtosis(self):
        """
        Compute the velocity kurtosis <v_parallel^4>, using the 1-loop divergence
        auto spectra Pvv.
        """
        # compute P22_bar first (using linear power spectra)
        I23_1, I32_1, I33_1 = self.P22bar
        
        I1 = I23_1 + (2./3)*I32_1 + (1./5)*I33_1
        power_norm = self._power.power_norm/NORM_FACTOR
        try:
            kurtosis = self.__kurtosis
        except:
            J = _integral_base.integral1D('g22', self.kmin_1loop, self.kmax_1loop, 
                                          self.k1loop, I1)
            
            # evaluate at any k (should all be same)  
            k = np.array([0.1])
            self.__kurtosis = kurtosis = (J.evaluate(k)[0]/k**2)[0]
        
        return  0.25 * (self.D**2 * power_norm)**2 * kurtosis
    #---------------------------------------------------------------------------
    @property
    def sigmasq_k(self):
        """
        This is sigma^2 as a function of k. 
        """
        power_norm = self._power.power_norm/NORM_FACTOR
        try:
            return (self.D**2 * power_norm) * self.__sigmasq_k
        except:
            J = _integral_base.integral1D('unity', self.kmin_1loop, self.kmax_1loop, 
                                          self.klin, self._power._unnormalized_P*NORM_FACTOR)
            
            sigmasq_k = []
            # evaluate at any k
            for k in self.k_eval:
                J.kmax = 0.5*k # integrate up to 0.5*k
                sigmasq_k.append(J.evaluate(np.array([k]))[0])
            self.__sigmasq_k = 1./3*np.array(sigmasq_k)
            
            return (self.D**2 * power_norm) * self.__sigmasq_k
    #---------------------------------------------------------------------------
    @property
    def sigmasq(self):
        """
        This is sigma^2 at z = 0. 
        """
        power_norm = self._power.power_norm/NORM_FACTOR
        try:
            return power_norm*self.__sigmasq
        except:
            J = _integral_base.integral1D('unity', self.kmin_1loop, self.kmax_1loop, 
                                          self.klin, self._power._unnormalized_P*NORM_FACTOR)
            # evaluate at any k (should all be same)  
            k = np.array([0.1])
            self.__sigmasq = 1./3.*J.evaluate(k)[0]
            
            return power_norm*self.__sigmasq
    #---------------------------------------------------------------------------
    # INTEGRAL BASE CLASSES FOR EACH SPECTRA COMBO
    #---------------------------------------------------------------------------
    @property
    def I_lin(self):
        try:
            return self.__I_lin
        except:
            self.__I_lin = _integral_base.integral2D('f00', self.kmin_lin, self.kmax_lin, 
                                                    self.num_threads, 
                                                    self.klin, 
                                                    self._power._unnormalized_P*NORM_FACTOR,
                                                    k2=self.klin, 
                                                    P2=self._power._unnormalized_P*NORM_FACTOR)
            return self.__I_lin
    #----------------------------------------------------------------------------
    @property
    def I_lin_vv1loop(self):
        try:
            return self.__I_lin_vv1loop
        except:
            self.__I_lin_vv1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.klin, 
                                                            self._power._unnormalized_P*NORM_FACTOR,
                                                            k2=self.k1loop, 
                                                            P2=self.Pvv_1loop)
            return self.__I_lin_vv1loop
    #----------------------------------------------------------------------------
    @property
    def I_vv1loop(self):
        try:
            return self.__I_vv1loop
        except:
            self.__I_vv1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                        self.num_threads, 
                                                        self.k1loop, 
                                                        self.Pvv_1loop,
                                                        k2=self.k1loop, 
                                                        P2=self.Pvv_1loop)
            return self.__I_vv1loop
    #---------------------------------------------------------------------------
    @property
    def I_lin_dd1loop(self):
        try:
            return self.__I_lin_dd1loop
        except:
            self.__I_lin_dd1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.klin, 
                                                            self._power._unnormalized_P*NORM_FACTOR,
                                                            k2=self.k1loop, 
                                                            P2=self.Pdd_1loop)
            return self.__I_lin_dd1loop
    #---------------------------------------------------------------------------
    @property
    def I_dd1loop(self):
        try:
            return self.__I_dd1loop
        except:
            self.__I_dd1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.k1loop, 
                                                            self.Pdd_1loop,
                                                            k2=self.k1loop, 
                                                            P2=self.Pdd_1loop)
            return self.__I_dd1loop
    #---------------------------------------------------------------------------
    @property
    def I_dd1loop_vv1loop(self):
        try:
            return self.__I_dd1loop_vv1loop
        except:
            self.__I_dd1loop_vv1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                                self.num_threads, 
                                                                self.k1loop, 
                                                                self.Pdd_1loop,
                                                                k2=self.k1loop, 
                                                                P2=self.Pvv_1loop)
            return self.__I_dd1loop_vv1loop
    #---------------------------------------------------------------------------
    @property
    def I_lin_dv1loop(self):
        try:
            return self.__I_lin_dv1loop
        except:
            self.__I_lin_dv1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                            self.num_threads, 
                                                            self.klin, 
                                                            self._power._unnormalized_P*NORM_FACTOR,
                                                            k2=self.k1loop, 
                                                            P2=self.Pdv_1loop)
            return self.__I_lin_dv1loop
    #---------------------------------------------------------------------------
    @property
    def I_dv1loop(self):
        try:
            return self.__I_dv1loop
        except:
            self.__I_dv1loop = _integral_base.integral2D('f00', self.kmin_1loop, self.kmax_1loop, 
                                                        self.num_threads, 
                                                        self.k1loop, 
                                                        self.Pdv_1loop,
                                                        k2=self.k1loop, 
                                                        P2=self.Pdv_1loop)
            return self.__I_dv1loop
    #---------------------------------------------------------------------------
    @property
    def J_lin(self):
        try:
            return self.__J_lin
        except:
            self.__J_lin = _integral_base.integral1D('g00', self.kmin_lin, self.kmax_lin, 
                                                    self.klin, self._power._unnormalized_P*NORM_FACTOR)
            return self.__J_lin
    
    #---------------------------------------------------------------------------
    # WAVENUMBERS FOR LINEAR/1-LOOP ORDER
    #---------------------------------------------------------------------------
    @property
    def klin(self):
        try:
            return self.__klin
        except:
            # must have wide enough k region to converge
            kmin = self.kmin/(CONVERGENCE_FACTOR)**2
            kmax = self.kmax*(CONVERGENCE_FACTOR)**2
            
            self.__klin = np.logspace(np.log10(kmin), np.log10(kmax), 1000)
            return self.__klin
    #---------------------------------------------------------------------------
    @property
    def k1loop(self):
        try:
            return self.__k1loop
        except:
            # must have wide enough k region to converge
            kmin = self.kmin/(CONVERGENCE_FACTOR)
            kmax = self.kmax*(CONVERGENCE_FACTOR)
            
            self.__k1loop = np.logspace(np.log10(kmin), np.log10(kmax), 200)
            return self.__k1loop
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
            return self.__D
        except:
            self.__D = growth.growth_function(self.z, normed=True, params=self._cosmo)
            return self.__D
    
    @D.deleter
    def D(self):
        try:
            del self.__D
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    # THE POWER SPECTRA
    #---------------------------------------------------------------------------
    @property
    def Plin(self):
        """
        The linear power spectrum evaluatated at klin
        """
        return self.D**2 * self._power.power_norm*self._power._unnormalized_P
    #---------------------------------------------------------------------------
    @property
    def P11(self):
        """
        For internal use.
        """
        try:
            return self.__P11
        except:
            # compute the linear power at k1loop
            self._power.k = self.k1loop
            self.__P11 = self._power._unnormalized_P*NORM_FACTOR
            
            # reset the to correct k values
            self._power.k = self.klin
            
            return self.__P11
    #---------------------------------------------------------------------------
    @property
    def Pdd_1loop(self):
        """
        The 1-loop component of the auto-correlation of density.
        """
        try:
            return self.__Pdd_1loop
        except:
            # compute I00
            self.I_lin.kernel_name = 'f00'
            I00 = self.I_lin.evaluate(self.k1loop)

            # compute J00
            self.J_lin.kernel_name = 'g00'
            J00 = self.J_lin.evaluate(self.k1loop)

            P22 = 2.*I00
            P13 = 6.*self.k1loop**2*J00*self.P11
            self.__Pdd_1loop = P22 + P13
            
            return self.__Pdd_1loop
    #---------------------------------------------------------------------------
    @property
    def Pdv_1loop(self):
        """
        The 1-loop component of the cross-correlation of density and velocity 
        divergence.
        """
        try:
            return self.__Pdv_1loop
        except:       
            # compute I01
            self.I_lin.kernel_name = 'f01'
            I01 = self.I_lin.evaluate(self.k1loop)

            # compute J01
            self.J_lin.kernel_name = 'g01'
            J01 = self.J_lin.evaluate(self.k1loop)

            P22  = 2.*I01
            P13  = 6.*self.k1loop**2*J01*self.P11
            self.__Pdv_1loop = P22 + P13
            
            return self.__Pdv_1loop
    #---------------------------------------------------------------------------
    @property
    def Pvv_1loop(self):
        """
        The 1-loop component of the auto-correlation of velocity divergence.
        """
        try:
            return self.__Pvv_1loop
        except:     
            # compute I11
            self.I_lin.kernel_name = 'f11'
            I11 = self.I_lin.evaluate(self.k1loop)

            # compute J11
            self.J_lin.kernel_name = 'g11'
            J11 = self.J_lin.evaluate(self.k1loop)

            P22  = 2.*I11
            P13  = 6.*self.k1loop**2*J11*self.P11
            self.__Pvv_1loop = P22 + P13
            
            return self.__Pvv_1loop
    #---------------------------------------------------------------------------
    @property
    def P22bar(self):
        """
        The linear order P22 bar components
        """
        try:
            I1 = self.__P22bar1_mu0
        except:
            self.I_lin.kernel_name = 'f23'
            self.__P22bar1_mu0 = I1 = self.I_lin.evaluate(self.k1loop)
        
        try:
            I2 = self.__P22bar1_mu2
        except:
            self.I_lin.kernel_name = 'f32'
            self.__P22bar1_mu2 = I2 = self.I_lin.evaluate(self.k1loop) 
            
        try:
            I3 = self.__P22bar1_mu4
        except:
            self.I_lin.kernel_name = 'f33'
            self.__P22bar1_mu4 = I3 = self.I_lin.evaluate(self.k1loop)
            
        return I1, I2, I3          
    #---------------------------------------------------------------------------
    @property
    def K00(self):
        """
        The nonlinear biasing term showing up in P00hh
        """
        try:
            return self.__K00
        except:
            self.I_lin.kernel_name = 'k00'
            self.__K00 = self.I_lin.evaluate(self.k1loop)
            return self.__K00
    #---------------------------------------------------------------------------
    @property
    def K00s(self):
        """
        The tidal nonlinear biasing term showing up in P00hh
        """
        try:
            return self.__K00s
        except:
            self.I_lin.kernel_name = 'k00s'
            self.__K00s = self.I_lin.evaluate(self.k1loop)
            return self.__K00s
    #---------------------------------------------------------------------------
