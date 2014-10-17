#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
"""
 power.pyx
 pyRSD: a class to provide methods to calculate the transfer function, matter 
        power spectrum and several other related quantities.
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/26/2014
"""
from .cosmo import Cosmology
from . import _camb, _functionator
from pyRSD.cosmology cimport cosmo_tools
import os
import numpy as np


class Power(object):
    """
    A class to compute matter power spectra, with the possiblity of using
    various transfer functions
    """
    fits = ["EH", "EH_no_wiggles", "EH_no_baryons", "BBKS", "Bond_Efs", "CAMB"]
    
    def __init__(self, k=np.logspace(-3, 0, 200), 
                       z=0., 
                       transfer_fit='CAMB', 
                       cosmo={'default':"Planck1_lens_WP_highL", 'flat': True},
                       initial_condition=1, 
                       l_accuracy_boost=1, 
                       accuracy_boost=1,
                       transfer_k_per_logint=11, 
                       transfer_kmax=200.):
                 
        # Set up a simple dictionary of cosmo params which can be later updated
        if isinstance(cosmo, Cosmology):
            self.cosmo = cosmo
        else:
            self.cosmo = Cosmology(**cosmo)
        self._camb_options = {'initial_condition' : initial_condition,
                              'l_accuracy_boost' : l_accuracy_boost,
                              'accuracy_boost' : accuracy_boost,
                              'transfer_k_per_logint' : transfer_k_per_logint,
                              'transfer_kmax' : transfer_kmax}
        
        self.k = k
        self.z = z
        self.transfer_fit = transfer_fit
         
    #end __init__
    
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Update the class optimally with given arguments.
        
        Accepts any argument that the constructor takes
        """
        cpdict = self.cosmo.dict()
        
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
            
            # now actually make a new Cosmology class
            cpdict.update(true_cp)
            self.cosmo = Cosmology(cpdict)

            # the following two parameters don't necessitate a complete recalculation
            if len(true_cp) == 1 and any(k in ['n', 'sigma_8'] for k in true_cp):
                if "n" in true_cp:
                    try: del self._unnormalized_P
                    except AttributeError: pass
                if "sigma_8" in true_cp:
                    try: del self.power_norm
                    except AttributeError: pass
            else:
                # all other parameters mean recalculating everything :(
                del self._unnormalized_T
                del self.growth

        # now do any other parameters
        for key, val in kwargs.iteritems():  # only camb options should be left
            # CAMB OPTIONS
            if key in self._camb_options:
                if self._camb_options[key] != val:
                    self._camb_options.update({key:val})
                    del self._unnormalized_T
            # ANYTHING ELSE
            else:
                if "_Power__" + key not in self.__dict__:
                    print "WARNING: %s is not a valid parameter for the %s class" %(str(key), self.__class__.__name__)
                else:
                    if np.any(getattr(self, key) != val):
                        setattr(self, key, val)  # doing it this way enables value-checking
    #end update
    #---------------------------------------------------------------------------
    # SET PROPERTIES
    #---------------------------------------------------------------------------
    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, val):
        del self._unnormalized_T
        self.__k = val
    #---------------------------------------------------------------------------
    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, val):
        del self.growth
        self.__z = val
    #---------------------------------------------------------------------------
    @property
    def cosmo(self):
        return self.__cosmo
    
    @cosmo.setter
    def cosmo(self, val):
        self.__cosmo = val
         
        transfer_int = -1
        if self.__dict__.has_key('_Power__transfer_int'): transfer_int = self.__transfer_int
        set_parameters(self.cosmo.omegam, self.cosmo.omegab, self.cosmo.omegal, 
                        self.cosmo.omegar, self.cosmo.sigma_8, self.cosmo.h, 
                        self.cosmo.n, self.cosmo.Tcmb, self.cosmo.w, transfer_int)
    #---------------------------------------------------------------------------
    @property
    def transfer_fit(self):
        return self.__transfer_fit
        
    @transfer_fit.setter
    def transfer_fit(self, val):
        if val not in Power.fits:
            raise ValueError("Transfer fit must be one of %s" %Power.fits)
            
        if not os.environ.has_key("CAMB_DIR") and val == "CAMB":
            raise ValueError("Must specify 'CAMB_DIR' environment variable to use 'CAMB' transfer function.")
            
        # delete dependencies
        del self._unnormalized_T
        
        if val != 'CAMB':
            del self._CAMB_transfer
        
        # set the integer value for this transfer fit
        if val == 'EH': 
            self.__transfer_int = 0
        elif val == 'EH_no_wiggles': 
            self.__transfer_int = 1
        elif val == 'EH_no_baryons':
            self.__transfer_int = 2
        elif val == 'BBKS': 
            self.__transfer_int = 3
        elif val == 'Bond_Efs': 
            self.__transfer_int = 4
        elif val == 'CAMB':
            self.__transfer_int = 5
            
        self.__transfer_fit = val
        self._set_transfer()
    #---------------------------------------------------------------------------
    def _set_transfer(self):
        """
        Set the parameters
        """
    
        set_parameters(self.cosmo.omegam, self.cosmo.omegab, self.cosmo.omegal, 
                        self.cosmo.omegar, self.cosmo.sigma_8, self.cosmo.h, 
                        self.cosmo.n, self.cosmo.Tcmb, self.cosmo.w, self.__transfer_int)
        if self.__transfer_fit == "CAMB":
            self._initialize_CAMB_transfer()
        else:
            # free the old transfer spline
            free_transfer()
    #---------------------------------------------------------------------------
    def _initialize_CAMB_transfer(self):
        """
        Initialize the CAMB transfer spline.
        """
        # set up the spline
        cdef np.ndarray xarr, yarr 
        
        # compute the CAMB transfer and intialize
        xarr, yarr = self._CAMB_transfer
        xarr = np.ascontiguousarray(xarr, dtype=np.double)
        yarr = np.ascontiguousarray(yarr, dtype=np.double)
        
        set_CAMB_transfer(<double*>xarr.data, <double*>yarr.data, xarr.shape[0])
        
    #---------------------------------------------------------------------------
    @property
    def _CAMB_transfer(self):
        try:
            return self.__camb_k, self.__camb_T
        except:
            cdict = dict(self.cosmo.camb_dict().items() + self._camb_options.items())
            k, T = _camb.camb_transfer(cdict)
                
            # check high k values
            if np.amax(self.k) > np.amax(k):
                
                # do a power law extrapolation at high k
                p_fit = np.polyfit(np.log(k[-10:]), np.log(T[-10:]), 1)
                p_gamma = p_fit[0]
                p_amp = T[-1]/(k[-1]**p_gamma)
                hik_extrap = _functionator.powerLawExtrapolator(gamma=p_gamma, A=p_amp)
                
                # make the combined transfer function
                k_hi = np.logspace(np.log10(np.amax(k)), np.log10(np.amax(self.k)), 200)[1:]
                T_hi = hik_extrap(k_hi)
                k = np.concatenate((k, k_hi))
                T = np.concatenate((T, T_hi))
                
            # check low k values, too
            if np.amin(self.k) < np.amin(k):
                
                # do a power law extrapolation at low k
                p_fit = np.polyfit(np.log(k[:10]), np.log(T[:10]), 1)
                p_gamma = p_fit[0]
                p_amp = T[0]/(k[0]**p_gamma)
                lok_extrap = _functionator.powerLawExtrapolator(gamma=p_gamma, A=p_amp)
                
                # make the combined transfer function
                k_lo = np.logspace(np.log10(np.amin(self.k)), np.log10(np.amin(k)), 100)[:-1]
                T_lo = lok_extrap(k_lo)
            
                T_lo[np.where(T_lo > 1.0)] = 1.0 # don't go over unity here
                k = np.concatenate((k_lo, k))
                T = np.concatenate((T_lo, T))

            self.__camb_k = k
            self.__camb_T = T
            
            return self.__camb_k, self.__camb_T
    
    @_CAMB_transfer.deleter
    def _CAMB_transfer(self):
        try:
            del self.__camb_k
            del self.__camb_T    
        except AttributeError:
            pass
    
    #----------------------------------------------------------------------------
    @property
    def _unnormalized_T(self):
        """
        The unnormaized transfer function
        
        This wraps the individual transfer_fit methods to provide unified access.
        """
        cdef np.ndarray output, k
        try:
            return self.__unnormalized_T
        except AttributeError:
            
            # reset transfer to be sure
            transfer_type = self.transfer_fit
            self.transfer_fit = transfer_type
            
            # set up C arrays to pass
            output = np.ascontiguousarray(np.empty(len(self.k)), dtype=np.double)
            k = np.ascontiguousarray(self.k, dtype=np.double)
            
            unnormalized_transfer(<double *>k.data, self.z, len(self.k), <double *>output.data)
            self.__unnormalized_T = output
            return self.__unnormalized_T

    @_unnormalized_T.deleter
    def _unnormalized_T(self):
        try:
            del self.__unnormalized_T
        except AttributeError:
            pass
        del self._unnormalized_P
        del self.transfer
        del self.power_norm
    #---------------------------------------------------------------------------
    @property
    def _unnormalized_P(self):
        """
        Unnnormalized power at :math:`z=0` [units :math:`Mpc^3/h^3`]
        """
        try:
            return self.__unnormalized_P
        except AttributeError:
            self.__unnormalized_P = (self.k**self.cosmo.n)*(self._unnormalized_T**2)
            return self.__unnormalized_P

    @_unnormalized_P.deleter
    def _unnormalized_P(self):
        try:
            del self.__unnormalized_P
        except AttributeError:
            pass
        del self._P_0
    #---------------------------------------------------------------------------        
    @property 
    def power_norm(self):
        """
        The power normalization
        """
        try:
            return self._power_norm
        except AttributeError:
            
            # reset transfer to be sure
            transfer_type = self.transfer_fit
            self.transfer_fit = transfer_type
            
            self._power_norm = normalize_power()
            return self._power_norm
    
    @power_norm.deleter
    def power_norm(self):
        try:
            del self._power_norm
        except AttributeError:
            pass
        del self._P_0
    #---------------------------------------------------------------------------
    @property
    def _P_0(self):
        """
        Normalized power at z=0 [units :math:`Mpc^3/h^3`]
        """
        try:
            return self.__P_0
        except:
            self.__P_0 = self.power_norm * self._unnormalized_P
            return self.__P_0
    
    @_P_0.deleter
    def _P_0(self):
        try:
            del self.__P_0
        except AttributeError:
            pass
        del self.power    
    #---------------------------------------------------------------------------
    @property
    def growth(self):
        """
        The growth factor
        """
        cdef np.ndarray output, z
        try:
            return self.__growth
        except:
            if self.z > 0:
                
                # set up C arrays to pass
                output = np.ascontiguousarray(np.empty(1), dtype=np.double)
                z = np.ascontiguousarray(np.array([self.z]), dtype=np.double)
                
                D_plus(<double *>z.data, 1, 1, <double *>output.data)
                self.__growth = output[0]
            else:
                self.__growth = 1.0
            return self.__growth

    @growth.deleter
    def growth(self):
        try:
            del self.__growth
        except:
            pass
        del self.power
    #---------------------------------------------------------------------------
    @property
    def power(self):
        """
        Normalized linear power spectrum [units :math:`Mpc^3/h^3`]
        """
        try:
            return self.__power
        except AttributeError:
            self.__power = self.growth**2 * self._P_0
            return self.__power

    @power.deleter
    def power(self):
        try:
            del self.__power
        except AttributeError:
            pass
        del self.delta_k
    #---------------------------------------------------------------------------
    @property
    def transfer(self):
        """
        Normalised transfer function.
        """
        try:
            return self.__transfer
        except AttributeError:
            self.__transfer = self.power_norm * self._unnormalized_T
            return self.__transfer

    @transfer.deleter
    def transfer(self):
        try:
            del self.__transfer
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    @property
    def delta_k(self):
        r"""
        Dimensionless power spectrum, :math:`\Delta_k = \frac{k^3 P(k)}{2\pi^2}`
        """
        try:
            return self.__delta_k
        except AttributeError:
            self.__delta_k = self.k**3 * self.power / (2.*np.pi**2)
            return self.__delta_k

    @delta_k.deleter
    def delta_k(self):
        try:
            del self.__delta_k
        except AttributeError:
            pass
        del self.nonlinear_power
    #----------------------------------------------------------------------------
    @property
    def nonlinear_power(self):
        """
        Non-linear log power [units :math:`Mpc^3/h^3`]

        Non-linear corrections come from HALOFIT (Smith 2003) with updated
        parameters from Takahashi 2012.       
        """
        cdef np.ndarray output, k, delta_k
        try:
            return self.__nonlinear_power
        except:            
            # set up C arrays to pass
            output = np.ascontiguousarray(np.empty(len(self.k)), dtype=np.double)
            k = np.ascontiguousarray(self.k, dtype=np.double)
            delta_k = np.ascontiguousarray(self.delta_k, dtype=np.double)
            
            # reset transfer to be sure
            transfer_type = self.transfer_fit
            self.transfer_fit = transfer_type
            
            nonlinear_power(<double *>k.data, self.z, len(self.k), 
                                <double *>delta_k.data, <double *>output.data)
            self.__nonlinear_power = output
            return self.__nonlinear_power

    @nonlinear_power.deleter
    def nonlinear_power(self):
        try:
            del self.__nonlinear_power
        except AttributeError:
            pass
    #---------------------------------------------------------------------------
    def sigma_r(self, r_Mpch, z):
        """
        The average mass fluctuation within a sphere of radius r, using the 
        specified transfer function for the linear power spectrum.

        Parameters
        ----------
        r_Mpch : {float, np.ndarray}
            the radius to compute the variance within in units of Mpc/h
        z : {float, np.ndarray}
            the redshift to compute the variance at
        """
        if not np.isscalar(r_Mpch) and not np.isscalar(z):
            raise ValueError("Radius and redshift inputs cannot both be arrays")
            
        cdef np.ndarray rarr, output, Dz, zarr
        
        # make r array-like
        r = cosmo_tools.vectorize(r_Mpch)
        z = cosmo_tools.vectorize(z)
        
        # set up the arrays to pass to the C code
        rarr = np.ascontiguousarray(r, dtype=np.double)
        output = np.ascontiguousarray(np.empty(len(r)), dtype=np.double)

        # reset transfer to be sure
        transfer_type = self.transfer_fit
        self.transfer_fit = transfer_type

        # compute sigma at z = 0, then multiply by the growth function
        unnormalized_sigma_r(<double *>rarr.data, 0., len(r), <double *>output.data)
        
        # the growth function
        Dz = np.ascontiguousarray(np.empty(len(z)), dtype=np.double)
        zarr = np.ascontiguousarray(z, dtype=np.double)
        D_plus(<double *>zarr.data, len(z), 1, <double *>Dz.data)

        return output*np.sqrt(self.power_norm)*Dz
    #---------------------------------------------------------------------------
    def dlnsdlnm(self, r_Mpch, sigma0=None):
        """
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln M}\right|`, ``len=len(r_Mpch)``
        For use in computing halo mass functions.

        Notes
        -----

        .. math:: frac{d\ln\sigma}{d\ln M} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk
        """
        cdef np.ndarray rarr, integral
        
        # vectorize the radius input
        r = cosmo_tools.vectorize(r_Mpch)
        
        # set up the arrays to pass to the C code
        rarr = np.ascontiguousarray(r, dtype=np.double)
        integral = np.ascontiguousarray(np.empty(len(r)), dtype=np.double)

        # reset transfer to be sure
        transfer_type = self.transfer_fit
        self.transfer_fit = transfer_type

        # compute the mass variance if it is not provided
        if sigma0 is None:
            sigma0 = self.sigma_r(r_Mpch, 0.)

        # compute the integral, (unnormalized)
        dlnsdlnm_integral(<double *>rarr.data, len(r), <double *>integral.data)

        # now compute the full derivative
        dlnsdlnm = 3./(2*sigma0**2*np.pi**2*r**4)*integral*self.power_norm

        return dlnsdlnm
#-------------------------------------------------------------------------------
    
    
    