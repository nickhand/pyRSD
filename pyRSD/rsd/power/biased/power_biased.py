from __future__ import print_function

import contextlib
from pyRSD import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# tools
from pyRSD.rsd._cache import parameter, cached_property, interpolated_function, CachedProperty
from pyRSD.rsd.tools import BiasToSigmaRelation

# base model
from pyRSD.rsd import DarkMatterSpectrum

# simulation fits
from pyRSD.rsd.simulation import VelocityDispersionFits
from pyRSD.rsd.simulation import Pmu2ResidualCorrection
from pyRSD.rsd.simulation import Pmu4ResidualCorrection
from pyRSD.rsd.simulation import AutoStochasticityFits
from pyRSD.rsd.simulation import CrossStochasticityFits

# nonliner biasing
from pyRSD.rsd.nonlinear_biasing import NonlinearBiasingMixin

GP_NK = 20

class BiasedSpectrum(DarkMatterSpectrum, NonlinearBiasingMixin):
    """
    The power spectrum of two biased tracers, with linear biases `b1`
    and `b1_bar` in redshift space
    """
    def __init__(self, use_tidal_bias=False,
                       use_mean_bias=False,
                       vel_disp_from_sims=False,
                       correct_mu2=False,
                       correct_mu4=False,
                       use_vlah_biasing=True,
                       **kwargs):

        # initalize the dark matter power spectrum
        super(BiasedSpectrum, self).__init__(**kwargs)

        # the nonlinear biasing class
        NonlinearBiasingMixin.__init__(self)

        # set the default parameters
        self.use_mean_bias      = use_mean_bias
        self.vel_disp_from_sims = vel_disp_from_sims
        self.use_tidal_bias     = use_tidal_bias
        self.include_2loop      = False # don't violate galilean invariance, fool
        self.b1                 = 2.

        # correction models
        self.correct_mu2 = correct_mu2
        self.correct_mu4 = correct_mu4

        # whether to use Vlah et al nonlinear biasing
        self.use_vlah_biasing = use_vlah_biasing

        # set b1_bar, unless we are fixed
        try: self.b1_bar = 2.
        except: pass

    def __getstate__(self):

        # delete these since the data is large
        del self.auto_stochasticity_fits
        del self.cross_stochasticity_fits
        return super(BiasedSpectrum, self).__getstate__()

    #---------------------------------------------------------------------------
    # attributes
    #---------------------------------------------------------------------------
    @contextlib.contextmanager
    def cache_override(self, b2_00=None, b2_01=None, **kws):
        """
        Context manager to handle overriding the cache for specific attributes
        """
        self._cache_overrides = {}

        if b2_00 is not None:
            assert np.isscalar(b2_00)
            props = ['b2_00_a', 'b2_00_b', 'b2_00_c', 'b2_00_d']
            for prop in props:
                self._cache_overrides[prop] = lambda b1: b2_00

        if b2_01 is not None:
            assert np.isscalar(b2_01)
            props = ['b2_01_a', 'b2_01_b']
            for prop in props:
                self._cache_overrides[prop] = lambda b1: b2_01

        for k in kws:
            prop = getattr(self.__class__, k, None)
            if prop is None or not getattr(prop, '__cache__', False):
                raise ValueError("'%s' is not a valid cached property" %k)
            self._cache_overrides[k] = kws[k]

        yield

        del self._cache_overrides

    @parameter
    def correct_mu2(self, val):
        """
        Whether to correct the halo P[mu2] model, using a sim-calibrated model
        """
        return val

    @parameter
    def correct_mu4(self, val):
        """
        Whether to correct the halo P[mu4] model, using a sim-calibrated model
        """
        return val

    @parameter
    def use_mean_bias(self, val):
        """
        Evaluate cross-spectra using the geometric mean of the biases
        """
        return val

    @parameter
    def z(self, val):
        """
        Redshift to evaluate power spectrum at
        """
        # update the dependencies
        models = ['P11_sim_model', 'Pdv_sim_model', 'bias_to_sigma_relation']
        self._update_models('z', models, val)

        # update redshift params
        if len(self.redshift_params):
            if 'f' in self.redshift_params:
                self.f = self.cosmo.f_z(val)
            if 'sigma8_z' in self.redshift_params:
                self.sigma8_z = self.cosmo.Sigma8_z(val)
                
        return val

    @parameter
    def vel_disp_from_sims(self, val):
        """
        Whether to use the linear velocity dispersion for halos as
        computed from simulations, which includes a slight mass
        dependence
        """
        return val

    @parameter
    def use_tidal_bias(self, val):
        """
        Whether to include the tidal bias terms
        """
        return val

    @parameter
    def use_Phm_model(self, val):
        """
        If `True`, use the `HZPT` model for Phm
        """
        return val

    @parameter
    def b1(self, val):
        """
        The linear bias factor of the first tracer.
        """
        return val

    @parameter
    def b1_bar(self, val):
        """
        The linear bias factor of the 2nd tracer.
        """
        return val

    #---------------------------------------------------------------------------
    # simulation-calibrated model parameters
    #---------------------------------------------------------------------------
    @cached_property()
    def vel_disp_fitter(self):
        """
        Interpolator from simulation data for linear velocity dispersion of halos
        """
        return VelocityDispersionFits()

    @cached_property()
    def Pmu2_correction(self):
        """
        The parameters for the halo P[mu2] correction
        """
        return Pmu2ResidualCorrection()

    @cached_property()
    def Pmu4_correction(self):
        """
        The parameters for the halo P[mu4] correction
        """
        return Pmu4ResidualCorrection()

    @cached_property()
    def auto_stochasticity_fits(self):
        """
        The prediction for the auto stochasticity from a GP
        """
        return AutoStochasticityFits()

    @cached_property()
    def cross_stochasticity_fits(self):
        """
        The prediction for the cross stochasticity from a GP
        """
        return CrossStochasticityFits()

    #---------------------------------------------------------------------------
    # cached properties
    #---------------------------------------------------------------------------
    @cached_property("use_mean_bias", "b1", "b1_bar")
    def _ib1(self):
        """
        The internally used bias of the 1st tracer
        """
        if not self.use_mean_bias:
            return self.b1
        else:
            return (self.b1*self.b1_bar)**0.5

    @cached_property("use_mean_bias", "b1", "b1_bar")
    def _ib1_bar(self):
        """
        The internally used bias of the first tracer
        """
        if not self.use_mean_bias:
            return self.b1_bar
        else:
            return (self.b1*self.b1_bar)**0.5

    @cached_property("_ib1", "use_tidal_bias")
    def bs(self):
        """
        The quadratic, nonlocal tidal bias factor for the first tracer
        """
        if self.use_tidal_bias:
            return -2./7 * (self._ib1 - 1.)
        else:
            return 0.

    @cached_property("_ib1_bar", "use_tidal_bias")
    def bs_bar(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        if self.use_tidal_bias:
            return -2./7 * (self._ib1_bar - 1.)
        else:
            return 0.

    @cached_property()
    def bias_to_sigma_relation(self):
        """
        The relationship between bias and velocity dispersion, using the
        Tinker et al. relation for halo mass and bias, and the virial theorem
        scaling between mass and velocity dispersion
        """
        return BiasToSigmaRelation(self.z, self.cosmo, interpolate=self.interpolate)

    @cached_property("sigma_v", "sigma_lin", "vel_disp_from_sims", "_ib1", "sigma8_z")
    def sigmav_halo(self):
        """
        The velocity dispersion for halos, possibly as a function of bias
        """
        if self.vel_disp_from_sims:
            return self.vel_disp_fitter(b1=self._ib1, sigma8_z=self.sigma8_z)
        else:
            return self.sigma_v

    @cached_property("sigma_v", "sigma_lin", "vel_disp_from_sims", "_ib1_bar", "sigma8_z")
    def sigmav_halo_bar(self):
        """
        The velocity dispersion for halos, possibly as a function of bias
        """
        if self.vel_disp_from_sims:
            return self.vel_disp_fitter(b1=self._ib1_bar, sigma8_z=self.sigma8_z)
        else:
            return self.sigma_v

    def sigmav_from_bias(self, s8_z, bias):
        """
        Return the velocity dispersion `sigmav` value for the specified linear
        bias value
        """
        try:
            return self.bias_to_sigma_relation(s8_z, bias)
        except Exception as e:
            msg = "Warning: error in computing sigmav from (s8_z, b1) = (%.2f, %.2f); original msg = %s" %(s8_z, bias, e)
            b1s = self.bias_to_sigma_relation.interpolation_grid['b1']
            if bias < np.amin(b1s):
                toret = self.bias_to_sigma_relation(s8_z, np.amin(b1s))
            elif bias > np.amax(b1s):
                toret = self.bias_to_sigma_relation(s8_z, np.amax(b1s))
            else:
                toret = 0.
            print(msg + "; returning sigmav = %.4e" %toret)
            return toret

    #---------------------------------------------------------------------------
    # power term attributes
    #---------------------------------------------------------------------------
    @interpolated_function("_ib1", "P00", "use_Phm_model", "sigma8_z", "b2_00_a", "k", interp="k")
    def Phm(self, k):
        """
        The halo - matter cross correlation for the 1st tracer
        """
        if self.use_Phm_model:
            toret = self.hzpt.Phm(b1=self._ib1, k=k)
        else:
            # the bias values to use
            b1, b2_00 = self._ib1, self.b2_00_a(self._ib1)

            term1 = b1*self.P00.mu0(k)
            term2 = b2_00*self.K00(k)
            term3 = self.bs*self.K00s(k)
            toret = term1 + term2 + term3

        return toret

    @interpolated_function("_ib1_bar", "P00", "use_Phm_model", "sigma8_z", "b2_00_a", "k", interp="k")
    def Phm_bar(self, k):
        """
        The halo - matter cross correlation for the 2nd tracer
        """
        if self.use_Phm_model:
            toret = self.hzpt.Phm(b1=self._ib1_bar, k=k)
        else:
            # the bias values to use
            b1_bar, b2_00_bar = self._ib1_bar, self.b2_00_a(self._ib1_bar)

            term1 = b1_bar*self.P00.mu0(k)
            term2 = b2_00_bar*self.K00(k)
            term3 = self.bs_bar*self.K00s(k)
            toret = term1 + term2 + term3

        return toret

    @interpolated_function("_ib1", "_ib1_bar", "z", "sigma8_z", "k", interp="k")
    def stochasticity(self, k):
        """
        The isotropic (type B) stochasticity term due to the discreteness of the
        halos, i.e., Poisson noise at 1st order.

        Notes
        -----
        *   The model for the (type B) stochasticity, interpolated as a function
            of sigma8(z), b1, and k using a Gaussian process
        """
        _k = np.logspace(np.log10(self.k.min()), np.log10(self.k.max()), GP_NK)

        params = {'sigma8_z' : self.sigma8_z, 'k':_k}
        if self._ib1 != self._ib1_bar:
            b1_1, b1_2 = sorted([self._ib1, self._ib1_bar])
            toret = self.cross_stochasticity_fits(b1_1=b1_1, b1_2=b1_2, **params)
        else:
            toret = self.auto_stochasticity_fits(b1=self._ib1, **params)

        return spline(_k, toret)(k)

    @cached_property("P00_ss_no_stoch", "stochasticity")
    def P00_ss(self):
        """
        The isotropic, halo-halo power spectrum, including the (possibly
        k-dependent) stochasticity term, as specified by the user.
        """
        from .P00 import P00PowerTerm
        return P00PowerTerm(self)

    @cached_property("P00", "Phm", "Phm_bar")
    def P00_ss_no_stoch(self):
        """
        The isotropic, halo-halo power spectrum, without any stochasticity term.
        """
        from .P00 import NoStochP00PowerTerm
        return NoStochP00PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "max_mu", "Pdv", "P01", "b2_01_a")
    def P01_ss(self):
        """
        The correlation of the halo density and halo momentum fields, which
        contributes mu^2 terms to the power expansion.
        """
        from .P01 import P01PowerTerm
        return P01PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "max_mu", "P02", "P00",
                     "sigmav_halo", "sigmav_halo_bar", "b2_00_b", "b2_00_c")
    def P02_ss(self):
        """
        The correlation of the halo density and halo kinetic energy, which
        contributes mu^2 and mu^4 terms to the power expansion.
        """
        from .P02 import P02PowerTerm
        return P02PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "max_mu", "P11", "Pvv")
    def P11_ss(self):
        """
        The auto-correlation of the halo momentum field, which
        contributes mu^2 and mu^4 terms to the power expansion. The mu^2 terms
        come from the vector part, while the mu^4 dependence is dominated by
        the scalar part on large scales (linear term) and the vector part
        on small scales.
        """
        from .P11 import P11PowerTerm
        return P11PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "max_mu", "Pdv", "P01", "sigmav_halo", "sigmav_halo_bar", "b2_01_b")
    def P03_ss(self):
        """
        The cross-corelation of halo density with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 terms.
        """
        from .P03 import P03PowerTerm
        return P03PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "max_mu", "Pdv", "P01", "P12", "sigmav_halo", "sigmav_halo_bar", "b2_01_b")
    def P12_ss(self):
        """
        The correlation of halo momentum and halo kinetic energy density, which
        contributes mu^4 and mu^6 terms to the power expansion.
        """
        from .P12 import P12PowerTerm
        return P12PowerTerm(self)

    @cached_property("P11_ss", "sigmav_halo", "sigmav_halo_bar")
    def P13_ss(self):
        """
        The cross-correlation of halo momentum with the rank three tensor field
        ((1+delta_h)v)^3, which contributes mu^4 and mu^6 terms.
        """
        from .P13 import P13PowerTerm
        return P13PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "P00", "P02", "P22", "sigmav_halo", "sigmav_halo_bar", "b2_00_d")
    def P22_ss(self):
        """
        The auto-corelation of halo kinetic energy density, which contributes
        mu^4, mu^6, mu^8 terms to the power expansion. There are no linear
        contributions here.
        """
        from .P22 import P22PowerTerm
        return P22PowerTerm(self)

    @cached_property("_ib1", "_ib1_bar", "P00", "P02", "sigmav_halo", "sigmav_halo_bar", "b2_00_d")
    def P04_ss(self):
        """
        The cross-correlation of halo density with the rank four tensor field
        ((1+delta_h)v)^4, which contributes mu^4 and mu^6 terms, at 2-loop order.
        """
        from .P04 import P04PowerTerm
        return P04PowerTerm(self)

    @interpolated_function("_ib1", "_ib1_bar", "sigma8_z", "f", "k", interp="k")
    def mu2_model_correction(self, k):
        """
        The mu2 correction to the model evaluated at `k`
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        mean_bias = (b1*b1_bar)**0.5
        params = {'b1':mean_bias, 'sigma8_z':self.sigma8_z, 'k':k, 'f':self.f}
        return self.Pmu2_correction(**params)

    @interpolated_function("_ib1", "_ib1_bar", "sigma8_z", "f", "k", interp="k")
    def mu4_model_correction(self, k):
        """
        The mu4 correction to the model evaluated at `k`
        """
        b1, b1_bar = self._ib1, self._ib1_bar
        mean_bias = (b1*b1_bar)**0.5
        params = {'b1':mean_bias, 'sigma8_z':self.sigma8_z, 'k':k, 'f':self.f}
        return self.Pmu4_correction(**params)

    #---------------------------------------------------------------------------
    # power as a function of mu
    #---------------------------------------------------------------------------
    @interpolated_function("P00_ss", "k", interp="k")
    def P_mu0(self, k):
        """
        The full halo power spectrum term with no angular dependence. Contributions
        from P00_ss.
        """
        return self.P00_ss.mu0(k)

    @interpolated_function("P01_ss", "P11_ss", "P02_ss", "correct_mu2", "k", interp="k")
    def P_mu2(self, k):
        """
        The full halo power spectrum term with mu^2 angular dependence. Contributions
        from P01_ss, P11_ss, and P02_ss.
        """
        P_mu2 = self.P01_ss.mu2(k) + self.P11_ss.mu2(k) + self.P02_ss.mu2(k)
        if self.correct_mu2:
            P_mu2 += self.mu2_model_correction(k)

        return P_mu2

    @interpolated_function("P11_ss", "P02_ss", "P12_ss", "P22_ss", "P03_ss",
                           "P13_ss", "P04_ss", "correct_mu4", "k", interp="k")
    def P_mu4(self, k):
        """
        The full halo power spectrum term with mu^4 angular dependence. Contributions
        from P11_ss, P02_ss, P12_ss, P03_ss, P13_ss, P22_ss, and P04_ss.
        """
        P_mu4 = self.P11_ss.mu4(k) + self.P02_ss.mu4(k) + self.P12_ss.mu4(k) + self.P03_ss.mu4(k) + \
                self.P22_ss.mu4(k) + self.P13_ss.mu4(k) + self.P04_ss.mu4(k)
        if self.correct_mu4:
            P_mu4 += self.mu4_model_correction(k)

        return P_mu4

    @interpolated_function("P12_ss", "k", interp="k")
    def P_mu6(self, k):
        """
        The full halo power spectrum term with mu^6 angular dependence. Contributions
        from P12_ss, P13_ss, P22_ss.
        """
        return self.P12_ss.mu6(k) + 1./8*self.f**4 * self.I32(k)
