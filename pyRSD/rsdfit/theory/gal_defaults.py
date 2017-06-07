from six import add_metaclass
from .base import Schema, fixed, free

@add_metaclass(Schema)
class DefaultGalaxyPowerTheory(object):
    """
    A class defining the default parameters for the
    :class:`pyRSD.rsd.GalaxySpectrum` model
    """
    model_kwargs = {'z' : None,
                    'cosmo_filename' : None,
                    'include_2loop' : False,
                    'transfer_fit' : 'CLASS',
                    'vel_disp_from_sims' : False,
                    'use_mean_bias' : False,
                    'fog_model' : 'modified_lorentzian',
                    'use_tidal_bias' : False,
                    'use_P00_model' : True,
                    'use_P01_model' : True,
                    'use_P11_model' : True,
                    'use_Phm_model' : True,
                    'use_Pdv_model' : True,
                    'Pdv_model_type' : 'jennings',
                    'correct_mu2' : False,
                    'correct_mu4' : False,
                    'max_mu' : 4,
                    'interpolate' : True,
                    'use_so_correction' : False,
                    'use_vlah_biasing' : True}

    @property
    def model_params(self):
        return sorted(self._model_params)

    @property
    def deprecated(self):
        return sorted(self._deprecated)

    @property
    def extra_params(self):
        return sorted(self._extra_params)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, val):
        return val in self.model_params or val in self.extra_params

    def __iter__(self):
        names = sorted(self._free|self._fixed)
        for name in names:
            yield getattr(self, name)

    def iterfree(self):
        for name in sorted(self._free):
            yield getattr(self, name)

    def iterfixed(self):
        for name in sorted(self._fixed):
            yield getattr(self, name)

    @free(model_param=True)
    def sigma8_z(self):
        """
        The mass variance at r = 8 Mpc/h at `z`
        """
        return {'fiducial':0.61, 'prior':'uniform', 'lower':0.3, 'upper':0.9}

    @free(model_param=True)
    def f(self):
        """
        The growth rate, f = dlnD/dlna at `z`
        """
        return {'fiducial':0.78, 'prior':'uniform', 'lower':0.6, 'upper':1.0}

    @free(model_param=True)
    def alpha_perp(self):
        """
        The Alcock-Paczynski effect parameter for perpendicular to the line-of-sight
        """
        return {'fiducial':1.00, 'prior':'uniform', 'lower':0.8, 'upper':1.2}

    @free(model_param=True)
    def alpha_par(self):
        """
        The Alcock-Paczynski effect parameter for parallel to the line-of-sight
        """
        return {'fiducial':1.00, 'prior':'uniform', 'lower':0.8, 'upper':1.2}

    @fixed(model_param=True)
    def alpha_drag(self):
        """
        The ratio of the sound horizon at `z_drag` in the fiducial and true cosmologies
        """
        return {'fiducial':1.00}

    @free(model_param=True)
    def b1_cA(self):
        """
        The linear bias of type A centrals (no satellites in the same halo)
        """
        return {'fiducial':1.90, 'prior':'uniform', 'lower':1.2, 'upper':2.5}

    @fixed(model_param=True)
    def b1_cB(self):
        """
        The linear bias of type B centrals (1 or more satellite(s) in the same halo)
        """
        expr = "(1-fsB)/(1+fsB*(1./Nsat_mult - 1)) * b1_sA +  (1 - (1-fsB)/(1+fsB*(1./Nsat_mult - 1))) * b1_sB"
        return {'fiducial':2.84, 'expr':expr}

    @fixed(model_param=True)
    def b1_sA(self):
        """
        The linear bias of type A satellites (no other satellites in the same halo)
        """
        return {'fiducial':2.63, 'expr':"gamma_b1sA*b1_cA"}

    @fixed(model_param=True)
    def b1_sB(self):
        """
        The linear bias of type B satellites (1 or more other satellites in the same halo)
        """
        return {'fiducial':3.62, 'expr':"gamma_b1sB*b1_cA"}

    @free(model_param=True)
    def fs(self):
        """
        The satellite fraction
        """
        return {'fiducial':0.104, 'prior':'uniform', 'lower':0., 'upper':0.25, 'min':0., 'max':1.}

    @fixed(model_param=True)
    def fcB(self):
        """
        The fraction of centrals that are classified as type B
        """
        expr = "fs / (1 - fs) * (1 + fsB*(1./Nsat_mult - 1))"
        return {'fiducial':0.089, 'min':0, 'max':1, 'expr':expr}

    @free(model_param=True)
    def fsB(self):
        """
        The fraction of satellites that are classified as type B
        """
        return {'fiducial':0.400, 'prior':'uniform', 'lower':0., 'upper':1., 'min':0., 'max':1}

    @free(model_param=False)
    def Nsat_mult(self):
        """
        The mean number of satellites in halos with >1 sat
        """
        return {'fiducial':2.400, 'prior':'normal', 'mu':2.4, 'sigma':0.2, 'min':2.}

    @free(model_param=True)
    def sigma_c(self):
        """
        The centrals Finger-of-God velocity disperion in Mpc/h
        """
        return {'fiducial':1., 'prior':'uniform', 'lower':0., 'upper':3.}

    @free(model_param=True)
    def sigma_sA(self):
        """
        The type A satellites Finger-of-God velocity disperion in Mpc/h
        """
        return {'fiducial':3.5, 'prior':'uniform', 'lower':2., 'upper':8.}

    @fixed(model_param=True)
    def sigma_sB(self):
        """
        The type B satellites Finger-of-God velocity disperion in Mpc/h
        """
        expr = "sigma_sA * sigmav_from_bias(sigma8_z, b1_sB) / sigmav_from_bias(sigma8_z, b1_sA)"
        return {'fiducial':5.0, 'prior':'uniform', 'lower':3., 'upper':10.}

    @fixed(model_param=True)
    def NcBs(self):
        """
        The amplitude of the 1-halo power for type B centrals and satellites in (Mpc/h)^3
        """
        expr = "f1h_cBs / (fcB*(1 - fs)*nbar)"
        return {'fiducial':4.5e4, 'expr':expr}

    @fixed(model_param=True)
    def NsBsB(self):
        """
        The amplitude of the 1-halo power for type B satellites in (Mpc/h)^3
        """
        expr = "f1h_sBsB / (fsB**2 * fs**2 * nbar) * (fcB*(1 - fs) - fs*(1-fsB))"
        return {'fiducial':9.45e4, 'expr':expr}

    @fixed(model_param=False)
    def nbar(self):
        """
        The number density of the sample in (Mpc/h)^(-3)
        """
        return {'fiducial':3.117e-4}

    @fixed(model_param=True)
    def N(self):
        """
        A constant shot noise offset to the model, in (Mpc/h)^3
        """
        return {'fiducial':0., 'prior':'uniform', 'lower':-500., 'upper':500.}

    @fixed(model_param=True)
    def f_so(self):
        """
        The extra satellite fraction accounting for differences between SO and FoF halos
        """
        return {'fiducial':0.03, 'prior':'normal', 'mu':0.04, 'sigma':0.02}

    @fixed(model_param=True)
    def sigma_so(self):
        """
        The velocity dispersion in Mpc/h accounting for differences between SO and FoF halos
        """
        return {'fiducial':4., 'prior':'uniform', 'lower':1., 'upper':7}

    @free(model_param=False)
    def gamma_b1sA(self):
        """
        The relative fraction of ``b1_sA`` to ``b1_cA``
        """
        return {'fiducial':1.45, 'prior':'normal', 'mu':1.45, 'sigma':0.3, 'min':1.0}

    @free(model_param=False)
    def gamma_b1sB(self):
        """
        The relative fraction of ``b1_sB`` to ``b1_cA``
        """
        return {'fiducial':2.05, 'prior':'normal', 'mu':2.05, 'sigma':0.3, 'min':1.0}

    @fixed(model_param=False)
    def gamma_b1cB(self):
        """
        The relative fraction of ``b1_cB`` to ``b1_cA``
        """
        return {'fiducial':0.4, 'prior':'normal', 'mu':0.4, 'sigma':0.2, 'min':0., 'max':1.}

    @fixed(model_param=False, deprecated=True)
    def delta_sigsB(self):
        """
        The relative fraction of ``sigma_sB`` to ``sigma_s``
        """
        return {'fiducial':1., 'prior':'normal', 'mu':1.0, 'sigma':0.2, 'min':0.}

    @fixed(model_param=False, deprecated=True)
    def delta_sigsA(self):
        """
        The relative fraction of ``sigma_sA`` to ``sigma_s``
        """
        return {'fiducial':1., 'prior':'normal', 'mu':1.0, 'sigma':0.2, 'min':0.}

    @fixed(model_param=False)
    def f1h_cBs(self):
        """
        An order unity amplitude value multiplying the 1-halo term, ``NcBs``
        """
        return {'fiducial':1.0, 'prior':'normal', 'mu':1.0, 'sigma':0.75 , 'min':0}

    @free(model_param=False)
    def f1h_sBsB(self):
        """
        An order unity amplitude value multiplying the 1-halo term, ``NsBsB``
        """
        return {'fiducial':4.0, 'prior':'normal', 'mu':4.0, 'sigma':1.0, 'min':0.}

    @fixed(model_param=False)
    def b1_c(self):
        """
        The linear bias of centrals
        """
        return {'expr' : "(1 - fcB)*b1_cA + fcB*b1_cB"}

    @fixed(model_param=False)
    def b1_s(self):
        """
        The linear bias of satellites
        """
        return {'expr' : "(1 - fsB)*b1_sA + fsB*b1_sB"}

    @fixed(model_param=False)
    def b1(self):
        """
        The total linear bias
        """
        return {'expr' : "(1 - fs)*b1_c + fs*b1_s"}

    @fixed(model_param=False)
    def fsigma8(self):
        """
        The value of f(z)*sigma8(z) at z of measurement
        """
        return {'expr' : "f*sigma8_z"}

    @fixed(model_param=False)
    def b1sigma8(self):
        """
        The value of b1(z)*sigma8(z) at z of measurement
        """
        return {'expr' : "b1*sigma8_z"}

    @fixed(model_param=False)
    def F_AP(self):
        """
        The AP parameter: ``alpha_par/alpha_perp``
        """
        return {'expr' : "alpha_par/alpha_perp"}

    @fixed(model_param=False)
    def alpha(self):
        """
        The isotropic AP dilation
        """
        return {'expr' : "(alpha_perp**2 * alpha_par)**(1./3)"}

    @fixed(model_param=False)
    def epsilon(self):
        """
        The anisotropic AP warping
        """
        return {'expr' : "(alpha_perp/alpha_par)**(-1./3) - 1.0"}
