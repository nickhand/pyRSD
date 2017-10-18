from six import add_metaclass
from .base import Schema, fixed, free

@add_metaclass(Schema)
class DefaultQuasarPowerTheory(object):
    """
    A class defining the default parameters for the
    :class:`pyRSD.rsd.QuasarSpectrum` model
    """
    model_kwargs = {'z' : None,
                    'cosmo_filename' : None,
                    'include_2loop' : False,
                    'transfer_fit' : 'CLASS',
                    'vel_disp_from_sims' : False,
                    'use_mean_bias' : False,
                    'fog_model' : 'gaussian',
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
                    'interpolate' : True
                    }

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
    def b1(self):
        """
        The linear bias of the quasar
        """
        return {'fiducial':2.0, 'prior':'uniform', 'lower':1.2, 'upper':6.0}

    @free(model_param=True)
    def sigma_fog(self):
        """
        The Finger-of-God velocity disperion in Mpc/h
        """
        return {'fiducial':4., 'prior':'uniform', 'lower':0., 'upper':20.}

    @free(model_param=True)
    def f_nl(self):
        """
        The primordial non-Gaussianity amplitude
        """
        return {'fiducial':0., 'prior':'uniform', 'lower':-500., 'upper':500.}

    @fixed(model_param=True)
    def N(self):
        """
        A constant shot noise offset to the model, in (Mpc/h)^3
        """
        return {'fiducial':0., 'prior':'uniform', 'lower':-500., 'upper':500.}

    @fixed(model_param=True)
    def p(self):
        """
        The bias type for PNG; either 1.6 or 1 usually
        """
        return {'fiducial':1.6}

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
