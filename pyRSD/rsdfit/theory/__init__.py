from .gal_defaults import DefaultGalaxyPowerTheory
from .qso_defaults import DefaultQuasarPowerTheory
from .base import BasePowerParameters, BasePowerTheory

from pyRSD.rsd import GalaxySpectrum, QuasarSpectrum

class GalaxyPowerParameters(BasePowerParameters):
    """
    A ParameterSet for :class:`pyRSD.rsd.GalaxySpectrum`
    """
    defaults = DefaultGalaxyPowerTheory()
    _model_cls = GalaxySpectrum

class GalaxyPowerTheory(BasePowerTheory):
    """
    A class representing a theory for computing a redshift-space
    galaxy power spectrum.

    It handles the dependencies between model parameters and the
    evaluation of the model itself.
    """
    def __init__(self, param_file, model=None, extra_param_file=None, kmin=None, kmax=None):
        """
        Parameters
        ----------
        param_file : str
            name of the file holding the parameters for the theory
        extra_param_file : str
            name of the file holding the names of any extra parameter files
        model : subclass of DarkMatterSpectrum, optional
            the model instance; if not provided, a new model
            will be initialized
        kmin : float, optional
            If not `None`, initalize the model with this `kmin` value
        kmax : float, optional
            If not `None`, initalize the model with this `kmax` value
        """
        args = (GalaxySpectrum, GalaxyPowerParameters, param_file)
        kws = {'extra_param_file':extra_param_file, 'kmin':kmin, 'kmax':kmax, 'model':model}
        super(GalaxyPowerTheory, self).__init__(*args, **kws)

class QuasarPowerParameters(BasePowerParameters):
    """
    A ParameterSet for :class:`pyRSD.rsd.QuasarSpectrum`
    """
    defaults = DefaultQuasarPowerTheory()
    _model_cls = QuasarSpectrum

class QuasarPowerTheory(BasePowerTheory):
    """
    A class representing a theory for computing a redshift-space
    quasar power spectrum.

    It handles the dependencies between model parameters and the
    evaluation of the model itself.
    """
    def __init__(self, param_file, model=None, extra_param_file=None, kmin=None, kmax=None):
        """
        Parameters
        ----------
        param_file : str
            name of the file holding the parameters for the theory
        extra_param_file : str
            name of the file holding the names of any extra parameter files
        model : subclass of DarkMatterSpectrum, optional
            the model instance; if not provided, a new model
            will be initialized
        kmin : float, optional
            If not `None`, initalize the model with this `kmin` value
        kmax : float, optional
            If not `None`, initalize the model with this `kmax` value
        """
        args = (QuasarSpectrum, QuasarPowerParameters, param_file)
        kws = {'extra_param_file':extra_param_file, 'kmin':kmin, 'kmax':kmax, 'model':model}
        super(QuasarPowerTheory, self).__init__(*args, **kws)
