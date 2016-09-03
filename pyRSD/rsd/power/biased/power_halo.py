from pyRSD.rsd._cache import cached_property
from .power_biased import BiasedSpectrum

class HaloSpectrum(BiasedSpectrum):
    """
    The power spectrum of halos with linear bias `b1` in redshift space
    """
    def __init__(self, **kwargs):
        super(HaloSpectrum, self).__init__(**kwargs)

    @cached_property("b1")
    def b1_bar(self):
        """
        The linear bias factor.
        """
        return self.b1

    @cached_property("bs")
    def bs_bar(self):
        """
        The quadratic, nonlocal tidal bias factor
        """
        return self.bs