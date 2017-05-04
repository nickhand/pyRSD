from pyRSD.rsd.power.gradient import PkmuDerivative

class PqsoDerivative(PkmuDerivative):
    """
    Abstract base class for derivatives of the `QuasarPowerSpectrum` model
    """
    pass

from .alpha_par  import dPqso_dalpha_par
from .alpha_perp import dPqso_dalpha_perp
from .b1         import dPqso_db1
from .f          import dPqso_df
from .sigma_fog  import dPqso_dsigma_fog
from .sigma8_z   import dPqso_dsigma8_z
from .f_nl       import dPqso_df_nl


__all__ = ['dPqso_dalpha_par',
            'dPqso_dalpha_perp',
            'dPqso_db1',
            'dPqso_df',
            'dPqso_dsigma_fog',
            'dPqso_dsigma8_z',
            'dPqso_df_nl'
            ]
