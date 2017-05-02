from pyRSD.rsd.power.gradient import PkmuDerivative

class PgalDerivative(PkmuDerivative):
    """
    Abstract base class for derivatives of the `GalaxyPowerSpectrum` model
    """
    pass

from .alpha_par  import dPgal_dalpha_par
from .alpha_perp import dPgal_dalpha_perp
from .f_so       import dPgal_df_so
from .fcB        import dPgal_dfcB
from .fs         import dPgal_dfs
from .fsB        import dPgal_dfsB
from .NcBs       import dPgal_dNcBs
from .NsBsB      import dPgal_dNsBsB
from .sigma_c    import dPgal_dsigma_c
from .sigma_so   import dPgal_dsigma_so
from .nuisance   import (dPgal_dNsat_mult, dPgal_df1h_sBsB,
                        dPgal_df1h_cBs, dPgal_dlog10_fso)


__all__ = [ 'dPgal_dalpha_par',
            'dPgal_dalpha_perp',
            'dPgal_df_so',
            'dPgal_dfcB',
            'dPgal_dfs',
            'dPgal_dfsB',
            'dPgal_dNcBs',
            'dPgal_dNsBsB',
            'dPgal_dsigma_c',
            'dPgal_dsigma_so',
            'dPgal_dNsat_mult',
            'dPgal_df1h_sBsB',
            'dPgal_df1h_cBs',
            'dPgal_dlog10_fso'
          ]
