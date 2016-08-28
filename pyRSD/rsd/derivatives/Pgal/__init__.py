import abc

class PgalDerivative(abc.ABC):
    """
    Abstract base class for derivatives of the `Pgal` model
    """
    @classmethod
    def registry(cls):
        """
        Return the registered subclasses
        """
        d = {}
        for subclass in cls.__subclasses__():
            d[subclass.param] = subclass
            
        return d
    
    @abc.abstractstaticmethod
    def eval(model, pars, k, mu):
        pass
        
def get_Pgal_derivative(name):
    """
    Given an input parameter name, return the
    class that computes `dPgal/dname`
    """
    registry = PgalDerivative.registry()
    if name not in registry:
        raise ValueError("no registered subclass for dPgal/d%s" %name)
    return registry[name]
    
    
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


__all__ = [ 'get_Pgal_derivative',
            'dPgal_dalpha_par',
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
