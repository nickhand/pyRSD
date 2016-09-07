from .. import GalaxyPowerTerm, ZeroShotNoise

class SOCorrection(object):
    """
    Class to manage the handling of `sigma_c` when using 
    an SO correction
    """
    def __init__(self, model):
        self.model = model
        self._sigma_c = self.model.sigma_c
        
    def __enter__(self):
        if self.model.use_so_correction:
            self.model.sigma_c = 0
        return self.model

    def __exit__(self, *args):
        self.model.sigma_c = self._sigma_c


from .PcAcA import PcAcA
from .PcAcB import PcAcB
from .PcBcB import PcBcB

  
class Pcc(GalaxyPowerTerm):
    """
    The auto specturm of central galaxies
    """
    name = "Pcc"
    
    def __init__(self, model):
        super(Pcc, self).__init__(model, PcAcA, PcAcB, PcBcB)
    
    @property
    def coefficient(self):
        return (1.-self.model.fs)**2 
    
    def __call__(self, k, mu):
        """
        The total centrals auto spectrum
        """
        with ZeroShotNoise(self.model):
            Pk = super(Pcc, self).__call__(k, mu)
        
        # no SO correction
        if not self.model.use_so_correction:
            return Pk
            
        # use an SO correction
        else:
            G1 = self.model.FOG(k, mu, self.model.sigma_c)
            G2 = self.model.FOG(k, mu, self.model.sigma_so)
            
            term1 = (G1*(1-self.model.f_so))**2 * Pk
            term2 = 2*self.model.f_so*(1-self.model.f_so) * G1*G2 * Pk
            term3 = (G2*self.model.f_so)**2 * Pk
            term4 = 2*G1*G2*self.model.f_so*self.model.fcB*self.model.NcBs

            return term1 + term2 + term3 + term4
            
    def derivative_k(self, k, mu):
        """
        The `k` derivative, optionally including the SO correction
        """
        dk = super(Pcc, self).derivative_k(k, mu)
        
        # no SO correction
        if not self.model.use_so_correction:
            return dk
            
        # use an SO correction
        else:
        
            Pk = super(Pcc, self).__call__(k, mu)
        
            G1      = self.model.FOG(k, mu, self.model.sigma_c)
            G2      = self.model.FOG(k, mu, self.model.sigma_so)
            G1prime = self.model.FOG.derivative_k(k, mu, self.model.sigma_c)
            G2prime = self.model.FOG.derivative_k(k, mu, self.model.sigma_so)
            
            f_so = self.model.f_so
            toret = ((G1*(1-f_so))**2 + 2*f_so*(1-f_so)*G1*G2 + (G2*f_so)**2) * dk
            toret += 2*G1*G1prime*(1-f_so)**2 * Pk
            toret += 2*f_so*(1-f_so) * (G1prime*G2 + G2prime*G1) * Pk
            toret += 2*G2*G2prime * f_so**2 * Pk
            toret += 2*f_so*self.model.fcB*self.model.NcBs * (G1prime*G2 + G2prime*G1)
            
            return toret
            
    def derivative_mu(self, k, mu):
        """
        The `mu` derivative, optionally including the SO correction
        """
        dmu = super(Pcc, self).derivative_mu(k, mu)
        
        # no SO correction
        if not self.model.use_so_correction:
            return dmu
            
        # use an SO correction
        else:
            Pk = super(Pcc, self).__call__(k, mu)
        
            G1      = self.model.FOG(k, mu, self.model.sigma_c)
            G2      = self.model.FOG(k, mu, self.model.sigma_so)
            G1prime = self.model.FOG.derivative_mu(k, mu, self.model.sigma_c)
            G2prime = self.model.FOG.derivative_mu(k, mu, self.model.sigma_so)
            
            f_so = self.model.f_so
            toret = ((G1*(1-f_so))**2 + 2*f_so*(1-f_so)*G1*G2 + (G2*f_so)**2) * dmu
            toret += 2*G1*G1prime*(1-f_so)**2 * Pk
            toret += 2*f_so*(1-f_so) * (G1prime*G2 + G2prime*G1) * Pk
            toret += 2*G2*G2prime * f_so**2 * Pk
            toret += 2*f_so*self.model.fcB*self.model.NcBs * (G1prime*G2 + G2prime*G1)
            
            return toret