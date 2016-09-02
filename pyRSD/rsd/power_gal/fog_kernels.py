import numpy
from .. import tools

class FOGKernel(object):
    """
    Factor class for returning a specific `FOGKernel`
    """
    @staticmethod
    def factory(model, name):
        if name == "modified_lorentzian": 
            return ModifiedLorentizanKernel(model)
        elif name == "lorentzian": 
            return LorentzianKernel(model)
        elif name == 'gaussian':
            return GaussianKernel(model)
        else:
            raise TypeError("no FOG kernel with name '%s'" %name) 

    def __init__(self, model):
        self.model = model
        
    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
    
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def __call__(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__kernel__(x)
    
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_k(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__derivative__(x) * mu*sigma
        
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_mu(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__derivative__(x) * k*sigma
        
    @tools.broadcast_kmu
    @tools.alcock_paczynski
    def derivative_sigma(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__derivative__(x) * k*mu
        

class ModifiedLorentizanKernel(FOGKernel):
    """
    A FOG kernel with the functional form:
    
    .. math::
    
        G(x) = 1 / (1 + 0.5 x^2)^2
    """
    def __kernel__(self, x):
        return 1./(1 + 0.5*x**2)**2
        
    def __derivative__(self, x):
        return -2*x / (1. + 0.5*x**2)**3
        
    
        
class LorentzianKernel(FOGKernel):
    """
    A FOG kernel with the functional form:
    
    .. math::
    
        G(x) = 1 / (1 + 0.5 x^2)
    """
    def __kernel__(self, x):
        return 1./(1 + 0.5*x**2)
    
    def __derivative__(self, x):
        return -x / (1. + 0.5*x**2)**2
    

class GaussianKernel(FOGKernel):
    """
    A FOG kernel with the functional form:
    
    .. math::
    
        G(x) = exp[-0.5 x^2]
    """
    def __kernel__(self, x):
        return numpy.exp(-0.5 * x**2)
    
    def __derivative__(self, x):
        return -x * numpy.exp(-0.5 * x**2)
    
