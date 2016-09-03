import numpy
from pyRSD.rsd import tools

class FOGKernel(object):
    """
    Factor class for returning a specific `FOGKernel`
    """
    @staticmethod
    def factory(name):
        if name == "modified_lorentzian": 
            return ModifiedLorentizanKernel()
        elif name == "lorentzian": 
            return LorentzianKernel()
        elif name == 'gaussian':
            return GaussianKernel()
        else:
            raise TypeError("no FOG kernel with name '%s'" %name) 
        
    @tools.broadcast_kmu
    def __call__(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__kernel__(x)
    
    @tools.broadcast_kmu
    def derivative_k(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__derivative__(x) * mu*sigma
        
    @tools.broadcast_kmu
    def derivative_mu(self, k, mu, sigma):
        x = k*mu*sigma
        return self.__derivative__(x) * k*sigma
        
    @tools.broadcast_kmu
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
    
