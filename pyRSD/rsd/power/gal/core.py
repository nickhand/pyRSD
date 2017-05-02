import contextlib
from pyRSD.rsd.tools import cacheable

class OneHaloTerm(object):
    """
    A class to represent a constant, one-halo term
    """
    name = None

    def __init__(self, model):
        """
        Parameters
        ----------
        model : GalaxySpectrum
            the model instance
        """
        self.model = model
        self.coefficient = 1.

    def __iter__(self):
        yield self.coefficient, self

    def __call__(self, *args):
        """
        Return the model attribute given by :attr:`name`
        """
        return getattr(self.model, self.name)

    def derivative_k(self, *args):
        """
        Constant as a function of `k`, so derivative is zero
        """
        return 0.

    def derivative_mu(self, *args):
        """
        No `mu` dependence so derivative is zero
        """
        return 0.

class TwoHaloTerm(object):
    """
    A class to represent a two-halo term
    """
    name = None

    def __init__(self, model, b1, b2=None):
        """
        Parameters
        ----------
        model : GalaxySpectrum
            the model instance
        b1 : str
            the name of the first linear bias attribute
        b2 : str, optional
            the name of the second linear bias attribute;
            if not provided, defaults to `b1`
        """
        self.model       = model
        self.coefficient = 1.
        self._b1_name    = b1
        self._b2_name    = b2

    def __iter__(self):
        yield self.coefficient, self

    @property
    def b1(self):
        """
        The first linear bias for this term
        """
        return getattr(self.model, self._b1_name)

    @property
    def b2(self):
        """
        The second linear bias for this term, defaulting
        to `b1`
        """
        if self._b2_name is None: return self.b1
        return getattr(self.model, self._b2_name)

    @contextlib.contextmanager
    def set_biases(self):
        """
        Context manager to set the biases of the model
        """
        # save the original biases and set the new ones
        b1, b1_bar = self.model.b1, self.model.b1_bar
        self.model.b1, self.model.b1_bar = self.b1, self.b2

        yield

        # restore the old ones
        self.model.b1, self.model.b1_bar = b1, b1_bar

    @cacheable
    def __call__(self, k, mu):
        """
        Evaluate the two-halo power by calling :func:`power`,
        evaluated at (`k`,`mu`)
        """
        with self.set_biases():
            return super(self.model.__class__, self.model).power(k, mu)

    @cacheable
    def derivative_k(self, k, mu):
        """
        Evaluate the `k` derivative by calling :func:`derivative_k`,
        evaluated at (`k`,`mu`)
        """
        with self.set_biases():
            return super(self.model.__class__, self.model).derivative_k(k, mu)

    @cacheable
    def derivative_mu(self, k, mu):
        """
        Evaluate the `mu` derivative by calling :func:`derivative_mu`,
        evaluated at (`k`,`mu`)
        """
        with self.set_biases():
            return super(self.model.__class__, self.model).derivative_mu(k, mu)


class GalaxyPowerTerm(object):
    """
    A class to represent a generic galaxy power spectrum term
    """
    name = None

    def __init__(self, model, *terms):
        """
        Parameters
        ----------
        model : GalaxySpectrum
            the model instance
        *terms :
            classes that represent the sub-terms that contribute
            to this term
        """
        self.model = model
        self.terms = [term(model) for term in terms]

    @property
    def coefficient(self):
        return 1.

    def __getitem__(self, key):
        names = [term.name for term in self.terms]
        if key in names:
            return self.terms[names.index(key)]
        else:
            raise KeyError("no such term '%s' in '%s' object" %(key, self.name))

    def __iter__(self):
        for term in self.terms:
            yield term.coefficient, term

    def __call__(self, k, mu):
        """
        Sum the power for each sub-term, weighting by the
        coefficient of each sub-term
        """
        return sum(coeff*term(k, mu) for coeff, term in self)

    def derivative_k(self, k, mu):
        """
        Sum the `k` derivative for each sub-term, weighting by the
        coefficient of each sub-term
        """
        return sum(coeff*term.derivative_k(k, mu) for coeff, term in self)

    def derivative_mu(self, k, mu):
        """
        Sum the `mu` derivative for each sub-term, weighting by the
        coefficient of each sub-term
        """
        return sum(coeff*term.derivative_mu(k, mu) for coeff, term in self)

class DampedGalaxyPowerTerm(GalaxyPowerTerm):
    """
    A class to represent a damped, galaxy power spectrum term,
    which is a `GalaxyPowerTerm`, with FOG kernels applied
    """
    name = None

    def __init__(self, model, *terms, **kws):
        """
        Parameters
        ----------
        model : GalaxySpectrum
            the model instance
        *terms :
            classes that represent the sub-terms that contribute
            to this term
        sigma1 : str
            the name of the first velocity dispersion attribute
        sigma2 : str
            the name of the second velocity dispersion attribute;
            defaults to sigma1
        """
        sigma1 = kws.pop('sigma1', None)
        sigma2 = kws.pop('sigma2', None)

        super(DampedGalaxyPowerTerm, self).__init__(model, *terms)
        self._sigma1_name = sigma1
        self._sigma2_name = sigma2

    @property
    def sigma1(self):
        """
        The first velocity dispersion value
        """
        return getattr(self.model, self._sigma1_name)

    @property
    def sigma2(self):
        """
        The second velocity disperson value, or `None`
        """
        if self._sigma2_name is None: return None
        return getattr(self.model, self._sigma2_name)

    def __call__(self, k, mu):
        """
        Return the damped power spectrum
        """
        G1 = G2 = self.model.FOG(k, mu, self.sigma1)
        if self.sigma2 is not None:
            G2 = self.model.FOG(k, mu, self.sigma2)

        toret = super(DampedGalaxyPowerTerm, self).__call__(k, mu)
        return G1*G2 * toret + self.model.N

    def derivative_k(self, k, mu):
        """
        Derivative with respect to `k`
        """
        # FOG and derivative
        G1      = G2 = self.model.FOG(k, mu, self.sigma1)
        G1prime = G2prime = self.model.FOG.derivative_k(k, mu, self.sigma1)

        if self.sigma2 is not None:
            G2      = self.model.FOG(k, mu, self.sigma2)
            G2prime = self.model.FOG.derivative_k(k, mu, self.sigma2)

        deriv = super(DampedGalaxyPowerTerm, self).derivative_k(k, mu)
        power = super(DampedGalaxyPowerTerm, self).__call__(k, mu)

        return G1*G2 * deriv + (G2*G1prime + G1*G2prime) * power

    def derivative_mu(self, k, mu):
        """
        Derivative with respect to `mu`
        """
        G1      = G2 = self.model.FOG(k, mu, self.sigma1)
        G1prime = G2prime = self.model.FOG.derivative_mu(k, mu, self.sigma1)

        if self.sigma2 is not None:
            G2      = self.model.FOG(k, mu, self.sigma2)
            G2prime = self.model.FOG.derivative_mu(k, mu, self.sigma2)

        deriv = super(DampedGalaxyPowerTerm, self).derivative_mu(k, mu)
        power = super(DampedGalaxyPowerTerm, self).__call__(k, mu)

        return G1*G2 * deriv + (G2*G1prime + G1*G2prime) * power
