from astropy import cosmology, units
import numpy as np
import functools
from pyRSD.pygcl import transfers

def removeunits(f):
    """
    Decorator to remove units from :class:`astropy.units.Quantity`
    instances
    """
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        ans = f(*args, **kwargs)
        if isinstance(ans, units.Quantity):
            ans = ans.value
        return ans
    return wrapped

class fittable(object):
    """
    A "fittable" function

    There exists a `.fit()` method of the original function
    which returns a spline-interpolated version of the function
    for a specified variable
    """
    def __init__(self, func, instance=None):

        # update the docstring, etc from the original func
        functools.update_wrapper(self, func)
        self.func = func
        self.instance = instance

    def __get__(self, instance, owner):
        # descriptor that binds the method to an instance
        return fittable(self.func, instance=instance)

    def __call__(self, *args, **kwargs):
        return self.func(self.instance, *args, **kwargs)

    def fit(self, argname, kwargs={}, bins=1024, range=None):
        """
        Interpolate the function for the given argument (`argname`)
        with a :class:`~scipy.interpolate.InterpolatedUnivariateSpline`

        `range` and `bins` behave like :func:`numpy.histogram`

        Parameters
        ----------
        argname : str
            the name of the variable to interpolate
        kwargs : dict; optional
            dict of keywords to pass to the original function
        bins : int, iterable; optional
            either an iterable specifying the bin edges, or an
            integer specifying the number of linearly-spaced bins
        range : tuple; optional
            the range to fit over if `bins` specifies an integer

        Returns
        -------
        spl : callable
            the callable spline function
        """
        from scipy import interpolate
        from astropy.units import Quantity

        if isiterable(bins):
            bin_edges = np.asarray(bins)
        else:
            assert len(range) == 2
            bin_edges = np.linspace(range[0], range[1], bins + 1, endpoint=True)

        # evaluate at binned points
        d = {}
        d.update(kwargs)
        d[argname] = bin_edges
        y = self.__call__(**d)

        # preserve the return value of astropy functions by attaching
        # the right units to the splined result
        spl = interpolate.InterpolatedUnivariateSpline(bin_edges, y)
        if isinstance(y, Quantity):
            return lambda x: Quantity(spl(x), y.unit)
        else:
            return spl

def vectorize_if_needed(func, *x):
    """
    Helper function to vectorize functions on array inputs;
    borrowed from :mod:`astropy.cosmology.core`
    """
    if any(map(isiterable, x)):
        return np.vectorize(func)(*x)
    else:
        return func(*x)

def isiterable(obj):
    """
    Returns `True` if the given object is iterable;
    borrowed from :mod:`astropy.cosmology.core`
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False

class Cosmology(dict):
    """
    Dict-like object for cosmological parameters and related calculations

    An extension of the :mod:`astropy.cosmology` framework that can
    store additional, orthogonal parameters and behaves like a read-only
    dictionary

    The class relies on :mod:`astropy.cosmology` as the underlying
    "engine" for calculation of cosmological quantities. This "engine"
    is stored as :attr:`engine` and supports :class:`~astropy.cosmology.LambdaCDM`
    and :class:`~astropy.cosmology.wCDM`, and their flat equivalents

    Any attributes or functions of the underlying astropy engine
    can be directly accessed as attributes or keys of this class

    .. note::

        A default set of units is assumed, so attributes stored internally
        as :class:`astropy.units.Quantity` instances will be returned
        here as numpy arrays. Those units are:

        - temperature: ``K``
        - distance: ``Mpc``
        - density: ``g/cm^3``
        - neutrino mass: ``eV``
        - time: ``Gyr``
        - H0: ``Mpc/km/s``

    .. warning::

        This class does not currently support a non-constant dark energy
        equation of state
    """
    def __init__(self, H0=67.6, Om0=0.31, Ob0=0.0486, Ode0=0.69, w0=-1., Tcmb0=2.7255,
                    Neff=3.04, m_nu=0., n_s=0.9667, sigma8=0.8159, flat=False, name=None):

        """
        Parameters
        ----------
        H0 : float
            the Hubble constant at z=0, in km/s/Mpc
        Om0 : float
            matter density/critical density at z=0
        Ob0 : float
            baryon density/critical density at z=0
        Ode0 : float
            dark energy density/critical density at z=0
        w0 : float
            dark energy equation of state
        Tcmb0 : float
            temperature of the CMB in K at z=0
        Neff : float
            the effective number of neutrino species
        m_nu : float, array_like
            mass of neutrino species in eV
        sigma8 : float
            the present day value of ``sigma_r(r=8 Mpc/h)``, used to normalize
            the power spectrum, which is proportional to the square of this
            value
        n_s : float
            the spectral index of the primoridal power spectrum
        flat : bool
            if `True`, automatically set `Ode0` such that `Ok0` is zero
        name : str
            a name for the cosmology
        """
        # convert neutrino mass to a astropy `Quantity`
        if m_nu is not None:
            m_nu = units.Quantity(m_nu, 'eV')

        # the astropy keywords
        kws = {'name':name, 'Ob0':Ob0, 'w0':w0, 'Tcmb0':Tcmb0, 'Neff':Neff,
              'm_nu':m_nu, 'Ode0':Ode0}

        # determine the astropy class
        if w0 == -1.0: # cosmological constant
            cls = 'LambdaCDM'
            kws.pop('w0')
        else:
            cls = 'wCDM'

        # use special flat case if Ok0 = 0
        if flat:
            cls = 'Flat' + cls
            kws.pop('Ode0')

        # initialize the astropy engine
        self.engine = getattr(cosmology, cls)(H0=H0, Om0=Om0, **kws)

        # add valid params to the underlying dict
        kwargs = {'flat':flat, 'w0':w0}
        for k in kws:
            if hasattr(self.engine, k):
                kwargs[k] = getattr(self.engine, k)
        dict.__init__(self, H0=H0, Om0=Om0, sigma8=sigma8, n_s=n_s, **kwargs)

    def __repr__(self):
        """
        Representation that removes Quantity
        """
        return repr({k:self[k] for k in self.keys()})

    def copy(self):
        return Cosmology(**dict.copy(self))

    def __eq__(self, other):

        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return False
        if self.flat is not other.flat:
            return False
        if not np.allclose(self.m_nu, other.m_nu, rtol=1e-4):
            return False
        keys = ['H0', 'Om0', 'Ob0', 'Ode0', 'w0', 'Tcmb0','Neff','n_s','sigma8']
        for key in keys:
            if not np.isclose(getattr(self, key), getattr(other, key), rtol=1e-4):
                return False

        return True

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        return self.__init__(**state)

    def __reduce__(self):
        return (self.__class__, (), self.__getstate__())

    def __dir__(self):
        """
        Explicitly the underlying astropy engine's attributes as
        part of the attributes of this class
        """
        this_attrs = set(dict.__dir__(self)) | set(self.keys())
        engine_attrs = set(self.engine.__dir__())
        return list(this_attrs|engine_attrs)

    @classmethod
    def from_astropy(self, cosmo, n_s=0.9667, sigma8=0.8159, **kwargs):
        """
        Return a :class:`Cosmology` instance from an astropy cosmology

        Parameters
        ----------
        cosmo : subclass of :class:`astropy.cosmology.FLRW`
            the astropy cosmology instance
        **kwargs :
            extra key/value parameters to store in the dictionary
        """
        valid = ['H0', 'Om0', 'Ob0', 'Ode0', 'w0', 'Tcmb0', 'Neff', 'm_nu']
        for name in valid:
            if hasattr(cosmo, name):
                par = getattr(cosmo, name)
                if isinstance(par, units.Quantity):
                    par = par.value
                kwargs[name] = par
        kwargs['flat'] = cosmo.Ok0 == 0.
        kwargs.setdefault('name', getattr(cosmo, 'name', None))

        toret = Cosmology(sigma8=sigma8, n_s=n_s, **kwargs)
        toret.__doc__ += "\n" + cosmo.__doc__
        return toret

    def __setitem__(self, key, value):
        """
        No setting --> read-only
        """
        print(key, value)
        raise ValueError("Cosmology is a read-only dictionary; see clone() to create a copy with changes")

    @removeunits
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    @removeunits
    def __missing__(self, key):
        """
        Missing dict keys returned only if they are attributes of the
        underlying astropy engine and are not callable functions
        """
        # try to return the parameter from the engine
        if hasattr(self.engine, key):
            toret = getattr(self.engine, key)
            if not callable(toret):
                return toret

        # otherwise fail
        raise KeyError("no such parameter '%s' in Cosmology" %key)

    @removeunits
    def __getattr__(self, key):
        """
        Try to return attributes from the underlying astropy engine and
        then from the dictionary

        Notes
        -----
        For callable attributes part of the astropy engine's public API (i.e.,
        functions that do not begin with a '_'), the function will be decorated
        with the :class:`fittable` class
        """
        try:
            toret = getattr(self.engine, key)
            # if a callable function part of public API of the "engine", make it fittable
            if callable(toret) and not key.startswith('_'):
                toret = fittable(toret.__func__, instance=self)

            return toret
        except:
            if key in self:
                return self[key]
            raise AttributeError("no such attribute '%s'" %key)

    def clone(self, **kwargs):
        """
        Returns a copy of this object, potentially with some changes.

        Returns
        -------
        newcos : Subclass of FLRW
            A new instance of this class with the specified changes.

        Notes
        -----
        This assumes that the values of all constructor arguments
        are available as properties, which is true of all the provided
        subclasses but may not be true of user-provided ones.  You can't
        change the type of class, so this can't be used to change between
        flat and non-flat.  If no modifications are requested, then
        a reference to this object is returned.

        Examples
        --------
        To make a copy of the Planck15 cosmology with a different Omega_m
        and a new name:

        >>> from astropy.cosmology import Planck15
        >>> cosmo = Cosmology.from_astropy(Planck15)
        >>> newcos = cosmo.clone(name="Modified Planck 2013", Om0=0.35)
        """
        # filter out astropy-defined parameters and extras
        extras = {k:self[k] for k in self if not hasattr(self.engine, k)}
        extras.update({k:kwargs.pop(k) for k in list(kwargs) if not hasattr(self.engine, k)})

        # make the new astropy instance
        new_engine = self.engine.clone(**kwargs)

        # return a new Cosmology instance
        return self.from_astropy(new_engine, **extras)

    def to_class(self, transfer=transfers.CLASS, linear_power_file=None, **class_config):
        """
        Convert the object to a :class:`pyRSD.pygcl.Cosmology` instance in
        order to interface with the CLASS code

        Parameters
        ----------
        **class_config : key/value pairs
            keywords to pass to the CLASS engine; defaults are `z_max_pk=2.0`
            and `P_k_max_h/Mpc=20.0`

        Returns
        -------
        cosmo : pygcl.Cosmology
            the pygcl Cosmology object which interfaces with CLASS
        """
        from pyRSD.pygcl import Cosmology, ClassParams

        # set some default CLASS config params
        class_config.setdefault('z_max_pk', 2.0)
        class_config.setdefault('P_k_max_h/Mpc', 20.0)

        # the CLASS params
        pars = ClassParams.from_astropy(self.engine, extra=class_config)
        pars['n_s'] = self.n_s

        if linear_power_file is not None:
            k, Pk = np.loadtxt(linear_power_file, unpack=True)
            cosmo = Cosmology.from_power(linear_power_file, k, Pk)
        else:
            cosmo = Cosmology(pars, transfer)

        # set Sigma8
        cosmo.SetSigma8(self.sigma8)
        return cosmo

# override these with Cosmology classes below
from astropy.cosmology import Planck13, Planck15, WMAP5, WMAP7, WMAP9

# Planck defaults with sigma8, n_s
Planck13 = Cosmology.from_astropy(Planck13, sigma8=0.8288, n_s=0.9611)
Planck15 = Cosmology.from_astropy(Planck15, sigma8=0.8159, n_s=0.9667)

# WMAP defaults with sigma8, n_s
WMAP5 = Cosmology.from_astropy(WMAP5, sigma8=0.817, n_s=0.962)
WMAP7 = Cosmology.from_astropy(WMAP7, sigma8=0.810, n_s=0.967)
WMAP9 = Cosmology.from_astropy(WMAP9, sigma8=0.820, n_s=0.9608)
