from .. import numpy as np
from ._cache import Cache, parameter, cached_property
from scipy.special import legendre
from scipy.interpolate import InterpolatedUnivariateSpline as spline

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------
def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))
    
def _flatten(arr):
    """
    Flatten and remove any null entries
    """
    flat = arr.ravel(order='F')
    return flat[np.isfinite(flat)]
   
def trim_zeros_indices(filt):
    """
    Return the indices (first, last) specifying
    the indices to trim leading or trailing zeros
    """
    first = 0
    for i in filt:
        if i != 0.:
            break
        else:
            first = first + 1
    last = len(filt)
    for i in filt[::-1]:
        if i != 0.:
            break
        else:
            last = last - 1
    return first, last   

#------------------------------------------------------------------------------
# P(k,mu) grid
#------------------------------------------------------------------------------      
class PkmuGrid(object):
    """
    A class to represent a 2D grid of (k, mu), with `k` as the 0th axis and
    `mu` as the 1st axis
    """
    def __init__(self, coords, k, mu, modes):
        """
        Parameters
        ----------
        coords : list
            list of (k_cen, mu_cen) representing the bin centers
            of the grid coordinates
        k : array_like, (Nk, Nmu)
            the mean values of `k` at each grid point
        mu : array_like, (Nk, Nmu)
            the mean values of `mu` at each grid point
        modes : array_like, (Nk, Nmu)
            the number of modes in each grid point 
        """
        self.k_cen, self.mu_cen = coords
        self.k     = k
        self.mu    = mu
        self.modes = modes
        
    def to_plaintext(self, filename):
        """
        Write out the grid to a plaintext file
        """
        with open(filename, 'wb') as ff:
            header = "%d %d\n" %(self.Nk, self.Nmu)
            ff.write(header.encode())
            np.savetxt(ff, self.k_cen)
            np.savetxt(ff, self.mu_cen)
            d = np.dstack([self.k, self.mu, self.modes])
            np.savetxt(ff, d.reshape((-1, 3)))
            
    @classmethod
    def from_pkmuresult(cls, pkmu):
        """
        Convienence method to return a ``PkmuGrid`` from a 
        ``nbodykit.PkmuResult`` instance
        """
        coords = [pkmu.coords['k'], pkmu.coords['mu']]
        return cls(coords, pkmu['k'], pkmu['mu'], pkmu['modes'])
    
    @classmethod
    def from_structured(cls, coords, data):
        """
        Return a ``PkmuGrid`` from a list of coordinates and
        a structured array
        
        Parameters
        ----------
        coords : list
            list of (k_cen, mu_cen) representing the bin centers
            of the grid coordinates
        data : array_like
            structured array with `k`, `mu`, and `modes` fields
        """
        return cls(coords, data['k'], data['mu'], data['modes'])
    
    @classmethod
    def from_plaintext(cls, filename):
        """
        Return a `PkmuGrid` instance, reading the data from a plaintext
        """
        with open(filename, 'r') as ff:
            lines = ff.readlines()
            
        Nk, Nmu = [int(l) for l in lines[0].split()]
        k_cen = np.array([float(l) for l in lines[1:Nk+1]])
        mu_cen = np.array([float(l) for l in lines[Nk+1:Nk+Nmu+1]])
        data = np.array([[float(ll) for ll in l.split()] for l in lines[Nk+Nmu+1:]]).reshape((Nk, Nmu, 3))
        
        return cls([k_cen, mu_cen], *np.rollaxis(data, 2))
            
    @property
    def shape(self):
        return self.k.shape
    
    @property
    def Nk(self):
        return self.shape[0]
    
    @property
    def Nmu(self):
        return self.shape[1]
    
    @property
    def notnull(self):
        return (np.isfinite(self.modes))&(self.modes > 0)
        
    def __str__(self):
        """
        Builtin string representation
        """
        cls = self.__class__.__name__
        args = (cls, self.Nk, self.Nmu, (1.-1.*self.notnull.sum()/np.prod(self.shape))*100.)
        return "<%s: size (%dx%d), %.2f%% empty grid points>" %args
    
    def __repr__(self):
        return self.__str__()

class PkmuTransfer(Cache):
    """
    Class to facilitate the manipulations of P(k,mu) measurements
    on a (k, mu) grid
    """
    def __init__(self, grid, mu_bounds, kmin=None, kmax=None, power=None):
        """
        Parameters
        ----------
        grid : PkmuGrid
            the grid instance defining the (k,mu) grid
        mu_bounds : array_like
            a list of tuples specifying (lower, upper) for each desired
            mu bin, i.e., [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        kmin : float, array_like
            the minimum allowed wavenumber
        kmax : float, array_like
            the maximum wavenumber
        power : array_like
            the power values defined on the grid -- can have shape
            (Nk,Nmu) or (N,), in which case it is interpreted
            as the values at all valid grid points
        """        
        self.grid      = grid
        self.mu_bounds = mu_bounds
        self.kmin      = kmin
        self.kmax      = kmax
        self.power     = power
        
    @classmethod
    def from_structured(cls, coords, data, mu_bounds, **kwargs):
        """
        Convenience function to teturn a `PkmuTransfer` object 
        from the coords+data instead of a `PkmuGrid`
        """
        grid = PkmuGrid.from_structured(coords, data)
        return cls(grid, mu_bounds, **kwargs)
        
    #--------------------------------------------------------------------------
    # parameters
    #--------------------------------------------------------------------------            
    @parameter
    def power(self, val):
        """
        The power array holding P(k,mu) on the grid -- shape is 
        (grid.Nk, grid.Nmu), with NaNs for any null grid points
        """
        toret = np.ones(self.grid.shape)*np.nan
        if val is None:
            return toret
            
        valid = self.grid.notnull.sum()
        if np.ndim(val) == 1:
            if len(val) == valid:
                toret[self.grid.notnull] = val
            else:
                raise ValueError("if 1D array is passed for ``power``, must have length %d" %valid)
        else:
            toret[self.grid.notnull] = val[self.grid.notnull]
        return toret
        
    @parameter
    def mu_bounds(self, val):
        """
        A list of tuples specifying the lower and upper limits for each 
        desired `mu` bin
        """
        # if mu = 1.0 is upper bound of bin, increase by epsilon to include edge
        for i, (lo, hi) in enumerate(val):
            if np.isclose(hi, 1.0): 
                val[i] = (val[i][0], 1.005*val[i][1])
        return val
        
    @parameter
    def kmin(self, val):
        """
        The minimum wavenumber to include -- can be either
        a float or array of length ``N2``
        """
        if val is None: val = -np.inf
        toret = np.empty(self.N2)
        toret[:] = val
        return toret
    
    @parameter
    def kmax(self, val):
        """
        The maximum wavenumber to include -- can be either
        a float or array of length ``N2``
        """
        if val is None: val = np.inf
        toret = np.empty(self.N2)
        toret[:] = val
        return toret

    #--------------------------------------------------------------------------
    # cached properties
    #--------------------------------------------------------------------------            
    @cached_property("mu_bounds")
    def mu_edges(self):
        """
        The edges of the mu bins desired for output
        """
        toret = np.asarray(self.mu_bounds).ravel()
        if not non_decreasing(toret):
            raise ValueError("specified `mu` bounds are not monotonically increasing")
        return toret
            
    @cached_property()
    def size(self):
        """
        The size of the flattened (valid) grid points, i.e., the number of
        (k,mu) or (k,ell) data points
        """
        return len(self.coords_flat[0])
        
    @cached_property()
    def N1(self):
        return self.grid.Nk
        
    @cached_property("mu_edges")
    def N2(self):
        return len(self.mu_bounds)
        
    @cached_property("coords")
    def coords_flat(self):
        """
        List of the flattened coordinates, with NaNs removed
        """
        return [_flatten(x) for x in self.coords]
    
    @cached_property("mu_edges", "in_range_idx")
    def coords(self):
        """
        The (k,mu) coordinates with shape (`N1`, `N2`)
        """
        # broadcast (k_cen, mu_cen) to the shape (N1, N2)
        idx = self.in_range_idx
        k_cen = np.ones((self.N1, self.N2)) * self.grid.k_cen[:,None]
        mu_cen = 0.5*(self.mu_edges[1:] + self.mu_edges[:-1])[::2]
        mu_cen = np.ones((self.N1, self.N2)) * mu_cen[None,:]
        
        # restrict the k-range and return
        return self.restrict_k(k_cen), self.restrict_k(mu_cen)
    
    @cached_property("kmin", "kmax")
    def in_range_idx(self):
        """
        A boolean array with elements set to `True` specifying the elements
        on the underlying (k,mu) grid within the desired `kmin` and `kmax`
        """
        k_cen = self.grid.k_cen
        return np.squeeze((k_cen[:,None] >= self.kmin)&(k_cen[:,None] <= self.kmax))
        
    @cached_property("mu_edges")
    def ndims(self):
        """
        Convenience attribute to store the dimensions of the bin counting arrays
        """
        Nk = self.grid.Nk; Nmu = len(self.mu_edges)-1
        return (Nk+2, Nmu+2)
    
    @cached_property("mu_edges")
    def mu_indices(self):
        """
        Multi-index for re-binning the `mu` grid
        """
        k_idx = np.arange(self.grid.Nk, dtype=int)[:,None]
        dig_k = np.repeat(k_idx, self.grid.Nmu, axis=1)[self.grid.notnull] + 1
        dig_mu = np.digitize(self.grid.mu[self.grid.notnull], self.mu_edges)
        return np.ravel_multi_index([dig_k, dig_mu], self.ndims)
         
    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def sum(self, d):
        """
        Sum the input data over the `mu` bins specified by `mu_edges`
        
        Parameters
        ----------
        d : array_like, (`grid.Nk`, `grid.Nmu`)
            the data array to sum over the grid
        """
        toret = np.zeros(self.ndims)
        minlength = np.prod(self.ndims)
        toret.flat = np.bincount(self.mu_indices, weights=d[self.grid.notnull], minlength=minlength)
        return np.squeeze(toret.reshape(self.ndims)[1:-1, 1:-1][..., ::2])
        
    def average(self, d, w=None):
        """
        Average the input data, with optional weights, as a function of `mu` 
        on the grid, using `mu_edges` to specify the edges of the new mu bins
        
        Parameters
        ----------
        d : array_like, (`grid.Nk`, `grid.Nmu`)
            the data array to average over the grid
        w : array_like, (`grid.Nk`, `grid.Nmu`)
            optional weight array to apply before averaging
        """
        if w is None: w = np.ones(d.shape)
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.sum(d*w) / self.sum(w)
                
    def restrict_k(self, arr):
        """
        Restrict the k-range of `arr` to the specified `kmin` and `kmax`
        
        Parameters
        ----------
        arr : array_like (`grid.Nk`, `grid.Nmu`)
            trim the array defined on the grid to the desired k-range, 
            filling with NaNs if not aligned properly 
        """
        cnts = self.in_range_idx.astype(int)
        first, last = trim_zeros_indices(cnts.sum(axis=1))
        
        toret = arr.copy()
        toret[~self.in_range_idx] = np.nan
        return toret[first:last,...]
        
    def __call__(self, flatten=False, full_grid_output=False):
        """
        Return the `power` attribute re-binned into the mu bins 
        corresponding to `mu_edges`, optionally flattening the return array
        
        Notes
        -----
        The returned array will be restricted to (`kmin`, `kmax`)
        
        Parameters
        ----------
        flatten : bool, optional (`False`)
            whether to flatten the return array (column-wise)
        """
        if not (~np.isnan(self.power)).sum():
            raise ValueError("please set ``power`` arrary; all NaN right now")
            
        toret = self.average(self.power, self.grid.modes)
        if not full_grid_output:
            toret = self.restrict_k(toret)
        if flatten: toret = _flatten(toret)
        return toret
        
    def to_covariance(self, components=False):
        """
        Return the P(k,mu) covariance, which is diagonal and equal to
        
        :math: C = 2 * P(k,mu)^2 / N(k, mu)
        
        Parameters
        ----------
        components : bool, optional (`False`)
            If `True`, the mean squared power and modes (with the the shape
            of the covariance matrix), such that C = 2*Psq/modes
        """
        # the covariance per mu bin
        Psq = self.average(self.power**2, self.grid.modes)
        modes = self.sum(self.grid.modes)
        
        # reshape squared power and modes
        Psq = np.diag(Psq.ravel(order='F'))
        modes = np.diag(modes.ravel(order='F'))
        return self._return_covariance(Psq, modes, components)
            
    def _return_covariance(self, Psq, modes, components):
        """
        Internal function to restrict the components of 
        the covariance matrix and return 
        """
        idx = self.in_range_idx.ravel(order='F')
        Psq = Psq[idx,:][:,idx]; modes = modes[idx,:][:,idx]
        return (Psq**0.5, modes) if components else np.nan_to_num(2*Psq/modes)
    
    
class PolesTransfer(PkmuTransfer):
    """
    Class to facilitate the manipulations of multipole measurements
    on an underlying (k, mu) grid
    """
    def __init__(self, grid, ells, kmin=-np.inf, kmax=np.inf, power=None, ells_mask=None):
        """
        Parameters
        ----------
        grid : PkmuGrid
            the grid instance defining the (k,mu) grid
        ells : array_like
            the multipole numbers desired for the output
        kmin : float, array_like
            the minimum allowed wavenumber
        kmax : float, array_like
            the maximum wavenumber
        power : array_like
            the power values defined on the grid -- can have shape
            (Nk,Nmu) or (N,), in which case it is interpreted
            as the values at all valid grid points
        ells_mask : list, None, optional
            boolean mask specifying which ell values to return; this is needed in the 
            case that we are doing a window convolution and need more ell values
        """
        self.ells = np.array(ells)
        if ells_mask is None: 
            ells_mask = [True for ell in self.ells]
        self.ells_mask = np.asarray(ells_mask)
        
        super(PolesTransfer, self).__init__(grid, [(0., 1.)], kmin=kmin, kmax=kmax, power=power)
        
    @classmethod
    def from_structured(cls, coords, data, ells, **kwargs):
        """
        Convenience function to teturn a `PolesTransfer` object 
        from the coords+data instead of a `PkmuGrid`
        """
        grid = PkmuGrid.from_structured(coords, data)
        return cls(grid, ells, **kwargs)
        
    #--------------------------------------------------------------------------
    # parameters
    #--------------------------------------------------------------------------
    @parameter
    def kmin(self, val):
        """
        The minimum wavenumber to include -- can be either
        a float or array of length ``N2``
        """
        N2 = self.ells_mask.sum()
        if val is None: val = -np.inf
        toret = np.empty(N2)
        toret[:] = val
        return toret
    
    @parameter
    def kmax(self, val):
        """
        The maximum wavenumber to include -- can be either
        a float or array of length ``N2``
        """
        N2 = self.ells_mask.sum()
        if val is None: val = np.inf
        toret = np.empty(N2)
        toret[:] = val
        return toret
        
    @parameter
    def ells(self, val):
        """
        The multipole numbers to compute -- this corresponds to the
        second axis (columns)
        """
        return val
                
    @cached_property()
    def k_mean(self):
        """
        The `k` values averaged over the `mu` dimension
        """
        return self.average(self.grid.k, self.grid.modes)
            
    @cached_property("ells")
    def N2(self):
        return len(self.ells)  
        
    @cached_property("ells")
    def legendre_weights(self):
        """
        Legendre weighting for multipoless
        """
        return np.array([(2*ell+1)*legendre(ell)(self.grid.mu) for ell in self.ells])
        
    @cached_property("ells", "in_range_idx")
    def coords(self):
        """
        The (k,ell) coordinates with shape (`N1`, `N2`)
        """
        # broadcast (k_cen, ells) to the shape (N1, N2)
        idx = self.in_range_idx
        k_cen = np.ones((self.N1, self.N2)) * self.grid.k_cen[:,None]
        ells = np.ones((self.N1, self.N2)) * self.ells[None,:]
        
        # restrict the k-range and return
        return self.restrict_k(k_cen), self.restrict_k(ells)

    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def restrict_k(self, arr):
        """
        Apply ``ells_mask`` and then restrict_k
        """        
        # ells mask
        arr = arr[:,self.ells_mask]
        return super(PolesTransfer, self).restrict_k(arr)
        
    def __call__(self, flatten=False, full_grid_output=False):
        """
        Return the multipoles corresponding to `ells`, by weighting the
        `power` attribute by the appropriate Legendre weights and summing
        over the `mu` dimension
        
        Notes
        -----
        The returned array will be restricted to (`kmin`, `kmax`)
        
        Parameters
        ----------
        flatten : bool, optional (`False`)
            whether to flatten the return array (column-wise)
        """
        if not (~np.isnan(self.power)).sum():
            raise ValueError("please set ``power`` arrary; all NaN right now")
            
        # average over grid
        tobin = self.legendre_weights*self.power
        toret = np.asarray([self.average(d, self.grid.modes) for d in tobin]).T
                
        # restrict k range
        if not full_grid_output:
            toret = self.restrict_k(toret)
        
        # flatten and return
        if flatten: toret = _flatten(toret)
        return toret 
                
    def to_covariance(self, components=False):
        """
        Return the P(k,mu) covariance, which is diagonal and equal to
        
        :math: C_{l,l'} = 2 / N(k) * \sum_\mu L(\mu, l) L(\mu, l') P(k,mu)^2 
        
        Parameters
        ----------
        components : bool, optional (`False`)
            If `True`, the mean squared power and modes (with the the shape
            of the covariance matrix), such that C = 2*Psq/modes
        """
        # initialize the return array
        Psq = np.zeros((self.N2, self.N1)*2)
        modes = np.zeros((self.N2, self.N1)*2)
        
        # legendre weights*power squared -- shape is (Nell, Nell, Nk, Nmu)
        weights = self.legendre_weights[:,None]*self.legendre_weights[None,:]
        tobin = weights*self.power**2
        
        # fill the covariance for each ell, ell_prime
        for i in range(self.N2):
            for j in range(i, self.N2):
                Psq[i,:,j,:] = np.diag(self.average(tobin[i,j,:], self.grid.modes))
                modes[i,:,j,:] = np.diag(self.sum(self.grid.modes))
                if i != j: 
                    Psq[j,:,i,:] = Psq[i,:,j,:]
                    modes[j,:,i,:] = modes[i,:,j,:]
        
        # reshape squared power and modes
        Psq = Psq.reshape((self.N1*self.N2,)*2)
        modes = modes.reshape((self.N1*self.N2,)*2)
        return self._return_covariance(Psq, modes, components)
        
        
    def to_cutsky_covariance(self, cosmo, zbins, nbar, P0, fsky, components=False):
        """
        Return the P(k,mu) covariance, which is diagonal and equal to
        
        :math: C_{l,l'} = 2 / N(k) * \sum_\mu L(\mu, l) L(\mu, l') P(k,mu)^2 
        
        Parameters
        ----------
        components : bool, optional (`False`)
            If `True`, the mean squared power and modes (with the the shape
            of the covariance matrix), such that C = 2*Psq/modes
        """
        # volume of redshift shells
        dV = cosmo.V(zbins[:-1], zbins[1:]) * cosmo.h()**3 * 1e9 * fsky # in (Mpc/h)^3
        
        # number density
        zcen = 0.5*(zbins[:-1] + zbins[1:])
        nbar_ = nbar(zcen)
        w = 1. / (1 + nbar_*P0) # FKP weights
        
        # initialize the return array
        Psq = np.zeros((self.N2, self.N1)*2)
        modes = np.zeros((self.N2, self.N1)*2)
        
        # legendre weights*power squared -- shape is (Nell, Nell, Nk, Nmu, Nz)
        weights = self.legendre_weights[:,None]*self.legendre_weights[None,:]
        power = (self.power[...,None] + 1./nbar_)**2
        tobin = weights[...,None] * power[None,...]
        
        # normalization for redshift integral
        W = ((nbar_*w)**2 * dV).sum()
        
        # k-shell volume 
        dk = np.diff(self.k_mean).mean()
        Vk  = 4*np.pi*self.k_mean**2*dk
        
        # fill the covariance for each ell, ell_prime
        for i in range(self.N2):
            for j in range(i, self.N2):
                
                # do the sum over redshift first
                x = ( (w*nbar_)**4 * dV * tobin ).sum(axis=-1) / W**2
                Psq[i,:,j,:] = np.diag(self.average(x[i,j,:], self.grid.modes))
                modes[i,:,j,:] = 2 * Vk / (2*np.pi)**3
                if i != j: 
                    Psq[j,:,i,:] = Psq[i,:,j,:]
                    modes[j,:,i,:] = modes[i,:,j,:]
        
        # reshape squared power and modes
        Psq = Psq.reshape((self.N1*self.N2,)*2)
        modes = modes.reshape((self.N1*self.N2,)*2)
        return self._return_covariance(Psq, modes, components)

    

        

    


    
    
