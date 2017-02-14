from __future__ import print_function

import numpy as np
import logging
from collections import namedtuple
import scipy.optimize as opt

# machine precision
_epsilon = np.finfo(float).eps

class LineSearchError(Exception):
    pass

class LBFGSStepError(Exception):
    pass

PARAM_LEVEL = logging.ERROR+1
DERIV_LEVEL = logging.ERROR+2
logging.addLevelName("PARAM", PARAM_LEVEL)
logging.addLevelName("DERIV", DERIV_LEVEL)

class State(object):
    """
    A class implementing the `state` of the minimizer algorithm, storing:
    
        X : current value of the parameters
        F : value of the objective function
        G : value of the gradient vector
        Gnorm : norm of the gradient vector
    """
    _names = ['X', 'F', 'G', 'Gnorm']
    
    def __init__(self, X, F, G, Gnorm=None):
        """
        Parameters
        ----------
        X : array_like
            the current value of the parameters
        F : float
            the value of the objective function
        G : array_like
            the value of the gradient vector
        Gnorm : float, optional
            the norm of the gradient vector; will be set automatically
            if not provided
        """
        self.X = X
        self.F = F
        self.G = G
        if Gnorm is None:
            Gnorm = np.linalg.norm(G)
        self.Gnorm = Gnorm
        
    def __contains__(self, name):
        return name in self._names
    
    def keys(self):
        return self._names
    
    def __str__(self):
        return "<State: F=%.6g, Gnorm=%.6g>" %(self.F, self.Gnorm)
        
    def __repr__(self):
        return self.__str__()
        
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise AttributeError(key)
    
    def __iter__(self):
        for a in (self.X, self.F, self.G, self.Gnorm):
            yield a
            
    def update(self, X, F, G, Gnorm=None):
        """
        Update the `State`, copying the elements of the 
        input `X` and `G` arrays into memory owned by 
        this object
        """
        self.X[:] = X
        self.F = F
        self.G[:] = G
        if Gnorm is None:
            Gnorm = np.linalg.norm(G)
        self.Gnorm = Gnorm

    def copy(self):
        """
        Return a full copy of the current `State`
        """
        args = (self.X.copy(), self.F, self.G.copy(), self.Gnorm)
        return State(*args)
        
class LimitedMemoryInverseHessian(object):
    """
    A class implementing the limit-memory BFGS approximation of the inverse Hessian
    """
    def __init__(self, H0, shape):
    
        self.H0k = H0
        self.s   = np.zeros(shape)
        self.y   = np.zeros(shape)
        self.rho = np.zeros(shape[0])
        
    def __len__(self):
        return len(self.rho)
        
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise AttributeError(key)
            
    def update(self, sk, yk, logger=None):
        """
        Update the inverse Hessian, by iterating the `s`, `y`,
        and `rho` arrays and updating the value of `H0k`
        """
        # move everything down one element
        self.s[:]   = np.roll(self.s, 1, axis=0)
        self.y[:]   = np.roll(self.y, 1, axis=0)
        self.rho[:] = np.roll(self.rho, 1)
        
        # updates for rho and H0k
        yy = np.linalg.norm(yk)
        yy *= yy
        
        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (np.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if logger is not None:
                logger.warning("Divide-by-zero encountered: rhok assumed large")
        if np.isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            if logger is not None:
                logger.warning("Divide-by-zero encountered: rhok assumed large")
        
        # the update didn't move
        if (yy == 0.):
            raise LBFGSStepError("no difference in gradient norm; cannot compute LBFGS step")
        
        # set the zero element, corresponding to this iteration
        self.s[0] = sk
        self.y[0] = yk
        self.rho[0] = rhok
        self.H0k = 1.0 / (rhok*yy)

class LBFGS(object):
    """
    Class to implement the limited-memory BFGS algorithm for 
    nonlinear optimization
    
    see: https://en.wikipedia.org/wiki/Limited-memory_BFGS
    """
    logger = logging.getLogger("LBFGS")
    
    def __init__(self, f, fprime, p0, args=[], kwargs={}, M=100, unscaler=None):
        """
        Parameters
        ----------
        f : callable
            the function to minimize
        fprime : callable
            the gradient of `f`; returning an array with shape (n,)
            where `n` is the number of parameters
        p0 : array_like, (n, )
            the array of initial parameter values to start from
        args : sequence, optional
            a list of additional arguments to pass to `f` and `fprime`, 
            where both functions will be called `f(x, *args, **kwargs)`
        kwargs : dict, optional
            a list of additional keywords to pass to `f` and `fprime`
        M : int, optional 
            the number of previous iterations to use in determining the 
            optimal LBFGS step
        unscaler : callable, optional
            a function that unscales the parameters back to their original
            values
        """
        self.f        = f
        self.fprime   = fprime
        self.M        = int(M)
        self.args     = args
        self.kwargs   = kwargs
        self.unscaler = unscaler
        
        # set the default options
        self.options = {}
        default = LBFGS.default_options()
        for k in default:
            self.options.setdefault(k, default[k])

        self.p0 = p0.copy()
        self.N  = len(self.p0)
        
        if self.M <= 0:
            raise ValueError("LBFGS parameter `M` must be a positive integer")
                
        # store the relevant information in a dictionary
        self.data = {}
        self.data['iteration'] = 0
        self.data['funcalls']  = 0
        self.data['status']    = 0
        
        # the inverse Hessian
        self.data['H'] = LimitedMemoryInverseHessian(1., (self.M, self.N))

        # the current state
        X  = self.p0.copy()
        F, G = self.f(self.p0), self.fprime(self.p0)
        self.data['curr_state'] = State(X, F, G)
                
        # save the previous state
        self.data['prev_state'] = self.data['curr_state'].copy()
            
    @classmethod
    def from_restart(cls, f, fprime, data, **kwargs):
        """
        Initialize a new ``LBFGS`` class, using ``data`` from 
        an exisiting class
        """
        p0 = data['curr_state']['X']
        kwargs['M'] = len(data['H'])
        obj = cls(f, fprime, p0, **kwargs)
        obj.data = data
        return obj
        
    @classmethod
    def default_options(cls):
        """
        Return a dictionary of the default options for the LBFGS 
        optimization algorithm
        
        Stopping criteria based on tolerance of objective function
        and its gradient: 
        
        1) ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``
            where ``eps`` is the machine precision
        2) ``max(abs(G_k)) <= gtol``
        
        Parameters
        ----------
        factr : float
            The iteration stops when ``(f^k - 
            f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``, where ``eps``
            is the machine precision, which is automatically generated by
            the code. Typical values for `factr` are: 1e12 for low
            accuracy; 1e7 for moderate accuracy; 10.0 for extremely high
            accuracy. Default is 1e7
        gtol : float
            the absolute tolerance of the gradient norm required for
            convergence; default is 2000
        max_iter : int
            the maximum number of steps to run; default is `500`
        display : int
            the level of logging desired; default is second to highest, `2`
        record : list of str
            the names of variables to track per iteration; default
            is `F` and `Gnorm`
        """
        default = {}
        default['factr']   = 1e5
        default['gtol']    = 1e-5
        default['max_iter'] = 500
        default['display'] = 2
        default['record']  = ['F', 'Gnorm']
        default['test_convergence'] = True

        return default
        
    def record_progress(self, **args):
        """
        Record the progress of each iteration for all variables
        specified in `options['record']`
        """
        if not args:
            args = self.options['record']
            
        if not hasattr(self, 'history'):
            self.history = {arg: [] for arg in args}
            
        curr = self.data['curr_state']
        for arg in args:
            if arg not in curr:
                raise ValueError("`%s` can not be recorded from the current state" %arg)
            v = curr[arg]
            if hasattr(v, 'copy'): v = v.copy()
            self.history[arg].append(v)
            
    def check_convergence(self):
        """
        Test the convergence of the current state
        """
        curr   = self.data['curr_state']
        prev   = self.data['prev_state']
        opt    = self.options
        self.data['status'] = 0
        
        # if on iteration 0, just return
        if self.data['iteration'] == 0:
            return self.data['status']
        
        # exceed maximum iterations
        if self.data['iteration'] >= opt['max_iter']:
            self.data['status'] = -1
            return self.data['status']

        # if not testing convergence, return
        if not opt['test_convergence']:
            return self.data['status']
                       
        # check tolerance of objective function
        if 'factr' in opt and opt['factr'] is not None:
            
            delta = abs(curr['F'] - prev['F'])
            max_val = max(curr['F'], prev['F'])
            self.logger.info("convergence test: %.4e <= %.4e" %(delta / max(max_val, 1.), opt['factr'] * _epsilon))
            if delta / max(max_val, 1.) <= opt['factr'] * _epsilon:
                self.logger.info("objective function has reached the required precision")
                self.data['status'] = 2
                return self.data['status']
        
        # absolute tolerance of gradient norm
        if 'gtol' in opt and opt['gtol'] is not None:
            
            # inf norm of gradient
            gnorm = np.amax(np.abs(curr['G']))    
            if gnorm <= opt['gtol']:
                self.logger.info("gradient inf norm is now below the required tolerance")
                self.data['status'] = 3
                return self.data['status']
        
        return self.data['status']

    @property
    def parameter_status(self):
        """
        Information about the parameters for the current iteration
        """
        d = self.data
        curr = d['curr_state']
        
        values = curr['X']
        if self.unscaler is not None:
            values = self.unscaler(values)
        values = "   ".join(["%15.6f" %th for th in values])
        X = "%.4d   %s   %15.6f" %(d['iteration'], values, curr['F'])
        return X

    @property
    def gradient_status(self):
        """
        Information about the gradient vector for the current iteration
        """
        d = self.data
        curr = d['curr_state']
        values = "   ".join(["%15.6f" %g for g in curr['G']])
        G = "%.4d   %s   %15.6f" %(d['iteration'], values, curr['Gnorm'])
        return G
    
    @property
    def convergence_status(self):
        """
        The current convergence status
        """
        reasons = {-1: 'Maximum number of iterations reached.',
                    2: 'Tolerance reached: deltaF/F < ftol.',
                    3: 'Tolerance reached: inf norm of gradient < gtol.',
                   -4: 'Linesearch failed in both LBFGS and gradient directions',
                   -5: 'Algorithm has not moved, stopping',
                    0: 'Algorithm has not yet converged, with no errors so far'}
        return reasons[self.data['status']]
                    
    def compute_LBFGS_step(self, it, a):
        """
        Compute the parameter step direction, using the LBFGS algorithm
        
        This closely follows the notation here:
        https://en.wikipedia.org/wiki/Limited-memory_BFGS
        
        Parameters
        ----------
        it : int
            the current iteration step
        a : array_like, (M,)
            this is a scratch workspace to compute `alpha`, 
            and is passed in so we don't have to allocate for every step
            
        Returns
        -------
        z : array_like, (M,)
            the LBFGS step direction 
        """
        k = min(self.M, it)
        H = self.data['H']
        state = self.data['curr_state']
        q = state['G'].copy() # this is the memory that is returned
 
        for i in range(k):
            a[i] = H['rho'][i] * np.dot(H['s'][i,:], q)
            q -= a[i] * H['y'][i,:]
            
        # q is not used anymore after this, so we can use it as workspace
        z = q*H['H0k']
        
        # normalize first step
        if it == 0:
            z /= state['Gnorm']
        
        for i in reversed(range(k)):
            beta = H['rho'][i] * np.dot(H['y'][i,:], z)
            z += H['s'][i, :] * (a[i] - beta)
            
        return z
    
    def run_nlopt(self, **options):
        """
        Run the full nonlinear optimization, stopping only when
        convergence is reached or the maximum number of iterations 
        has been exceeded
        
        See `LBFGS.default_options()` for default keyword options
        """
        state = self.data['curr_state']
        for state in self.minimize(**options):
            pass
            
        return state
        
    def do_linesearch(self, state, pk):
        """
        Use scipy's backtracking Armijo line search to find the
        new search parameters
        
        Parameters
        ----------
        state : State
            the current state object
        pk : array_like
            the search direction
        
        Returns
        -------
        newX : array_like
            the new parameters satisfying the Armijo condition
        newF : float
            the value of the objective function at ``newX``
        
        Raises
        ------
        LineSearchError :
            cannot find suitable parameters that minimizes objective in
            this direction
        """      
        args = (self.f, state['X'], pk, state['G'], state['F'])
        alpha, _, newF = opt.linesearch.line_search_armijo(*args)
        
        # failed to find new minimum value
        if alpha is None: raise LineSearchError
        
        # update the search parameters
        newX = state['X'] + alpha * pk
        
        return newX, newF
        
    def minimize(self, **options):
        """
        Advance the minimization algorithm ``max_iter`` steps as a generator
        
        See `LBFGS.default_options()` for default keyword options
        """
        self.logger.info("options = %s" %options)
        self.options.update(**options)
        
        # scratch workspace for alpha in LBFGS step
        a = np.zeros(self.M)
        
        # some shortcuts
        d = self.data
        state = d['curr_state']
        
        # loop while convergence/errors are fine
        while self.check_convergence() == 0:
            
            # logging
            if self.options['display'] >= 2:
                self.logger.log(PARAM_LEVEL, self.parameter_status)
            if self.options['display'] >= 3:
                self.logger.log(DERIV_LEVEL, self.gradient_status)
                
            # record the progess
            self.record_progress()
            
            # find the LBGS step direction
            z = self.compute_LBFGS_step(d['iteration'], a)
                                
            # check angle between gradient and step direction
            costheta = np.dot(z, state['G']) / np.linalg.norm(z) / state['Gnorm']
            steepest_descent = False
            if costheta < 0.01:
                
                # log a warning
                theta = np.arccos(costheta) * 180./np.pi
                args = (costheta, theta)
                warn = "LBFGS iteration %d: the descent direction does not have a sufficient " %d['iteration']
                warn += "projection (cos(theta)=%.2e, theta=%.4f) into the gradient; using steepest descent at this step!" %args
                self.logger.info(warn)
                
                # use the gradient as search direction
                steepest_descent = True
                z[:] = state['G'] / state['Gnorm']
                
                # reset the Hessian if direction strayed from gradient
                self.data['H'] = LimitedMemoryInverseHessian(1., (self.M, self.N))
                
            # do the linesearch
            try:   
                newX, newF = self.do_linesearch(state, -z)
            
            except LineSearchError:
                
                # steepest descent already failed
                if steepest_descent:
                    d['status'] = -4
                    break
                else:
                    # try line search along steepest descent      
                    z[:] = state['G'] / state['Gnorm']
                    try:
                        newX, newF = self.do_linesearch(state, -z)
                        self.data['H'] = LimitedMemoryInverseHessian(1., (self.M, self.N))
                    except:
                        d['status'] = -4
                        break
                    
            # update the states
            d['prev_state'].update(*d['curr_state'])
            d['curr_state'].update(newX, newF, self.fprime(newX))
                    
            # difference in search parameters and gradient                
            sk = state['X'] - d['prev_state']['X']
            yk = state['G'] - d['prev_state']['G']
            
            # take the LBFGS step
            try:
                d['H'].update(sk, yk, logger=self.logger)
            except LBFGSStepError as e:
                self.logger.warning("error taking the LBFGS step: %s" %str(e))
                d['status'] = -5
                break

            d['iteration'] += 1
            yield state
    
        if self.options['display'] >= 1:
            self.logger.info(self.convergence_status)
            self.logger.log(PARAM_LEVEL, self.parameter_status)
            
    
if __name__ == '__main__':
    
    try: import numdifftools
    except: raise ImportError("install ``numdifftools`` to run LBFGS example")
    import scipy.optimize
    import time
    
    # set up console logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
                        
    x_true = np.arange(0,10,0.1)
    m_true = 2.5
    b_true = 1.0
    y_true = m_true*x_true + b_true

    def func(params, x, y):
        m, b = params
        y_model = m*x+b
        error = y-y_model
        return np.sum(error**2)
    
    grad = numdifftools.Gradient(func, step=1e-8)
    def fprime(params, x, y):
        return grad(params, x, y)

    def f(params):
        return func(params, x_true, y_true)
        
    def g(params):
        return fprime(params, x_true, y_true)
        
    initial_values = np.array([1.0, 0.0])
    
    minimizer = LBFGS(f, g, initial_values)
    start = time.time()
    for state in minimizer.minimize(display=3, record=['X', 'F']):
        pass
    stop = time.time()
    print(minimizer.history)
    print("done my LBFGS in %f" %(stop-start))
    
    start = time.time()
    res = scipy.optimize.fmin_l_bfgs_b(func, x0=initial_values, args=(x_true,y_true), approx_grad=True)
    stop = time.time()
    print("done scipy.optimize LBFGS in %f" %(stop-start))
    print(res)
        
        
        
        
