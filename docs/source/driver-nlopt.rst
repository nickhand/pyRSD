.. _nlopt-solver:

Nonlinear Optimization
======================

The pyRSD package includes a LBFGS solver to find the
maximum a posteriori probability (MAP) estimates of the best-fit
theory parameters.


Command-line Options
~~~~~~~~~~~~~~~~~~~~

The MAP parameter values can be found by passing the ``nlopt`` sub-command
to the ``rsdfit`` executable. The calling sequence for the ``nlopt`` command is

.. command-output:: rsdfit nlopt -h

The main options are the parameter file, passed by the ``-p`` option,
the directory to save results, passed by the ``-o`` option, and the
name of a model to load, passed by the ``-m`` file. In addition, the
are maximum number of optimaztion steps should be passed via
**-i, ---iterations** flag.

Initializing the NLOPT Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method used to initialize either the MCMC chains can be configured
using the ``driver.init_from`` parameter. The allowed values of this parameter
when using the MCMC solver are:

1. **fiducial** :

    Initialize the parameters to their fiducial values, specified in the
    parameter file via the ``fiducial`` keyword for each free parameter

2. **result** :

    Initialize the free parameters the values of the parameters from
    from a previous result. In this case, the ``driver.start_from``
    parameter should give the name of a ``.npz``, which
    can be loaded into either a :class:`pyRSD.rsdfit.results.LBFGSResults`
    or :class:`pyRSD.rsdfit.results.EmceeResults` object.

Additionally, the ``driver.init_scatter`` can be set to add random scatter
drawn from a normal distributition with mean zero and standard deviation
set by the value of ``driver.init_scatter``. Specifically, this parameter
gives the percent scatter to add, relative to the value of the parameter's
fiducial value.

Parameter Derivatives
~~~~~~~~~~~~~~~~~~~~~

The LBFGS algorithm requires the derivatives of the model with respective
to the free parameters, and analytic derivatives of nearly all of the parameters
in the default parametrization of the :class:`~pyRSD.rsd.GalaxySpectrum` model
are builtin into the package. However for some parameters, it is necessary
to estimate their derivatives numerically. To do so, the NLOPT solver
uses a central-difference numerical derivative with the step-size set by
the user, depending on the value of the ``driver.lbfgs_epsilon`` parameter.

The ``driver.lbfgs_epsilon`` parameter can be specified as a single float,
in which case this step size will be used for all parameters that require
numerical derivatives. Alternatively, the parameter can be specified as a
dictionary, providing different values of the step size for different parameters.
This is particularly useful for parameters that have drastically different
magnitudes in order to avoid numerical instabilities.

The Stopping Criteria
~~~~~~~~~~~~~~~~~~~~~

The LBFGS algorithm will stop when either the number of iterations has been reached,
or if ``driver.test_convergence`` is set to ``True``, when any of the
convergence criteria are satisfied. These criteria can be specified by the
user by adjusting the ``driver.lbfgs_options`` parameter. This parameter
is a dictionary of options with the following keys:

1. **factr** :

    Stopping criterion based on the value of the objective function,
    given by ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``
    where ``eps`` is the machine precision. Typical values for ``factr`` are:
    1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high
    accuracy. Default is 1e5.

2. **gtol** :

    Stopping criterion based on the absolute value of the gradient norm
    of the objective function, given by ``max(abs(G_k)) <= gtol``. Default
    is 1e-5.

Recommended Practices
~~~~~~~~~~~~~~~~~~~~~

For the default models and parametrization, the LBFGS algorithm typically
needs ~200 or so iterations to converge to the best-fit parameters. We recommend
at least this number of iterations in order to avoid converging to a local
maximum of the likelihood function. If the user is worried about
local optima, multiple NLOPT runs can be executed, initializing from
different fiducial values each time.

The user can also test the sensitivity of the best-fit parameter values
to the assumed prior distributions by setting the ``driver.lbfgs_use_priors``
to ``False``. This will run a maximum likelihood parameter estimation, ignoring
the prior probabilities. In nearly all cases, the MAP and maximum likelihood
results should agree, unless for some reason the prior distribution is the
dominant constraint on a given parameter.
