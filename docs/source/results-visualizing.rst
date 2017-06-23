.. currentmodule:: pyRSD.rsdfit

Visualizing the Results
=======================

Comparing the Best-fit Model to Data
------------------------------------

The best-fitting theory can be compared to the data visually by loading the
results of a fitting run using the :class:`pyRSD.rsdfit.FittingDriver`
and using the :class:`FittingDriver.plot` function. This function
will plot the data and best-fitting power spectra, properly
normalized by a linear power spectrum. For example,

.. code-block:: python

    from pyRSD.rsdfit import FittingDriver

    # load the model and results into one object
    d = FittingDriver.from_directory('pyRSD-example', model_file='pyRSD-example/model.npy', results_file='pyRSD-example/nlopt_result.npz')

    # set the fit results
    d.set_fit_results()

    # make a plot of the data vs the theory
    d.plot()
    show()

.. image:: _static/periodic-poles-plot.png
    :align: center

.. ipython:: python
    :suppress:

    import os
    from matplotlib import pyplot as plt

    startdir = os.path.abspath('.')
    home = startdir.rsplit('docs' , 1)[0]
    os.chdir(home); os.chdir('docs/data')

Visualizng MCMC Chains
----------------------

.. currentmodule:: pyRSD.rsdfit.results

The user can plot 2D correlations between parameters using the
:func:`EmceeResults.kdeplot_2d` function, which uses the :func:`seaborn.kdeplot`
function,

.. ipython:: python

    from pyRSD.rsdfit.results import EmceeResults
    r = EmceeResults.from_npz('mcmc_result.npz')

    # 2D kernel density plot
    r.kdeplot_2d('b1_cA', 'fsigma8', thin=10)

    @savefig kdeplot.png width=6in
    plt.show()

A joint 2D plot of the MCMC chains with 1D histograms can be plotted using the
:func:`EmceeResults.jointplot_2d`, which uses the :func:`seaborn.jointplot`
function

.. ipython:: python

    # 2D joint plot
    r.jointplot_2d('b1_cA', 'fsigma8', thin=10)

    @savefig jointplot.png width=6in
    plt.show()

In order to investigate whether the chains have converged, the user can
plot the timeline of the MCMC chain for a given parameter using
the :func:`EmceeResults.plot_timeline` function

.. ipython:: python

    # timeline plot
    r.plot_timeline('fsigma8', 'b1_cA', thin=10)

    @savefig timeline.png width=6in
    plt.show()

The correlation matrix between parameters can be plotted using the
:func:`EmceeResults.plot_correlation` function, which uses the
:func:`seaborn.heatmap` function

.. ipython:: python

    # correlation between free parameters
    r.plot_correlation(params='free')

    @savefig correlation.png width=8in
    plt.show()

And, finally, a triangle plot of 2D and 1D histograms for the desired
parameters can be produced using the :func:`EmceeResults.plot_triangle`,
which relies on the :func:`corner.corner` function

.. ipython:: python

    # make a triangle plot
    r.plot_triangle('fsigma8', 'alpha_perp', 'alpha_par', thin=10)

    @savefig triangle.png width=8in
    plt.show()

.. ipython:: python
    :suppress:

    os.chdir(startdir)
