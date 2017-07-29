.. _data-formats:

.. currentmodule:: pyRSD.rsd.data

File Formats for Data Files
===========================

.. ipython:: python

    import os

    startdir = os.path.abspath('.')
    home = startdir.rsplit('docs' , 1)[0]
    os.chdir(home);
    os.chdir('docs/source')

    if not os.path.exists('generated'):
      os.makedirs('generated')
    os.chdir('generated')

The data statistics, covariance matrix, and grid file are all read
from plaintext files by the :mod:`pyRSD.rsdfit` module, and
each requires a special format. We will describe those file file formats
in this section.


The Power Statistics
--------------------

The plaintext file holding the data statistics must be specified in the
parameter file using the ``data.data_file`` parameter. The package
can handle reading either :math:`P(k,\mu)` or :math:`P_\ell(k)` data, and
we will give examples for both below.

:math:`P(k,\mu)` Data
~~~~~~~~~~~~~~~~~~~~~

For :math:`P(k,\mu)` data, we have power measured in wide bins of :math:`\mu`.
If there are :math:`N_k` :math:`k` bins and :math:`N_\mu` :math:`\mu` bins,
the pyRSD code expects the following data:

1. **k** :

    an array of shape (:math:`N_k`, :math:`N_\mu`) providing the mean wavenumber value in each bin (units: :math:`h/\mathrm{Mpc}`)

2. **mu** :

    an array of shape (:math:`N_k`, :math:`N_\mu`) providing the mean :math:`\mu` value in each bin

3. **power** :

    an array of shape (:math:`N_k`, :math:`N_\mu`) providing the mean power value in each bin (units: :math:`h^{-3} \mathrm{Mpc}^3`)

The data can be saved easily to a plaintext file from a numpy structured array
using the :class:`pyRSD.rsdfit.data.PowerMeasurements` class. For example,
here we will generate some fake :math:`P(k,\mu)` and write this data to a
plaintext file with the correct format.

.. ipython:: python

    from pyRSD.rsdfit.data import PowerMeasurements
    import numpy as np

    Nk = 20
    Nmu = 5

    # generate 5 mu bins from 0 to 1
    mu_edges = np.linspace(0, 1, Nmu+1)
    mu_cen = 0.5 * (mu_edges[1:] + mu_edges[:-1])

    # generate 20 k bins from 0.01 to 0.4
    k_edges = np.linspace(0.01, 0.4, Nk+1)
    k_cen = 0.5 * (k_edges[1:] + k_edges[:-1])

    # make 2D k, mu arrays with shape (20, 5)
    k, mu = np.meshgrid(k_cen, mu_cen, indexing='ij')

    # some random power data
    pkmu = np.random.random(size=(Nk,Nmu))

    # make a structured array
    data = np.empty((Nk,Nmu), dtype=[('k', 'f8'), ('mu', 'f8'), ('power', 'f8')])
    data['k'] = k[:]
    data['mu'] = mu[:]
    data['power'] = pkmu[:]

    # identifying names for each statistic
    names = ['pkmu_0.1', 'pkmu_0.3', 'pkmu_0.5', 'pkmu_0.7', 'pkmu_0.9']

    # initialize the PowerMeasurements object
    measurements = PowerMeasurements.from_array(names, data)
    measurements

    # save to file
    measurements.to_plaintext("pkmu_data.dat")

Our fake data has been saved to a plaintext text file with the desired format.
The first few lines of this plaintext file look like:

.. literalinclude:: generated/pkmu_data.dat
    :lines: 1-10

We can easily re-initialize a :class:`PowerMeasurements` object from this plaintext
file using

.. ipython:: python

    names = ['pkmu_0.1', 'pkmu_0.3', 'pkmu_0.5', 'pkmu_0.7', 'pkmu_0.9']
    measurements_2 = PowerMeasurements.from_plaintext(names, 'pkmu_data.dat')
    measurements_2

.. note::

    The ``names`` specified in this example serve as identifying strings for
    each power bin. They should be specified via the :attr:`data.statistics`
    keyword in the parameter file. For :math:`P(k,\mu)` data, the names
    must begin with ``pkmu_`` and are typically followed by the mean :math:`\mu`
    value in each power bin, i.e., ``pkmu_0.1`` identifies the statistic
    measuring :math:`P(k,\mu)` in a bin centered at :math:`\mu=0.1`.

:math:`P_\ell(k)` Data
~~~~~~~~~~~~~~~~~~~~~~

For :math:`P_\ell(k)` data, we have measured the multipoles moments of
:math:`P(k,\mu)` for several multipole numbers :math:`\ell`.
If there are :math:`N_k` :math:`k` bins and :math:`N_\ell` multipoles,
the pyRSD code expects the following data:

1. **k** :

    an array of shape (:math:`N_k`, :math:`N_\ell`) providing the mean wavenumber for each multipole (units: :math:`h/\mathrm{Mpc}`)

3. **power** :

    an array of shape (:math:`N_k`, :math:`N_\ell`) providing the mean power for each multipole (units: :math:`h^{-3} \mathrm{Mpc}^3`)

Again, here we will generate some fake :math:`P_\ell(k)` and write this data to a
plaintext file with the correct format as an example.

.. ipython:: python

    Nk = 20
    Nell = 3

    # fit the monopole, quadrupole, and hexadecapole
    ells = [0, 2, 4]

    # generate 20 k bins from 0.01 to 0.4
    k_edges = np.linspace(0.01, 0.4, Nk+1)
    k_cen = 0.5 * (k_edges[1:] + k_edges[:-1])

    # make a structured array
    data = np.empty((Nk,Nell), dtype=[('k', 'f8'), ('power', 'f8')])

    for i, ell in enumerate(ells):
        data['k'][:,i] = k_cen[:]
        data['power'][:,i] = np.random.random(size=Nk)

    # identifying names for each statistic
    names = ['pole_0', 'pole_2', 'pole_4']

    # initialize the PowerMeasurements object
    measurements = PowerMeasurements.from_array(names, data)
    measurements

    # save to file
    measurements.to_plaintext("pole_data.dat")

Our fake data has been saved to a plaintext text file with the desired format.
The first few lines of this plaintext file look like:

.. literalinclude:: generated/pole_data.dat
    :lines: 1-10

We can easily re-initialize a :class:`PowerMeasurements` object from this plaintext
file using

.. ipython:: python

    names = ['pole_0', 'pole_2', 'pole_4']
    measurements_2 = PowerMeasurements.from_plaintext(names, 'pole_data.dat')
    measurements_2

.. note::

    The ``names`` specified in this example serve as identifying strings for
    each multipole. They should be specified via the :attr:`data.statistics`
    keyword in the parameter file. For :math:`P_\ell(k)` data, the names
    must begin with ``pole_`` and are typically followed by the multipole
    number, i.e., ``pole_0`` identifies the monopole (:math:`\ell=0`).

.. _covariance-matrix:

The Covariance Matrix
---------------------

The parameter estimation procedure relies on the likelihood function, which tells
use the probability of the measured data given our theoretical model. In order
to evaluate the likelihood, we need the covariance matrix of the data statistics.
pyRSD provides two classes for dealing with covariance matrices,
:class:`pyRSD.rsdfit.data.PkmuCovarianceMatrix` for :math:`P(k,\mu)` data and
:class:`pyRSD.rsdfit.data.PoleCovarianceMatrix` for :math:`P_\ell(k)` data.

Again, when running the ``rsdfit`` command, the covariance matrix will be
loaded from a plaintext file with a specific format. The easiest way to
save your desired covariance matrix is take advantage of the functionality
provided by the :class:`~pyRSD.rsdfit.data.PkmuCovarianceMatrix` and
:class:`pyRSD.rsdfit.data.PoleCovarianceMatrix` classes.

The most important thing to realize when dealing with the covariance matrix
is that it is the covariance matrix of the full, concatenated data vector.
For example, if we are fitting the :math:`\ell=0,2,4` multipoles, then
the full data vector is

.. math::

    \mathcal{D} = [P_0, P_2, P_4].

Then, if we have :math:`N_k` data points measured for each multipole, then
the covariance matrix has a size of :math:`(3 N_k, 3 N_k)`. The upper left
sub-matrix of size :math:`(N_k, N_k)` is the covariance of :math:`P_0` with itself,
the next sub-matrix of size :math:`(N_k, N_k)` to the right is the covariance
between :math:`P_0` and :math:`P_2` and similarly, the last sub-matrix on
the top row is the covariance between :math:`P_0` and :math:`P_4`.

As an example, we will generate some fake multipole data and compute the
covariance below,

.. ipython:: python

    from pyRSD.rsdfit.data import PoleCovarianceMatrix

    # generate 100 fake monopoles
    P0 = np.random.random(size=(100, Nk)) # Nk = 20, see above

    # 100 fake quadrupoles
    P2 = np.random.random(size=(100, Nk))

    # 100 fake hexadecapoles
    P4 = np.random.random(size=(100, Nk))

    # make the full data vector
    D = np.concatenate([P0, P2, P4], axis=-1) # shape is (100, 60)

    # compute the covariance matrix
    cov = np.cov(D, rowvar=False) # shape is (20,20)

    # initialize the PoleCovarianceMatrix
    ells = [0,2,4]
    C = PoleCovarianceMatrix(cov, k_cen, ells)
    C

    # write to plaintext file
    C.to_plaintext('pole_cov.dat')

The covariance matrix of our fake data has been saved to a plaintext text file
with the desired format. The first few lines of this plaintext file look like:

    .. literalinclude:: generated/pole_cov.dat
        :lines: 1-10

We can easily re-initialize a :class:`PoleCovarianceMatrix` object from this plaintext
file using

.. ipython:: python

    names = ['pole_0', 'pole_2', 'pole_4']
    C_2 = PoleCovarianceMatrix.from_plaintext('pole_cov.dat')
    C_2

The case of :math:`P(k,\mu)` data is very similar to multipoles, except now
the second dimension specifies the :math:`\mu` bins, rather than the
multipole numbers. The class :class:`PkmuCovarianceMatrix` should be used
for :math:`P(k,\mu)` data.

.. warning::

    Be sure to ensure that the number of data points, specifically the
    :math:`k` range used, of the data statistics in the file specified by the
    :attr:`data.data_file` attribute agrees with the number of elements in
    the covariance matrix.

The Window Function
-------------------

If the user wishes to fit window-convolved power spectrum multipole, a
file holding the correlation function multipoles of the window function should
be specified using the :attr:`data.window_file` parameter. Then, the theoretical
model will be convolved with this window function to accurately compare
model and data.

The window function file should hold several columns of data, with the first
column specifying the separation array :math:`s`, and the other columns
giving the even-numbered correlation function multipoles of the window.
See :ref:`window-power` for a full example of the window multipoles.

The :math:`P(k,\mu)` Grid
----------------------------

As discussed in :ref:`discrete-binning`, the user can optionally take into account
the effects of dicretely binned data with a finely-binned :math:`P(k,\mu)` grid.
The name of the file holding this grid should be specified in the parameter
file using the :attr:`data.grid_file` attribute.

Given a 2D data array specifies this grid, the easiest way to save the grid
to a plaintext file in the right format is to use the :func:`PkmuGrid.to_plaintext`
function. See :ref:`discrete-binning` for a full example of how to do this.

.. note::

    If the user is fitting window-convolved multipoles, the grid file does not
    need to be specified.

.. ipython:: python
    :suppress:

    os.chdir(startdir)
