import os
import numpy
from matplotlib import rc
from matplotlib import pyplot as plt
import os
import argparse

from pyRSD.rsd.cosmology import Planck15
from pyRSD.rsd.hzpt import HaloZeldovichP00, HaloZeldovichCF00
from pyRSD.rsd import GalaxySpectrum, PkmuTransfer, PolesTransfer, PkmuGrid

def savefig(filename, size=(6,3)):

    fig = plt.gcf()
    fig.set_size_inches(*size)
    fig.subplots_adjust(left=0.13, top=0.97, bottom=0.2, right=0.97)
    plt.savefig(os.path.join("source", "_static", filename), dpi=300)
    plt.clf()

def gal_power_compute(model):

    k = numpy.logspace(-2, numpy.log10(0.4), 100)
    mu = numpy.arange(0.1, 1.0, 0.2)
    Pkmu = model.power(k, mu)

    for i, imu in enumerate(mu):
      plt.loglog(k, Pkmu[:,i], label=r"$\mu = %.1f$" %imu)

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("pkmu_model_plot.png")

    ells = [0, 2, 4]
    Pell = model.poles(k, ells)

    for i, ell in enumerate(ells):
      plt.loglog(k, Pell[i], label=r"$\ell = %d$" %ell)

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P_\ell$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("poles_model_plot.png")


def gal_power_discrete(model):

    # set up fake 1D k, mu bins
    k_1d = numpy.arange(0.01, 0.4, 0.005)
    mu_1d = numpy.linspace(0, 1.0, 100)

    # convert to a 2D grid
    # shape is (78, 100)
    k, mu = numpy.meshgrid(k_1d, mu_1d, indexing='ij')

    # assign random weights for each bin for illustration purposes
    modes = numpy.random.random(size=k.shape)

    # simulate missing data
    missing = numpy.random.randint(0, numpy.prod(modes.shape), size=10)
    modes.flat[missing] = numpy.nan

    # initialize the grid
    grid = PkmuGrid([k_1d, mu_1d], k, mu, modes)

    # edges of the mu bins
    mu_bounds = [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    # the transfer function, with specified valid k range
    transfer = PkmuTransfer(grid, mu_bounds, kmin=0.01, kmax=0.4)

    # evaluate the model with this transfer function
    Pkmu_binned = model.from_transfer(transfer)

    # get the coordinate arrays from the grid
    k, mu = transfer.coords # this has shape of (Nk, Nmu)

    for i in range(mu.shape[1]):
      plt.loglog(k[:,i], Pkmu_binned[:,i], label=r"$\mu = %.1f$" % mu[:,i].mean())

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("pkmu_binned_plot.png")

    # the multipoles to compute
    ells = [0, 2, 4]

    # the transfer function, with specified valid k range
    transfer = PolesTransfer(grid, ells, kmin=0.01, kmax=0.4)

    # evaluate the model with this transfer function
    poles_binned = model.from_transfer(transfer) # shape is (78, 3)

    # get the coordinate arrays from the grid
    k, mu = transfer.coords # this has shape of (Nk, Nmu)

    for i, iell in enumerate(ells):
      plt.loglog(k[:,i], poles_binned[:,i], label=r"$\ell = %d$" % iell)

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P_\ell$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("poles_binned_plot.png")


def hzpt_overview(*args):

    # power spectrum at z = 0
    P00 = HaloZeldovichP00(Planck15, z=0.)

    # compute the full power and each term
    k = numpy.logspace(-2, 0, 100)
    Pk = P00(k)
    Pzel = P00.zeldovich(k)
    Pbb = P00.broadband(k)

    # and plot
    plt.loglog(k, Pk, label='full P00')
    plt.loglog(k, Pzel, label='Zeldovich term')
    plt.loglog(k, Pbb, label='broadband term')

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("P00_hzpt_plot.png")

    # correlation function at z = 0
    CF = HaloZeldovichCF00(Planck15, z=0.)

    # compute the full correlation and each term
    r = numpy.logspace(0, numpy.log10(150), 100)
    xi = CF(r)
    xi_zel = CF.zeldovich(r)
    xi_bb = CF.broadband(r)

    # and plot
    plt.loglog(r, r**2 * xi, label='full CF')
    plt.loglog(r, r**2 * xi_zel, label='Zeldovich term')
    plt.loglog(r, r**2 * xi_bb, label='broadband term')

    plt.legend(loc=0)
    plt.xlabel(r"$r$ $[h^{-1} \mathrm{Mpc}]$", fontsize=10)
    plt.ylabel(r"$r^2 \xi$ $[h^{-2} \mathrm{Mpc}^{2}]$", fontsize=10)
    savefig("CF_hzpt_plot.png")

def pygcl_overview(*args):

    from pyRSD.rsd.cosmology import Planck15
    from pyRSD import pygcl

    class_cosmo = Planck15.to_class()

    # initialize at z = 0
    Plin = pygcl.LinearPS(class_cosmo, 0)

    # renormalize to different SetSigma8AtZ
    Plin.SetSigma8AtZ(0.62)

    # evaluate at k
    k = numpy.logspace(-2, 0, 100)
    Pk = Plin(k)

    # plot
    plt.loglog(k, Pk, c='k')

    # format
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("Plin_plot.png")

    # density auto power
    P00 = pygcl.ZeldovichP00(class_cosmo, 0)

    # density - radial momentum cross power
    P01 = pygcl.ZeldovichP01(class_cosmo, 0)

    # radial momentum auto power
    P11 = pygcl.ZeldovichP11(class_cosmo, 0)

    # plot
    k = numpy.logspace(-2, 0, 100)
    plt.loglog(k, P00(k), label=r'$P_{00}^\mathrm{zel}$')
    plt.loglog(k, P01(k), label=r'$P_{01}^\mathrm{zel}$')
    plt.loglog(k, P11(k), label=r'$P_{11}^\mathrm{zel}$')

    # format
    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    savefig("Pzel_plot.png")

    # linear correlation function
    CF = pygcl.CorrelationFunction(Plin)

    # Zeldovich CF at z = 0.55
    CF_zel = pygcl.ZeldovichCF(class_cosmo, 0.55)

    # plot
    r = numpy.logspace(0, numpy.log10(150), 1000)
    plt.plot(r, r**2 * CF(r), label=r'$\xi^\mathrm{lin}$')
    plt.plot(r, r**2 * CF_zel(r), label=r'$\xi^\mathrm{zel}$')

    # format
    plt.legend(loc=0)
    plt.xlabel(r"$r$ $[h^{-1} \mathrm{Mpc}]$", fontsize=10)
    plt.ylabel(r"$r^2 \xi$ $[h^{-2} \mathrm{Mpc}^{2}]$", fontsize=10)
    savefig("cf_plot.png")

def gal_power_window(model):

    from pyRSD.rsd.window import WindowTransfer

    # load the window function correlation multipoles array from disk
    # first column is s, followed by W_ell
    Q = numpy.loadtxt('data/window.dat')

    # now plot
    for i in range(1, Q.shape[1]):
      plt.semilogx(Q[:,0], Q[:,i], label=r"$\ell=%d$" %(2*(i-1)))

    plt.legend(loc=0, ncol=2)
    plt.xlabel(r"$s$ $[\mathrm{Mpc}/h]$", fontsize=14)
    plt.ylabel(r"$Q_\ell$", fontsize=14)
    savefig("window_poles_plot.png")

    # the multipoles to compute
    ells = [0, 2, 4]

    model.kmin = 1e-4
    model.kmax = 0.7

    # the window transfer function, with specified valid k range
    transfer = WindowTransfer(Q, ells, grid_kmin=1e-3, grid_kmax=0.6)

    # evaluate the model with this transfer function
    Pell_conv = model.from_transfer(transfer) # shape is (78, 3)

    # get the coordinate arrays from the grid
    k, mu = transfer.coords # this has shape of (Nk, Nmu)

    Pell = model.poles(k[:,0], ells)

    for i, iell in enumerate(ells):
        label='unconvolved' if i == 0 else ""
        plt.loglog(k[:,0], Pell[i], c='k', label=label)
        plt.loglog(k[:,i], Pell_conv[:,i], label=r"$\ell = %d$" % iell)

    plt.legend(loc=0)
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$", fontsize=10)
    plt.ylabel(r"$P_\ell$ $[h^{-3} \mathrm{Mpc}^3]$", fontsize=10)
    plt.xlim(5e-3, 0.6)
    savefig("poles_conv_plot.png")

def visualizing_results(*args):

    from pyRSD.rsdfit.results import EmceeResults

    r = EmceeResults.from_npz('data/mcmc_result.npz')

    # 2D kernel density plot
    r.kdeplot_2d('b1_cA', 'fsigma8', thin=10)
    savefig('kdeplot.png', size=(8,6))

    # 2D joint plot
    r.jointplot_2d('b1_cA', 'fsigma8', thin=10)
    savefig('jointplot.png', size=(8,6))

    # timeline plot
    r.plot_timeline('fsigma8', 'b1_cA', thin=10)
    savefig('timeline.png', size=(8,6))

    # correlation between free parameters
    r.plot_correlation(params='free')
    savefig('correlation.png', size=(8,6))

    # make a triangle plot
    r.plot_triangle('fsigma8', 'alpha_perp', 'alpha_par', thin=10)
    savefig('triangle.png', size=(8,6))

if __name__ == '__main__':

    desc = 'make the documenation plots'
    parser = argparse.ArgumentParser(description=desc)

    choices = ['gal_power_compute', 'gal_power_discrete', 'gal_power_window', 'pygcl_overview', 'hzpt_overview', 'visualizing_results']
    parser.add_argument('which', choices=choices)
    ns = parser.parse_args()

    # use latex
    rc('text', usetex=True)

    # load the model
    model = GalaxySpectrum.from_npy('data/galaxy_power.npy')

    # make plots
    locals()[ns.which](model)
