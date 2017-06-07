"""
This module will reproduce Figure 1 of Vlah et al. 2013., which
shows the various Kmn integrals
"""
from .utils import new_axes
from . import cache_manager
from pyRSD import data_dir

import pytest
from matplotlib import pyplot as plt
import numpy
import os

@pytest.fixture(scope='module')
def model():

    from pyRSD.rsd import HaloSpectrum

    # inititalize the model
    config                       = {}
    config['include_2loop']      = False
    config['transfer_fit']       = 'CLASS'
    config['params']             = 'teppei_sims.ini'
    config['max_mu']             = 4
    config['use_P00_model']      = False
    config['use_P01_model']      = False
    config['use_P11_model']      = False
    config['use_Pdv_model']      = False
    config['interpolate']        = False
    config['vel_disp_from_sims'] = True
    m = HaloSpectrum(**config)

    # load the model
    with cache_manager(m, "vlahetal_halo.npy") as model:

        # set klims
        model.kmin = 0.01
        model.kmax = 0.5

        # update redshift-dependent quantities
        model.z        = 0.
        model.sigma8_z = model.cosmo.Sigma8_z(0.)
        model.f        = model.cosmo.f_z(0.)

    return model

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=10)
def test_Kmn_0(model):
    """
    Reproduce top left panel of Figure 1.
    """
    # axis limits
    xlims = (0.01, 1.0)
    ylims = (1., 1e3)
    ylabel = r"$P [Mpc/h]^3}$"

    # k array
    k = numpy.logspace(-2, 0, 1000)

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    plt.loglog(k, model.K00(k), c='b', label=r'$K_{00}$')
    plt.loglog(k, -model.K00s(k), c='b', ls='--', label=r"$-K_{00,s}$")

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_Kmn_1(model):
    """
    Reproduce top right panel of Figure 1.
    """
    # axis limits
    xlims = (0.01, 1.0)
    ylims = (10., 1e4)
    ylabel = r"$P \ \mathrm{[Mpc/h]^3}$"

    # k array
    k = numpy.logspace(-2, 0, 1000)

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    plt.loglog(k, model.K01(k), c='b', label=r'$K_{01}$')
    plt.loglog(k, model.K01s(k), c='b', ls='dashdot', label=r"$K_{01,s}$")
    plt.loglog(k, model.K02s(k), c='b', ls='dashed', label=r"$K_{02,s}$")

    ax.legend(loc=0, fontsize=12)
    return fig


@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_Kmn_2(model):
    """
    Reproduce bottom left panel of Figure 1.
    """
    # axis limits
    xlims = (0.01, 1.0)
    ylims = (1, 2.5e3)
    ylabel = r"$P \ \mathrm{[Mpc/h]^3}$"

    # k array
    k = numpy.logspace(-2, 0, 1000)

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    plt.loglog(k, model.K11(k), c='b', label=r'$K_{11}$')
    plt.loglog(k, abs(model.K10(k)), c='b', ls='dashdot', label=r"$|K_{10}|$")
    plt.loglog(k, abs(model.K11s(k)), c='b', ls='dashed', label=r"$|K_{11,s}|$")
    plt.loglog(k, -model.K10s(k), c='b', ls='dotted', label=r"$-K_{10,s}$")

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_Kmn_3(model):
    """
    Reproduce bottom right panel of Figure 1.
    """
    # axis limits
    xlims = (0.01, 1.0)
    ylims = (10., 1e6)
    ylabel = r"$P \ \mathrm{[Mpc/h]^3}$"

    # k array
    k = numpy.logspace(-2, 0, 1000)

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    A = 1./k**2
    plt.loglog(k, -A*model.K20_a(k), c='b', ls='dashed', label=r'$-K_{20}[\mu^2]$')
    plt.loglog(k, A*abs(model.K20s_a(k)), c='b', label=r'$|K_{20,s}[\mu^2]|$')
    plt.loglog(k, A*model.K20_b(k), c='b', ls='dotted', label=r'$K_{20}[\mu^4]$')
    plt.loglog(k, -A*model.K20s_b(k), c='b', ls='dashdot', label=r'$-K_{20,s }[\mu^4]$')

    ax.legend(loc=0, fontsize=12)
    return fig
