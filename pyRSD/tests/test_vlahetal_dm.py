"""
This module will reproduce the plots in Vlah et al. 2012., as
a test of :class:`DarkMatterSpectrum`
"""
from .utils import new_axes
from . import cache_manager
from pyRSD import data_dir

import os
import pytest
from matplotlib import pyplot as plt

redshifts = [0., 0.509, 0.989, 2.070]

@pytest.fixture(scope='module', params=redshifts)
def redshift(request):
    return request.param

def update_model(model, redshift):

    # update redshift-dependent quantities
    model.z        = redshift
    model.sigma8_z = model.cosmo.Sigma8_z(redshift)
    model.f        = model.cosmo.f_z(redshift)

@pytest.fixture(scope='module')
def model():

    from pyRSD.rsd import DarkMatterSpectrum
    redshift = 0.0

    # inititalize the model
    config                   = {}
    config['include_2loop']  = True
    config['transfer_fit']   = 'CLASS'
    config['params']         = 'teppei_sims.ini'
    config['max_mu']         = 6
    config['use_P00_model']  = False
    config['use_P01_model']  = False
    config['use_P11_model']  = False
    config['use_Pdv_model']  = False
    config['interpolate']    = False
    config['z']              = redshift
    m = DarkMatterSpectrum(**config)

    # load the model
    with cache_manager(m, "vlahetal_dm.npy") as model:

        # set klims
        model.kmin = 0.01
        model.kmax = 0.5

        # update redshift-dependent quantities
        model.z        = redshift
        model.sigma8_z = model.cosmo.Sigma8_z(redshift)
        model.f        = model.cosmo.f_z(redshift)

    return model

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------
def get_sigma_bv2(z):
    """
    Return the value of `sigma_bv2` based on the input redshift
    """
    sigmas = [375., 356., 282., 144.]
    return sigmas[redshifts.index(z)]

def get_sigma_v2(z):
    """
    Return the value of `sigma_v2` based on the input redshift
    """
    sigmas = [209., 198., 159., 80.]
    return sigmas[redshifts.index(z)]

def get_sigma_bv4(z):
    """
    Return the value of `sigma_bv4` based on the input redshift
    """
    sigmas = [432., 382., 315., 144.]
    return sigmas[redshifts.index(z)]

#------------------------------------------------------------------------------
# TESTS
#------------------------------------------------------------------------------
@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P00(redshift, model):
    """
    Reproduce Figure 1.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0., 0.3)
    ylims = (0.8, 1.5)
    ylabel = r"$P_{00}^\mathrm{DM} \ / P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims, ylims)

    # normalization
    norm = model.normed_power_lin_nw(model.k)

    # plot the model (1-loop SPT)
    kws = {'c':'b', 'ls':'-', 'label':"1-loop SPT"}
    plt.plot(model.k, model.P00.mu0(model.k) / norm, **kws)

    # linear
    kws = {'c':'k', 'ls':':', 'label':"linear"}
    plt.plot(model.k, model.normed_power_lin(model.k) / norm, **kws)

    # zeldovich
    kws = {'c':'Magenta', 'ls':'dashdot', 'label':"Zel'dovich"}
    plt.plot(model.k, model.hzpt._P00.zeldovich(model.k) / norm, **kws)

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P01(redshift, model):
    """
    Reproduce Figure 2.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0., 0.3)
    ylims = (0.8, 1.5)
    ylabel = r"$P_{01}^\mathrm{DM} \ / P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # normalization
    A = 2*model.f
    norm = A*model.normed_power_lin_nw(model.k)

    # plot the model (1-loop SPT)
    kws = {'c':'b', 'ls':'-', 'label':"1-loop SPT"}
    plt.plot(model.k, model.P01.mu2(model.k) / norm, **kws)

    # linear
    kws = {'c':'k', 'ls':':', 'label':"linear"}
    plt.plot(model.k, A*model.normed_power_lin(model.k) / norm, **kws)

    # zeldovich
    kws = {'c':'Magenta', 'ls':'dashdot', 'label':"Zel'dovich"}
    plt.plot(model.k, A*model.hzpt._P01.zeldovich(model.k) / norm, **kws)

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P11_scalar(redshift, model):
    """
    Reproduce Figure 3.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0., 0.2)
    ylims = (0.8, 2.0)
    ylabel = r"$P_{11}^\mathrm{DM}[\mu^4] \ / P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # 1-loop results
    with model.preserve(include_2loop=False):

        # normalization
        A = model.f**2
        norm = A * model.normed_power_lin_nw(model.k)

        # plot the model
        kws = {'c':'b', 'ls':'-', 'label':"model"}
        plt.plot(model.k, model.P11.mu4.scalar(model.k) / norm, **kws)

        # linear
        kws = {'c':'k', 'ls':':', 'label':"linear"}
        plt.plot(model.k, A*model.normed_power_lin(model.k) / norm, **kws)

        ax.legend(loc=0, fontsize=12)
        return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P11(redshift, model):
    """
    Reproduce Figure 4.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0.01, 0.5)
    ylims = (1.0, 1e4)
    ylabel = r"$P_{11}^\mathrm{DM} \ \mathrm{[Mpc/h]^3}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # 1-loop scalar
    with model.preserve(include_2loop=False):
        plt.loglog(model.k, model.P11.mu4.scalar(model.k), label='1-loop, scalar', c='b')

    # 1-loop and 2-loop vector
    plt.loglog(model.k, model.P11.mu2.vector(model.k), label='2-loop, vector', color='r')
    with model.preserve(include_2loop=False):
        plt.loglog(model.k, model.P11.mu2.vector(model.k), label='1-loop, vector', color='r', ls='--')

    # 1-loop and 2-loop C11
    plt.loglog(model.k, model.f**2 * model.I13(model.k), label=r'1-loop $C_{11}[\mu^4]$', c='g')
    I1 = model.Ivvdd_h02(model.k)
    I2 = model.Idvdv_h04(model.k)
    plt.loglog(model.k, model.f**2 * (I1+I2), label=r'2-loop $C_{11}[\mu^4]$', c='g', ls='--')

    # linear
    kws = {'c':'k', 'ls':':', 'label':"linear"}
    plt.loglog(model.k, model.f**2*model.normed_power_lin(model.k), **kws)

    ax.legend(loc=0, fontsize=12, ncol=2)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P02(redshift, model):
    """
    Reproduce Figure 5.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0.01, 0.5)
    ylims = (-8, 2)
    ylabel = r"$P^\mathrm{DM}_{02} \ / \ (f \sigma_v k)^2 P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # normalization
    norm = (model.f*model.sigma_lin*model.k)**2*model.normed_power_lin_nw(model.k)
    model.sigma_bv2 = get_sigma_bv2(redshift)

    # 2 loop anisotropic
    P02_anisotropic = (2./3)*model.P02.mu4(model.k)
    plt.semilogx(model.k, P02_anisotropic/norm, c='g', ls='--', label="anisotropic")

    # 2 loop isotropic
    P02_isotropic = (1./3)*model.P02.mu4(model.k) + model.P02.mu2(model.k)
    plt.semilogx(model.k, P02_isotropic/norm, c='b', ls='dashdot', label=r'2-loop isotropic, $\sigma^2$')

    # 1 loop isotropic
    with model.preserve(include_2loop=False):
        model.sigma_bv2 = 0.
        P02_isotropic = (1./3)*model.P02.mu4(model.k) + model.P02.mu2(model.k)
        plt.semilogx(model.k, P02_isotropic/norm, label=r'1-loop isotropic, no $\sigma^2$', c='r')

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P12(redshift, model):
    """
    Reproduce Figure 6.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0.01, 0.5)
    ylims = (-7.5, 3.5)
    ylabel = r"$P^\mathrm{DM}_{12} \ / \ f^3 (\sigma_v k)^2 P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims, nticks=4)

    # normalization
    norm = model.f*(model.f*model.sigma_lin*model.k)**2*model.normed_power_lin_nw(model.k)

    # mu4 with small scale sigma
    model.sigma_bv2 = get_sigma_bv2(redshift)
    kws = {'label':r"2-loop $P_{12}[\mu^4]$, $\sigma^2$", 'c':'b', 'ls':'dashdot'}
    plt.semilogx(model.k, model.P12.mu4(model.k)/norm, **kws)

    # mu6 term
    kws = {'label':r"$P_{12}[\mu^6]$", 'c':'g', 'ls':'--'}
    plt.semilogx(model.k, model.P12.mu6(model.k)/norm, **kws)

    # 1 loop mu4 with no additional sigma
    with model.preserve(include_2loop=False):
        model.sigma_bv2 = 0.
        kws = {'label':r"1-loop $P_{12}[\mu^4]$, no $\sigma^2$", 'c':'r'}
        plt.semilogx(model.k, model.P12.mu4(model.k)/norm, **kws)

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P22(redshift, model):
    """
    Reproduce Figure 7.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0.01, 0.5)
    ylims = (1e-3, 10.)
    ylabel = r"$P^\mathrm{DM}_{22} \ / \ (f^2 \sigma_v k)^2 P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # normalization
    norm = (model.f**2*model.sigma_lin*model.k)**2*model.normed_power_lin_nw(model.k)

    # 1 loop mu4 and mu6
    model.sigma_bv2 = 0.
    with model.preserve(include_2loop=False):
        plt.loglog(model.k, abs(model.P22.mu4(model.k)/norm), c='r', lw=0.5)
        plt.loglog(model.k, abs(model.P22.mu6(model.k)/norm), ls='--', c='b', lw=0.5)

    # 2 loop mu4 and mu6
    model.sigma_bv2 = get_sigma_bv2(redshift)
    plt.loglog(model.k, abs(model.P22.mu4(model.k)/norm), label=r"2-loop, $P_{22}[\mu^4]$", c='r')
    plt.loglog(model.k, abs(model.P22.mu6(model.k)/norm), label=r"2-loop, $P_{22}[\mu^6]$", c='b', ls='--')

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P03_and_P13(redshift, model):
    """
    Reproduce Figure 8.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0.01, 0.5)
    ylims = (0.6, 4.4)
    ylabel = r"$P^\mathrm{DM}_{ij} \ / \ k^2 \sigma_v^2 P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # plot P03
    model.sigma_v2 = get_sigma_v2(redshift)
    norm1 = -model.f*(model.f*model.sigma_lin*model.k)**2*model.normed_power_lin_nw(model.k)
    kws = {'color':'g', 'ls':'dashdot', 'label':r"2-loop, $P_{03}[\mu^4]$"}
    plt.semilogx(model.k, model.P03.mu4(model.k)/norm1, **kws)

    # plot P13
    norm2 = -(model.f**2*model.sigma_lin*model.k)**2*model.normed_power_lin_nw(model.k)
    kws = {'label':r"2-loop, $P_{13}[\mu^6]$", 'color':'b', 'ls':'dashed'}
    plt.semilogx(model.k, model.P13.mu6(model.k)/norm2, **kws)

    # also show 1 loop
    model.sigma_v2 = 0.
    with model.preserve(include_2loop=False):
        kws = {'label':r"1-loop, no $\sigma_v$, $P_{03}[\mu^4]$", 'color':'r'}
        plt.semilogx(model.k, model.P03.mu4(model.k)/norm1, **kws)

    ax.legend(loc=0, fontsize=12)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_dm_P13_and_P04(redshift, model):
    """
    Reproduce Figure 9.
    """
    update_model(model, redshift)

    # axis limits
    xlims = (0.01, 0.5)
    ylims = (1e-3, 100.)
    ylabel = r"$P^\mathrm{DM}_{ij} \ / \ k^2 \sigma_v^2 P_\mathrm{NW}$"

    # new axes
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)

    # setup
    model.sigma_v2 = get_sigma_v2(redshift)
    model.sigma_bv2 = get_sigma_bv2(redshift)
    model.sigma_bv4 = get_sigma_bv4(redshift)

    # plot P13 and P04
    norm = (model.f**2*model.sigma_lin*model.k)**2*model.normed_power_lin_nw(model.k)
    plt.loglog(model.k, abs(model.P13.mu4(model.k))/norm, label=r"2-loop, $P_{13}[\mu^4]$", c='b')
    plt.loglog(model.k, abs(model.P04.mu4(model.k))/norm, label=r"2-loop, $P_{04}[\mu^4]$", c='g')

    ax.legend(loc=0, fontsize=12)
    return fig
