"""
This module will reproduce the plots in Vlah et al. 2013., as
a test of :class:`HaloSpectrum`
"""
from .utils import new_axes
from . import cache_manager
from pyRSD import data_dir

from matplotlib import pyplot as plt
import os
import numpy
import pytest

sim_tags  = ['teppei_lowz', 'teppei_midz', 'teppei_highz']
redshifts = [0., 0.509, 0.989]
mass_bins = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2]]
binning_vals = [(z,mass_bin) for i, z in enumerate(redshifts) for mass_bin in mass_bins[i]]

#-------------------------------------------------------------------------------
# FIXTURES
#-------------------------------------------------------------------------------
@pytest.fixture(scope='module', params=binning_vals)
def binning(request):
    return request.param

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
    config['vel_disp_from_sims'] = False
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


#-------------------------------------------------------------------------------
# TOOLS
#-------------------------------------------------------------------------------
def get_sigma_v(z_str, mass):
    """
    Return the sigma_v for this bin
    """
    d = {}
    d['0.000'] = [306., 302., 296., 288.]
    d['0.509'] = [357., 352., 346., 339.]
    d['0.989'] = [340., 337., 330.]

    return d[z_str][mass]

def get_linear_bias(z_str, mass):
    """
    Return the linear bias for this bin
    """
    d = {}
    d['0.000'] = [1.18, 1.47, 2.04, 3.05]
    d['0.509'] = [1.64, 2.18, 3.13, 4.82]
    d['0.989'] = [2.32, 3.17, 4.64]

    return d[z_str][mass]

def get_stochasticity(z_str, mass):
    """
    Return the stochasticity function for this bin
    """
    d = {}
    d['0.000'] = [(83.2376, -246.78), (-645.891, -365.149), (-3890.46, -211.452), (-19202.9, 111.395)]
    d['0.509'] = [(-168.216, -137.268), (-1335.54, -104.929), (-6813.82, -241.446), (-35181.3, -5473.79)]
    d['0.989'] = [(-427.779, -41.3676), (-2661.11, -229.627), (-16915.5, -3215.98)]

    params = dict(zip(['A0', 'A1'], d[z_str][mass]))
    return lambda k: params['A0'] + params['A1'] * numpy.log(k)

def get_nonlinear_biases(z_str, mass):
    """
    Get the nonlinear biases for this bin
    """
    d = {}
    d['0.000'] = [(-0.39, -0.45), (-0.08, -0.35), (0.91, 0.14), (3.88, 2.00)]
    d['0.509'] = [(0.18, -0.20), (1.29, 0.48), (4.48, 2.60), (12.70, 9.50)]
    d['0.989'] = [(1.75, 0.80), (4.77, 3.15), (12.80, 10.80)]

    return dict(zip(['b2_00', 'b2_01'], d[z_str][mass]))

def get_params(z, mass):
    """
    Return the model parameters as measured by
    """
    z_str = "%.3f" %z

    toret = get_nonlinear_biases(z_str, mass)
    toret['stochasticity'] = get_stochasticity(z_str, mass)
    toret['sigma_v'] = get_sigma_v(z_str, mass)
    toret['b1'] = get_linear_bias(z_str, mass)

    return toret

def update_model(model, z, b1):
    """
    Update the redshift-dependent quantities
    """
    # update redshift-dependent quantities
    model.z        = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f        = model.cosmo.f_z(z)

    # set the bias
    model.b1 = b1

#-------------------------------------------------------------------------------
# TESTS
#-------------------------------------------------------------------------------
@pytest.mark.parametrize("bins", [(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 1, 2])])
@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_halo_Phm(model, bins):
    """
    Reproduce Figure 3
    """
    iz, mass_bins = bins
    z = redshifts[iz]

    # new axes
    xlims = (0., 0.3)
    ylims = [(0.85, 1.3), (0.95, 1.5), (0.95, 1.5)]
    ylabel = r"$P^\mathrm{hm} \ / b_1 P_{00}$"

    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims[iz])

    # plot each mass bin
    for mass_bin in mass_bins:

        # get the params
        params = get_params(z, mass_bin)
        b1 = params.pop('b1')

        # update the model
        update_model(model, z, b1)

        # update the params
        model.sigma_v = params.pop('sigma_v') / 100. / model.f

        # plot Phm
        with model.load_dm_sims(sim_tags[iz]):
            norm = model.b1 * model.P00.mu0(model.k)
            with model.cache_override(**params):
                plt.plot(model.k, model.Phm(model.k) / norm, label=r"$b_1 = %.2f$" %b1)

    ax.legend(loc=0, fontsize=14)
    ax.axhline(y=1, c='k', ls='--')
    return fig

@pytest.mark.parametrize("bins", [(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 1, 2])])
@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_halo_P00(model, bins):
    """
    Reproduce Figure 4
    """
    iz, mass_bins = bins
    z = redshifts[iz]

    # new axes
    xlims = (0., 0.3)
    ylims = [(0.5, 1.1), (0.7, 1.1), (0.6, 1.2)]
    ylabel = r"$P_{00}^\mathrm{hh} \ / b_1^2 P_{00}$"

    fig, ax = new_axes(ylabel, xlims, ylims[iz])

    # plot each mass bin
    for mass_bin in mass_bins:

        # get the params
        params = get_params(z, mass_bin)
        b1 = params.pop('b1')

        # update the model
        update_model(model, z, b1)

        # update the params
        model.sigma_v = params.pop('sigma_v') / 100. / model.f

        # plot Phh
        with model.load_dm_sims(sim_tags[iz]):
            norm = model.b1**2 * model.P00.mu0(model.k)
            with model.cache_override(**params):
                P00_ss = model.b1**2 * model.P00.mu0(model.k)
                P00_ss += 2*(model.b1*model.b2_00_a(model.b1))*model.K00(model.k)
                plt.plot(model.k, (P00_ss + params['stochasticity'](model.k))/ norm, label=r"$b_1 = %.2f$" %b1)

    ax.legend(loc=0, fontsize=14)
    ax.axhline(y=1, c='k', ls='--')
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_halo_P01(model, binning):
    """
    Reproduce Figure 5
    """
    z, mass_bin = binning
    iz = redshifts.index(z)

    if iz == 0:
        ylims = [(0.8, 1.3), (0.8, 1.5), (0.8, 2.), (0.8, 3)]
    elif iz == 1:
        ylims = [(0.8, 1.5), (0.8, 2.), (0.8, 2.4), (0.8, 3)]
    else:
        ylims = [(0.8, 1.6), (0.8, 2.2), (0.8, 3)]

    # new axes
    xlims = (0.0, 0.25)
    ylabel = r"$P_{01}^\mathrm{hh} \ / 2 f b_1 P_\mathrm{NW}$"

    fig, ax = new_axes(ylabel, xlims, ylims[mass_bin])

    # get the params
    params = get_params(z, mass_bin)
    b1 = params.pop('b1')

    # update the model
    update_model(model, z, b1)

    # update the params
    params['stochasticity'] = params['stochasticity'](model.k)
    model.sigma_v = params.pop('sigma_v') / 100. / model.f

    # plot model P01
    with model.load_dm_sims(sim_tags[iz]):
        norm = 2 * model.f * model.b1 * model.normed_power_lin_nw(model.k)
        with model.cache_override(**params):
            plt.plot(model.k, model.P01_ss.mu2(model.k) / norm, label="model")

        # b1 * DM
        plt.plot(model.k, model.b1 * model.P01.mu2(model.k) / norm, c='k', label=r"$b_1$ DM", ls='dashed')

    # PT
    with model.use_spt():
        with model.cache_override(**params):
            plt.plot(model.k, model.P01_ss.mu2(model.k) / norm, label=r"PT: $b_1$+$b_2$", c='b')

    # linear
    kws = {'color':'k', 'ls':'dotted', 'label':"linear"}
    plt.plot(model.k, 2*model.f*model.b1 *  model.normed_power_lin(model.k) / norm, **kws)

    ax.legend(loc=0, fontsize=14)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_halo_P11_plus_P02(model, binning):
    """
    Reproduce Figure 9
    """
    z, mass_bin = binning
    iz = redshifts.index(z)
    if iz == 0:
        ylims = [(-2, 1.), (-2, 1.), (-3, 2.), (-6., 4)]
    elif iz == 1:
        ylims = [(-3., 1.), (-4., 2.), (-6., 4.), (-15., 5)]
    else:
        ylims = [(-5., 2.), (-6., 4), (-15., 5.)]

    # new axes
    xlims = (0., 0.3)
    ylabel = r"$P^\mathrm{hh}[\mu^2] \ / k^2 \sigma_v^2 P_\mathrm{NW}$"

    fig, ax = new_axes(ylabel, xlims, ylims[mass_bin])

    # get the params
    params = get_params(z, mass_bin)
    b1 = params.pop('b1')

    # update the model
    update_model(model, z, b1)

    # update the params
    params['stochasticity'] = params['stochasticity'](model.k)
    model.sigma_v = params.pop('sigma_v') / 100. / model.f

    # plot model
    with model.load_dm_sims(sim_tags[iz]):
        norm = (model.f*model.sigma_lin*model.k)**2 * model.b1 * model.normed_power_lin_nw(model.k)
        with model.cache_override(**params):

            # plot P11 + P02 [mu2]
            kws = {'label':r'$(P_{11}^{hh} + P_{02}^{hh})[\mu^2]$', 'c':'r'}
            plt.plot(model.k, (model.P11_ss.mu2(model.k) + model.P02_ss.mu2(model.k))/norm, **kws)

            # plot P11[mu4]
            kws = {'label':r'$P_{02}^{hh}[\mu^4]$', 'c':'b'}
            plt.plot(model.k, model.P02_ss.mu4(model.k)/norm, **kws)

    ax.legend(loc=0, fontsize=14)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_halo_P11_mu4(model, binning):
    """
    Reproduce Figure 10
    """
    z, mass_bin = binning
    iz = redshifts.index(z)
    if iz == 0:
        ylims = [(0.8, 1.3), (0.8, 1.5), (0.8, 2.4), (0.8, 6)]
    elif iz == 1:
        ylims = [(0.8, 1.8), (0.8, 2.4), (0.8, 4.), (0.8, 8.)]
    else:
        ylims = [(0.8, 2.5), (0.8, 3.5), (0.8, 7)]

    # new axes
    xlims = (0., 0.2)
    ylabel = r"$P_{11}^\mathrm{hh}[\mu^4] \ / P_{11}^\mathrm{DM}$"

    fig, ax = new_axes(ylabel, xlims, ylims[mass_bin])

    # get the params
    params = get_params(z, mass_bin)
    b1 = params.pop('b1')

    # update the model
    update_model(model, z, b1)

    # update the params
    params['stochasticity'] = params['stochasticity'](model.k)
    model.sigma_v = params.pop('sigma_v') / 100. / model.f

    # note that this won't agree exactly...we are using Jennings
    # Pvv here

    # model
    with model.load_dm_sims(sim_tags[iz]):
        norm = model.P11.mu4(model.k)
        with model.cache_override(**params):
            plt.plot(model.k, model.P11_ss.mu4(model.k) / norm, label="model", c='r')

    # PT
    with model.use_spt():
        with model.cache_override(**params):
            plt.plot(model.k, model.P11_ss.mu4(model.k) / norm, label=r"PT: $b_1$+$b_2$", c='b')

    ax.legend(loc=0, fontsize=14)
    return fig

@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
def test_halo_higher_order_mu4(model, binning):
    """
    Reproduce Figure 11
    """
    z, mass_bin = binning
    iz = redshifts.index(z)
    if iz == 0:
        ylims = [(-0.5, 4.), (-0.5, 4.), (-0.5, 6.), (-0.5, 8.)]
    elif iz == 1:
        ylims = [(-0.5, 4), (-0.5, 6.), (-0.5, 8.), (-0.5, 8.)]
    else:
        ylims = [(-0.5, 6.), (-0.5, 6.), (-0.5, 8.)]

    # new axes
    xlims = (0., 0.3)
    ylabel = r"$P^\mathrm{hh}[\mu^4] \ / k^2 \sigma_v^2 P_\mathrm{NW}$"

    fig, ax = new_axes(ylabel, xlims, ylims[mass_bin])

    # get the params
    params = get_params(z, mass_bin)
    b1 = params.pop('b1')

    # update the model
    update_model(model, z, b1)

    # update the params
    params['stochasticity'] = params['stochasticity'](model.k)
    model.sigma_v = params.pop('sigma_v') / 100. / model.f

    # plot model
    with model.load_dm_sims(sim_tags[iz]):
        norm = model.f*(model.f*model.sigma_lin*model.k)**2 * model.b1 * model.normed_power_lin_nw(model.k)
        with model.cache_override(**params):

            # plot - (P12 + P03)
            kws = {'label':r'$-(P_{12}^{hh}+P_{03}^{hh})$', 'c':'r'}
            plt.plot(model.k, -(model.P12_ss.mu4(model.k) + model.P03_ss.mu4(model.k))/norm, **kws)

            # plot P13 + P22 + P04
            kws = {'label':r'$P_{13}^{hh}+P_{22}^{hh}+P_{04}^{hh}$', 'c':'b'}
            y = (model.P13_ss.mu4(model.k)  + model.P22_ss.mu4(model.k) + model.P04_ss.mu4(model.k))
            plt.plot(model.k, y/norm, **kws)

    ax.legend(loc=0, fontsize=14)
    return fig
