"""
This module will compare the RunPB halo correlators to the
terms in :class:`HaloSpectrum`
"""
from . import model, a, mass, data, biases
from . import update_model, get_valid_data
from ..utils import new_axes, savefig, teardown_module

from matplotlib import pyplot as plt

def finalize_axes(ax, a, mass, b1):
    """
    Finalize the axes
    """    
    # set xlims and title
    ax.set_title(r"a = %s, mass = %d, $b_1 = %.2f$" %(a, mass, b1), fontsize=16)
    ax.set_xlim(0.01, 0.5)
        
    # save
    filename = 'mass_%d.jpg' %(mass)
    return filename
    
#------------------------------------------------------------------------------
# mu^0
#------------------------------------------------------------------------------
def test_P00(a, mass, model):
    """
    Compare halo P00 model and sims
    """
    # new axes
    ylabel = r"$P_{00}^\mathrm{hh} \ / b_1^2 P_\mathrm{NW}$"
    fig, ax = new_axes(ylabel)
    
    # update the model
    update_model(model, a, mass)
    
    # data for this test
    P00 = data.get_P00().sel(mu=0)
    
    # plot the model
    norm = model.b1**2 * model.normed_power_lin_nw(model.k)
    plt.plot(model.k, model.P00_ss.total.mu0 / norm, c='Crimson')
    
    # plot the data
    x, y, yerr = get_valid_data(model, P00, a, mass, subtract_shot_noise=True)
    norm = model.b1**2 * model.normed_power_lin_nw(x)
    plt.errorbar(x, y/norm, yerr/norm, ls='', c='k', alpha=0.5)
    
    # add bias and save
    savefig(fig, __file__, 'test_P00/'+a , finalize_axes(ax, a, mass, model.b1))
    
#------------------------------------------------------------------------------
# mu^2
#------------------------------------------------------------------------------
def test_P01(a, mass, model):
    """
    Compare halo P11 model and sims
    """
    # new axes
    ylabel = r"$P_{01}^\mathrm{hh} \ / 2 f b_1 P_\mathrm{NW}$"
    fig, ax = new_axes(ylabel)
    
    # update the model
    update_model(model, a, mass)
    
    # data for this test
    P01 = data.get_P01().sel(mu=2)
    
    # plot the model
    norm = 2*model.f*model.b1 * model.normed_power_lin_nw(model.k)
    plt.plot(model.k, model.P01_ss.total.mu2 / norm, c='Crimson')
    
    # plot the data
    x, y, yerr = get_valid_data(model, P01, a, mass)
    norm = 2*model.f*model.b1 * model.normed_power_lin_nw(x)
    plt.errorbar(x, y/norm, yerr/norm, ls='', c='k', alpha=0.5)
    
    # add bias and save
    savefig(fig, __file__, 'test_P01/'+a, finalize_axes(ax, a, mass, model.b1))
    
def test_P11_plus_P02(a, mass, model):
    """
    Compare halo P11+P02 model and sims
    """
    # new axes
    ylabel = r"$(P_{11}^\mathrm{hh} + P_{02}^\mathrm{hh})[\mu^2] \ / b_1 k^2 \sigma_v^2 P_\mathrm{NW}$"
    fig, ax = new_axes(ylabel)
    
    # update the model
    update_model(model, a, mass)
    
    # data for this test
    P11_plus_P02 = data.get_P11_plus_P02().sel(mu=2)
    
    # plot the model
    with model.preserve(use_vlah_biasing=True):
        norm = model.b1 * (model.D*model.sigma_lin*model.k)**2 * model.normed_power_lin_nw(model.k)
        plt.plot(model.k, (model.P11_ss.total.mu2 + model.P02_ss.total.mu2) / norm, c='Crimson')
    
    # plot the data
    x, y, yerr = get_valid_data(model, P11_plus_P02, a, mass)
    norm = model.b1 * (model.D*model.sigma_lin*x)**2 * model.normed_power_lin_nw(x)
    plt.errorbar(x, y/norm, yerr/norm, ls='', c='k', alpha=0.5)
    
    # add bias and save
    savefig(fig, __file__, 'test_P11_plus_P02/'+a, finalize_axes(ax, a, mass, model.b1))
    

#------------------------------------------------------------------------------
# mu^4
#------------------------------------------------------------------------------