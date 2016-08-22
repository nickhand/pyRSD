"""
This module will reproduce the plots in Vlah et al. 2013., as
a test of :class:`HaloSpectrum`
"""
from . import redshifts, sim_tags
from . import model, binning
from . import update_model, get_params
from ..utils import new_axes, savefig, teardown_module

from matplotlib import pyplot as plt
import os

def output_paths(d, z, mass=None):
    """
    Utility function to return the directory name and filename
    """
    if mass is not None: 
        d = os.path.join(d, "%.3f" %z)
        
    if mass is None:
        filename = 'z_%.3f.jpg' %z
    else:
        filename = 'mass_%d.jpg' %mass
        
    return d, filename
    

def test_Phm(model, binning):
    """
    Reproduce Figure 3
    """
    iz, mass_bins = binning
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
        params['stochasticity'] = params['stochasticity'](model.k)
        model.sigma_v = params.pop('sigma_v') / 100. / model.f
        
        # plot Phm
        with model.load_dm_sims(sim_tags[iz]):
            norm = model.b1 * model.P00.total.mu0
            with model.cache_override(**params):
                plt.plot(model.k, model.Phm.total.mu0 / norm, label=r"$b_1 = %.2f$" %b1)

    ax.legend(loc=0, fontsize=14)
    ax.axhline(y=1, c='k', ls='--')
    savefig(fig, __file__, *output_paths('test_Phm', z=z))

def test_P00(model, binning):
    """
    Reproduce Figure 4
    """
    iz, mass_bins = binning
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
        params['stochasticity'] = params['stochasticity'](model.k)
        model.sigma_v = params.pop('sigma_v') / 100. / model.f
        
        # plot Phh
        with model.load_dm_sims(sim_tags[iz]):
            norm = model.b1**2 * model.P00.total.mu0
            with model.cache_override(**params):
                P00_ss = model.b1**2 * model.P00.total.mu0 + 2*(model.b1*model.b2_00_a(model.b1))*model.K00(model.k)
                plt.plot(model.k, (P00_ss + model.stochasticity)/ norm, label=r"$b_1 = %.2f$" %b1)

    ax.legend(loc=0, fontsize=14)
    ax.axhline(y=1, c='k', ls='--')
    savefig(fig, __file__, *output_paths('test_P00', z=z))
    
def test_P01(model, binning):
    """
    Reproduce Figure 5
    """
    iz, mass_bins = binning
    z = redshifts[iz]
    if iz == 0:  
        ylims = [(0.8, 1.3), (0.8, 1.5), (0.8, 2.), (0.8, 3)]
    elif iz == 1:
        ylims = [(0.8, 1.5), (0.8, 2.), (0.8, 2.4), (0.8, 3)]
    else:
        ylims = [(0.8, 1.6), (0.8, 2.2), (0.8, 3)]
    
    # new axes
    xlims = (0.0, 0.25)
    ylabel = r"$P_{01}^\mathrm{hh} \ / 2 f b_1 P_\mathrm{NW}$"
    
    # plot each mass bin
    for mass_bin in mass_bins:
          
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
                plt.plot(model.k, model.P01_ss.total.mu2 / norm, label="model")
                
            # b1 * DM
            plt.plot(model.k, model.b1 * model.P01.total.mu2 / norm, c='k', label=r"$b_1$ DM", ls='dashed')
                
        # PT
        with model.use_spt():
            with model.cache_override(**params):
                plt.plot(model.k, model.P01_ss.total.mu2 / norm, label=r"PT: $b_1$+$b_2$", c='b')
                
        # linear
        kws = {'color':'k', 'ls':'dotted', 'label':"linear"}
        plt.plot(model.k, 2*model.f*model.b1 *  model.normed_power_lin(model.k) / norm, **kws)
                
        ax.legend(loc=0, fontsize=14)
        savefig(fig, __file__, *output_paths('test_P01', z=z, mass=mass_bin))

def test_P11_plus_P02(model, binning):
    """
    Reproduce Figure 9
    """
    iz, mass_bins = binning
    z = redshifts[iz]
    if iz == 0:  
        ylims = [(-2, 1.), (-2, 1.), (-3, 2.), (-6., 4)]
    elif iz == 1:
        ylims = [(-3., 1.), (-4., 2.), (-6., 4.), (-15., 5)]
    else:
        ylims = [(-5., 2.), (-6., 4), (-15., 5.)]
    
    # new axes
    xlims = (0., 0.3)
    ylabel = r"$P^\mathrm{hh}[\mu^2] \ / k^2 \sigma_v^2 P_\mathrm{NW}$"
    fig, ax = new_axes(ylabel, xlims, ylims[iz])
    
    # plot each mass bin
    for mass_bin in mass_bins:
        
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
                plt.plot(model.k, (model.P11_ss.total.mu2 + model.P02_ss.total.mu2)/norm, **kws)
                
                # plot P11[mu4]
                kws = {'label':r'$P_{02}^{hh}[\mu^4]$', 'c':'b'}
                plt.plot(model.k, model.P02_ss.total.mu4/norm, **kws)
                
        ax.legend(loc=0, fontsize=14)
        savefig(fig, __file__, *output_paths('test_P11_plus_P02', z=z, mass=mass_bin))

def test_P11_mu4(model, binning):
    """
    Reproduce Figure 10
    """
    iz, mass_bins = binning
    z = redshifts[iz]
    if iz == 0:  
        ylims = [(0.8, 1.3), (0.8, 1.5), (0.8, 2.4), (0.8, 6)]
    elif iz == 1:
        ylims = [(0.8, 1.8), (0.8, 2.4), (0.8, 4.), (0.8, 8.)]
    else:
        ylims = [(0.8, 2.5), (0.8, 3.5), (0.8, 7)]
    
    # new axes
    xlims = (0., 0.2)
    ylabel = r"$P_{11}^\mathrm{hh}[\mu^4] \ / P_{11}^\mathrm{DM}$"
    fig, ax = new_axes(ylabel, xlims, ylims[iz])
    
    # plot each mass bin
    for mass_bin in mass_bins:
        
        fig, ax = new_axes(ylabel, xlims, ylims[mass_bin])    
            
        # get the params
        params = get_params(z, mass_bin)
        b1 = params.pop('b1')
        
        # update the model
        update_model(model, z, b1)
        
        # update the params
        params['stochasticity'] = params['stochasticity'](model.k)
        model.sigma_v = params.pop('sigma_v') / 100. / model.f
                
        # preserve model
        with model.preserve():
            
            # don't use Jennings Pvv for now
            model.use_Pvv_model = False
            
            # model
            with model.load_dm_sims(sim_tags[iz]):
                norm = model.P11.total.mu4
                with model.cache_override(**params):
                    plt.plot(model.k, model.P11_ss.total.mu4 / norm, label="model", c='r')
        
            # PT
            with model.use_spt():
                with model.cache_override(**params):
                    plt.plot(model.k, model.P11_ss.total.mu4 / norm, label=r"PT: $b_1$+$b_2$", c='b')
                
        ax.legend(loc=0, fontsize=14)
        savefig(fig, __file__, *output_paths('test_P11_mu4', z=z, mass=mass_bin))
        
def test_higher_order_mu4(model, binning):
    """
    Reproduce Figure 11
    """
    iz, mass_bins = binning
    z = redshifts[iz]
    if iz == 0:  
        ylims = [(-0.5, 4.), (-0.5, 4.), (-0.5, 6.), (-0.5, 8.)]
    elif iz == 1:
        ylims = [(-0.5, 4), (-0.5, 6.), (-0.5, 8.), (-0.5, 8.)]
    else:
        ylims = [(-0.5, 6.), (-0.5, 6.), (-0.5, 8.)]
    
    # new axes
    xlims = (0., 0.3)
    ylabel = r"$P^\mathrm{hh}[\mu^4] \ / k^2 \sigma_v^2 P_\mathrm{NW}$"
    fig, ax = new_axes(ylabel, xlims, ylims[iz])
    
    # plot each mass bin
    for mass_bin in mass_bins:
        
        fig, ax = new_axes(ylabel, xlims, ylims[mass_bin])    
            
        # get the params
        params = get_params(z, mass_bin)
        b1 = params.pop('b1')
        
        # update the model
        update_model(model, z, b1)
        
        # update the params
        params['stochasticity'] = params['stochasticity'](model.k)
        model.sigma_v = params.pop('sigma_v') / 100. / model.f
        model.use_Pvv_model = True
        
        # plot model 
        with model.load_dm_sims(sim_tags[iz]):
            norm = model.f*(model.f*model.sigma_lin*model.k)**2 * model.b1 * model.normed_power_lin_nw(model.k)
            with model.cache_override(**params):
                
                # plot - (P12 + P03)
                kws = {'label':r'$-(P_{12}^{hh}+P_{03}^{hh})$', 'c':'r'}
                plt.plot(model.k, -(model.P12_ss.total.mu4 + model.P03_ss.total.mu4)/norm, **kws)
                
                # plot P13 + P22 + P04
                kws = {'label':r'$P_{13}^{hh}+P_{22}^{hh}+P_{04}^{hh}$', 'c':'b'}
                plt.plot(model.k, (model.P13_ss.total.mu4  + model.P22_ss.total.mu4 + model.P04_ss.total.mu4)/norm, **kws)
                
        ax.legend(loc=0, fontsize=14)
        savefig(fig, __file__, *output_paths('test_higher_order_mu4', z=z, mass=mass_bin))