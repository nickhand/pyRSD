from .. import unittest, pyplot as plt, cache_manager
import os

class TestVlahetalDM(unittest.TestCase):
    """
    Test `DarkMatterSpectrum` by reproducing the figures in 
    Vlah et al. 2012
    """
    def setUp(self):
        
        from pyRSD.rsd import DarkMatterSpectrum
        
        # inititalize the model
        config                   = {}
        config['include_2loop']  = True
        config['transfer_fit']   = 'CLASS'
        config['cosmo_filename'] = 'teppei_sims.ini'
        config['max_mu']         = 6
        config['use_P00_model']  = False
        config['use_P01_model']  = False
        config['use_P11_model']  = False
        config['use_Pdv_model']  = False
        config['interpolate']    = False
        model = DarkMatterSpectrum(**config)
        
        # load the model
        with cache_manager(model, "vlahetal_dm.npy") as m:
            self.model = m
            
        # set kmax
        self.model.kmax = 0.4
        
        # hi-res interpolation
        self.model.Nk = 500
            
        # redshift info
        self.redshifts = [0., 0.509, 0.989, 2.070]
        
    def update_redshift(self, z):
        """
        Update the redshift-dependent quantities
        """
        self.model.z        = z
        self.model.f        = self.model.cosmo.f_z(z)
        self.model.sigma8_z = self.model.cosmo.Sigma8_z(z)
    
    def new_axes(self, ylabel, xlims, ylims):
        """
        Return a new, formatted axes
        """
        from matplotlib.ticker import AutoMinorLocator
        fig = plt.figure()
        ax = fig.gca()
        
        # axes limits
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # axes labels
        ax.set_xlabel(r"$k$ (h/Mpc)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        
        # add minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        
        return fig, ax
    
    def savefig(self, fig, testname, z):
        """
        Save the input figure
        """
        # the output dir
        currdir = os.path.split(__file__)[0]
        d = os.path.join(currdir, 'figures', testname)
        if not os.path.exists(d): os.makedirs(d)
            
        # save
        filename = os.path.join(d, 'z_%.3f.jpg' %z)
        fig.savefig(filename, dpi=1000)
    
    def test_P00(self):
        """
        Reproduce Figure 1.
        """
        # axis limits
        xlims = (0., 0.3)
        ylims = (0.8, 1.5)
        ylabel = r"$P_{00}^\mathrm{DM} \ / P_\mathrm{NW}$"
                
        # make a figure for each redshift
        for z in self.redshifts:
        
            fig, ax = self.new_axes(ylabel, xlims, ylims)
            self.update_redshift(z)
            
            # normalization
            norm = self.model.normed_power_lin_nw(self.model.k)

            # plot the model (1-loop SPT)
            kws = {'c':'b', 'ls':'-', 'label':"1-loop SPT"}
            plt.plot(self.model.k, self.model.P00.total.mu0 / norm, **kws)
            
            # linear
            kws = {'c':'k', 'ls':':', 'label':"linear"}
            plt.plot(self.model.k, self.model.normed_power_lin(self.model.k) / norm, **kws)
            
            # zeldovich
            kws = {'c':'Magenta', 'ls':'dashdot', 'label':"Zel'dovich"}
            plt.plot(self.model.k, self.model.P00_hzpt_model.__zeldovich__(self.model.k) / norm, **kws)
            
            ax.legend(loc=0, fontsize=16)
            self.savefig(fig, 'test_P00', z)