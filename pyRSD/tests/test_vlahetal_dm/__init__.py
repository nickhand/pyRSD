from .. import unittest, pytest
from .. import pyplot as plt, cache_manager
import os

redshifts = [0., 0.509, 0.989, 2.070]

@pytest.fixture(scope='module', params=redshifts)
def redshift(request):
    return request.param

@pytest.fixture(scope='module')
def model(redshift):
        
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
    config['z']              = redshift
    m = DarkMatterSpectrum(**config)
    
    # load the model
    with cache_manager(m, "vlahetal_dm_%.3f.npy" %redshift) as model:
        
        # set klims
        model.kmin = 0.01
        model.kmax = 0.5
    
    return model

class TestVlahetalDM(object):
    """
    Test `DarkMatterSpectrum` by reproducing the figures in 
    Vlah et al. 2012
    """
    def update_redshift(self, thisz, model):
        """
        Update the redshift-dependent quantities
        """
        model.z        = thisz
        model.f        = model.cosmo.f_z(thisz)
        model.sigma8_z = model.cosmo.Sigma8_z(thisz)
    
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
    
    def savefig(self, fig, testname, z, dpi=1000):
        """
        Save the input figure
        """
        # the output dir
        currdir = os.path.split(__file__)[0]
        d = os.path.join(currdir, 'figures', testname)
        if not os.path.exists(d): os.makedirs(d)
            
        # save
        filename = os.path.join(d, 'z_%.3f.jpg' %z)
        fig.savefig(filename, dpi=dpi)
    
    def test_P00(self, redshift, model):
        """
        Reproduce Figure 1.
        """
        # axis limits
        xlims = (0., 0.3)
        ylims = (0.8, 1.5)
        ylabel = r"$P_{00}^\mathrm{DM} \ / P_\mathrm{NW}$"

        # new axes
        fig, ax = self.new_axes(ylabel, xlims, ylims)
        self.update_redshift(redshift, model)
        
        # normalization
        norm = model.normed_power_lin_nw(model.k)

        # plot the model (1-loop SPT)
        kws = {'c':'b', 'ls':'-', 'label':"1-loop SPT"}
        plt.plot(model.k, model.P00.total.mu0 / norm, **kws)
        
        # linear
        kws = {'c':'k', 'ls':':', 'label':"linear"}
        plt.plot(model.k, model.normed_power_lin(model.k) / norm, **kws)
        
        # zeldovich
        kws = {'c':'Magenta', 'ls':'dashdot', 'label':"Zel'dovich"}
        plt.plot(model.k, model.P00_hzpt_model.__zeldovich__(model.k) / norm, **kws)
        
        ax.legend(loc=0, fontsize=16)
        self.savefig(fig, 'test_P00', redshift)
            
    def test_P01(self, redshift, model):
        """
        Reproduce Figure 2.
        """
        # axis limits
        xlims = (0., 0.3)
        ylims = (0.8, 1.5)
        ylabel = r"$P_{01}^\mathrm{DM} \ / P_\mathrm{NW}$"

        # new axes
        fig, ax = self.new_axes(ylabel, xlims, ylims)
        self.update_redshift(redshift, model)

        # normalization
        A = 2*model.f
        norm = A*model.normed_power_lin_nw(model.k)

        # plot the model (1-loop SPT)
        kws = {'c':'b', 'ls':'-', 'label':"1-loop SPT"}
        plt.plot(model.k, model.P01.total.mu2 / norm, **kws)

        # linear
        kws = {'c':'k', 'ls':':', 'label':"linear"}
        plt.plot(model.k, A*model.normed_power_lin(model.k) / norm, **kws)

        # zeldovich
        kws = {'c':'Magenta', 'ls':'dashdot', 'label':"Zel'dovich"}
        plt.plot(model.k, A*model.P01_hzpt_model.__zeldovich__(model.k) / norm, **kws)

        ax.legend(loc=0, fontsize=16)
        self.savefig(fig, 'test_P01', redshift)

    def test_scalar_P11(self, redshift, model):
        """
        Reproduce Figure 3.
        """
        # axis limits
        xlims = (0., 0.2)
        ylims = (0.8, 2.0)
        ylabel = r"$P_{11}^\mathrm{DM}[\mu^4] \ / P_\mathrm{NW}$"

        # new axes
        fig, ax = self.new_axes(ylabel, xlims, ylims)
        
        # 1-loop results
        with model.preserve():
            
            model.include_2loop = False
            self.update_redshift(redshift, model)

            # normalization
            A = model.f**2
            norm = A * model.normed_power_lin_nw(model.k)

            # plot the model
            kws = {'c':'b', 'ls':'-', 'label':"model"}
            plt.plot(model.k, model.P11.scalar.mu4 / norm, **kws)

            # linear
            kws = {'c':'k', 'ls':':', 'label':"linear"}
            plt.plot(model.k, A*model.normed_power_lin(model.k) / norm, **kws)

            ax.legend(loc=0, fontsize=16)
            self.savefig(fig, 'test_scalar_P11', redshift)
            
    def test_P11(self, redshift, model):
        """
        Reproduce Figure 4.
        """
        # axis limits
        xlims = (0.01, 0.5)
        ylims = (1.0, 1e4)
        ylabel = r"$P_{11}^{ss} \ \mathrm{[Mpc/h]^3}$"

        # new axes
        fig, ax = self.new_axes(ylabel, xlims, ylims)

        # 1-loop results
        with model.preserve():
  
            # update z
            self.update_redshift(redshift, model)

            # 1-loop scalar
            model.include_2loop = False
            plt.loglog(model.k, model.P11.scalar.mu4, label='1-loop, scalar', c='b') 
            
            # 1-loop and 2-loop vector
            model.include_2loop = True
            plt.loglog(model.k, model.P11.vector.mu2, label='2-loop, vector', color='r') 
            model.include_2loop = False
            plt.loglog(model.k, model.P11.vector.mu2, label='1-loop, vector', color='r', ls='--')

            # 1-loop and 2-loop C11
            plt.loglog(model.k, model.f**2 * model.I13(model.k), label=r'1-loop $C_{11}[\mu^4]$', c='g')
            I1 = model.Ivvdd_h02(model.k)
            I2 = model.Idvdv_h04(model.k)
            plt.loglog(model.k, model.f**2 * (I1+I2), label=r'2-loop $C_{11}[\mu^4]$', c='g', ls='--')
            
            # linear
            kws = {'c':'k', 'ls':':', 'label':"linear"}
            plt.loglog(model.k, model.f**2*model.normed_power_lin(model.k), **kws)

            ax.legend(loc=0, fontsize=16, ncol=2)
            self.savefig(fig, 'test_P11', redshift)
            
    # def test_P02(self, redshift, model):
    #     """
    #     Reproduce Figure 5.
    #     """
    #     # axis limits
    #     xlims = (0.1, 0.5)
    #     ylims = (-8, 2)
    #     ylabel = r"$P_{11}^\mathrm{DM}[\mu^4] \ / P_\mathrm{NW}$"
    #
    #     # new axes
    #     fig, ax = self.new_axes(ylabel, xlims, ylims)
    #
    #     # 1-loop results
    #     with model.preserve():
    #
    #         model.include_2loop = False
    #         self.update_redshift(redshift, model)
    #
    #         # normalization
    #         A = model.f**2
    #         norm = A * model.normed_power_lin_nw(model.k)
    #
    #         # plot the model
    #         kws = {'c':'b', 'ls':'-', 'label':"model"}
    #         plt.plot(model.k, model.P11.scalar.mu4 / norm, **kws)
    #
    #         # linear
    #         kws = {'c':'k', 'ls':':', 'label':"linear"}
    #         plt.plot(model.k, A*model.normed_power_lin(model.k) / norm, **kws)
    #
    #         ax.legend(loc=0, fontsize=16)
    #         self.savefig(fig, 'test_P02', redshift)