from .. import unittest, pytest
from .. import pyplot as plt, cache_manager
import os, numpy

@pytest.fixture(scope='module')
def model():
        
    from pyRSD.rsd import HaloSpectrum
    
    # inititalize the model
    config                       = {}
    config['include_2loop']      = False
    config['transfer_fit']       = 'CLASS'
    config['cosmo_filename']     = 'teppei_sims.ini'
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

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------    
def new_axes(ylabel, xlims, ylims, nticks=5):
    """
    Return a new, formatted axes
    """
    from matplotlib.ticker import AutoMinorLocator
    plt.clf()
    ax = plt.gca()
    
    # axes limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # axes labels
    ax.set_xlabel(r"$k$ [$h$/Mpc]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(nticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(nticks))
    
    return plt.gcf(), ax

def savefig(fig, testname, panel, dpi=200):
    """
    Save the input figure
    """
    # the output dir
    currdir = os.path.split(__file__)[0]
    d = os.path.join(currdir, 'figures', testname)
    if not os.path.exists(d): os.makedirs(d)

    # save
    filename = os.path.join(d, 'panel_%d.jpg' %panel)
    fig.savefig(filename, dpi=dpi)

#------------------------------------------------------------------------------
# TESTS
#------------------------------------------------------------------------------
def test_Kmn_0(model):
    """
    Reproduce top left panel of Figure 1.
    """
    # axis limits
    xlims = (0.01, 1.0)
    ylims = (1., 1e3)
    ylabel = r"$P \ \mathrm{[Mpc/h]^3}$"

    # k array
    k = numpy.logspace(-2, 0, 1000)
    
    # new axes
    fig, ax = new_axes(ylabel, xlims, ylims)
    
    # this panel
    plt.loglog(k, model.K00(k), c='b', label=r'$K_{00}$')
    plt.loglog(k, -model.K00s(k), c='b', ls='--', label=r"$-K_{00,s}$")

    ax.legend(loc=0, fontsize=16)
    savefig(fig, 'test_Kmn', 0)
    
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
    fig, ax = new_axes(ylabel, xlims, ylims)
    
    # this panel
    plt.loglog(k, model.K01(k), c='b', label=r'$K_{01}$')
    plt.loglog(k, model.K01s(k), c='b', ls='dashdot', label=r"$K_{01,s}$")
    plt.loglog(k, model.K02s(k), c='b', ls='dashed', label=r"$K_{02,s}$")

    ax.legend(loc=0, fontsize=16)
    savefig(fig, 'test_Kmn', 1)
    
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
    fig, ax = new_axes(ylabel, xlims, ylims)
    
    # this panel
    plt.loglog(k, model.K11(k), c='b', label=r'$K_{11}$')
    plt.loglog(k, abs(model.K10(k)), c='b', ls='dashdot', label=r"$|K_{10}|$")
    plt.loglog(k, abs(model.K11s(k)), c='b', ls='dashed', label=r"$|K_{11,s}|$")
    plt.loglog(k, -model.K10s(k), c='b', ls='dotted', label=r"$-K_{10,s}$")

    ax.legend(loc=0, fontsize=16)
    savefig(fig, 'test_Kmn', 2)
    
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
    fig, ax = new_axes(ylabel, xlims, ylims)
    
    # this panel
    A = 1./k**2
    plt.loglog(k, -A*model.K20_a(k), c='b', ls='dashed', label=r'$-K_{20}[\mu^2]$')
    plt.loglog(k, A*abs(model.K20s_a(k)), c='b', label=r'$|K_{20,s}[\mu^2]|$')
    plt.loglog(k, A*model.K20_b(k), c='b', ls='dotted', label=r'$K_{20}[\mu^4]$')
    plt.loglog(k, -A*model.K20s_b(k), c='b', ls='dashdot', label=r'$-K_{20,s }[\mu^4]$')

    ax.legend(loc=0, fontsize=16)
    savefig(fig, 'test_Kmn', 3)
    
def teardown_module(module):
    
    thismod = module.__name__.split('.')[-1]
    remote_dir = os.path.join("/project/projectdirs/m779/www/nhand/pyRSDTests", thismod)
    
    cmd = "rsync -e ssh -avzl --progress --delete"
    cmd += " --exclude='.*'"
    
    # add the directories and run the command
    host = 'edison'
    cmd += " figures/* nhand@%s:%s/" %(host, remote_dir)
    ret = os.system(cmd)
    
    print ("teardown_module   module:%s" % module.__name__)
    

            
