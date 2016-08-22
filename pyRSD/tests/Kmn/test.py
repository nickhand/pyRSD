"""
This module will reproduce Figure 1 of Vlah et al. 2013., which
shows the various Kmn integrals
"""
from . import model
from ..utils import new_axes, savefig, teardown_module

from matplotlib import pyplot as plt
import numpy

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
    fig, ax = new_axes(ylabel, xlims=xlims, ylims=ylims)
    
    # this panel
    plt.loglog(k, model.K00(k), c='b', label=r'$K_{00}$')
    plt.loglog(k, -model.K00s(k), c='b', ls='--', label=r"$-K_{00,s}$")

    ax.legend(loc=0, fontsize=16)
    savefig(fig, __file__, '', "panel_0.jpg")
    
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
    
    # this panel
    plt.loglog(k, model.K01(k), c='b', label=r'$K_{01}$')
    plt.loglog(k, model.K01s(k), c='b', ls='dashdot', label=r"$K_{01,s}$")
    plt.loglog(k, model.K02s(k), c='b', ls='dashed', label=r"$K_{02,s}$")

    ax.legend(loc=0, fontsize=16)
    savefig(fig, __file__, '', "panel_1.jpg")
    
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
    
    # this panel
    plt.loglog(k, model.K11(k), c='b', label=r'$K_{11}$')
    plt.loglog(k, abs(model.K10(k)), c='b', ls='dashdot', label=r"$|K_{10}|$")
    plt.loglog(k, abs(model.K11s(k)), c='b', ls='dashed', label=r"$|K_{11,s}|$")
    plt.loglog(k, -model.K10s(k), c='b', ls='dotted', label=r"$-K_{10,s}$")

    ax.legend(loc=0, fontsize=16)
    savefig(fig, __file__, '', 'panel_2.jpg')
    
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
    
    # this panel
    A = 1./k**2
    plt.loglog(k, -A*model.K20_a(k), c='b', ls='dashed', label=r'$-K_{20}[\mu^2]$')
    plt.loglog(k, A*abs(model.K20s_a(k)), c='b', label=r'$|K_{20,s}[\mu^2]|$')
    plt.loglog(k, A*model.K20_b(k), c='b', ls='dotted', label=r'$K_{20}[\mu^4]$')
    plt.loglog(k, -A*model.K20s_b(k), c='b', ls='dashdot', label=r'$-K_{20,s }[\mu^4]$')

    ax.legend(loc=0, fontsize=16)
    savefig(fig, __file__, '', 'panel_3.jpg')