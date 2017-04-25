from __future__ import print_function
from matplotlib import pyplot as plt
import os

def new_axes(ylabel, xlims=None, ylims=None, nticks=5):
    """
    Return a new, formatted axes
    """
    from matplotlib.ticker import AutoMinorLocator
    plt.clf()
    ax = plt.gca()

    # axes limits
    if xlims is not None: ax.set_xlim(xlims)
    if ylims is not None: ax.set_ylim(ylims)

    # axes labels
    ax.set_xlabel(r"$k \ [h \mathrm{Mpc}^{-1}]$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(nticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(nticks))

    return plt.gcf(), ax
