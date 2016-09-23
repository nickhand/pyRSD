"""
This module will compare the nonlinear biasing GP predicition (as measured
from the RunPB sims) to that used in Vlah et al. 2013
"""
from ..vlahetal_halo import get_nonlinear_biases, get_linear_bias
from ..utils import savefig, teardown_module

from pyRSD.rsd.simulation import NonlinearBiasFits
import numpy
from matplotlib import pyplot as plt

redshifts = [0., 0.509, 0.989]
vlah_bins = [(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 1, 2])]


def test_nonlinear_biasing():
    """
    Compare the nonlinear biasing interpolation to that of Vlah et al. 2013
    """
    # initialize the GP
    gp = NonlinearBiasFits()
    b1s = numpy.linspace(0.9, 6, 500)
    
    # get the Vlah et al data
    x = []; y = []
    for i, z in enumerate(redshifts):
        z_str = "%.3f" %z
        for mass_bin in vlah_bins[i][1]:
            x.append(get_linear_bias(z_str, mass_bin))
            y.append(get_nonlinear_biases(z_str, mass_bin))
    
    
    # plot b2_00
    plt.plot(b1s, [gp(b1=b1, select='b2_00_a') for b1 in b1s], c='r', label=r"$b_2^{00}$ (GP prediction)", zorder=0)
    plt.scatter(x, [yy['b2_00'] for yy in y], c='r', alpha=0.8, zorder=10, label=r"$b_2^{00}$ (Vlah et al.)")
    
    # plot b2_01
    plt.plot(b1s, [gp(b1=b1, select='b2_01_a') for b1 in b1s], c='b', label=r"$b_2^{01}$ (GP prediction)", zorder=0)
    plt.scatter(x, [yy['b2_01'] for yy in y], c='b', alpha=0.8, zorder=10, label=r"$b_2^{01}$ (Vlah et al.)")
    
    # new axes
    ax = plt.gca()
    ax.set_xlabel(r"$b_1$", fontsize=16)
    ax.set_ylabel(r"$b_2$", fontsize=16)
    ax.legend(ncol=2, loc=0)
    
    # add bias and save
    savefig(plt.gcf(), __file__, "" , "comparison.jpg")
