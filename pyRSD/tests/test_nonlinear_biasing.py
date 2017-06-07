"""
This module will compare the nonlinear biasing GP predicition (as measured
from the RunPB sims) to that used in Vlah et al. 2013
"""
from .utils import new_axes
from pyRSD.rsd.simulation import NonlinearBiasFits
from pyRSD import data_dir

import pytest
import numpy
from matplotlib import pyplot as plt
import os

redshifts = [0., 0.509, 0.989]
vlah_bins = [(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 1, 2])]

def get_linear_bias(z_str, mass):
    """
    Return the linear bias for this bin
    """
    d = {}
    d['0.000'] = [1.18, 1.47, 2.04, 3.05]
    d['0.509'] = [1.64, 2.18, 3.13, 4.82]
    d['0.989'] = [2.32, 3.17, 4.64]

    return d[z_str][mass]

def get_nonlinear_biases(z_str, mass):
    """
    Get the nonlinear biases for this bin
    """
    d = {}
    d['0.000'] = [(-0.39, -0.45), (-0.08, -0.35), (0.91, 0.14), (3.88, 2.00)]
    d['0.509'] = [(0.18, -0.20), (1.29, 0.48), (4.48, 2.60), (12.70, 9.50)]
    d['0.989'] = [(1.75, 0.80), (4.77, 3.15), (12.80, 10.80)]

    return dict(zip(['b2_00', 'b2_01'], d[z_str][mass]))

#-------------------------------------------------------------------------------
# TESTS
#-------------------------------------------------------------------------------
@pytest.mark.mpl_image_compare(style='seaborn-ticks', remove_text=True, tolerance=25)
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

    # get a new axes
    plt.clf()
    ax = plt.gca()

    # plot b2_00
    ax.plot(b1s, [gp(b1=b1, select='b2_00_a') for b1 in b1s], c='r', label=r"$b_2^{00}$ (GP prediction)", zorder=0)
    ax.scatter(x, [yy['b2_00'] for yy in y], c='r', alpha=0.8, zorder=10, label=r"$b_2^{00}$ (Vlah et al.)")

    # plot b2_01
    ax.plot(b1s, [gp(b1=b1, select='b2_01_a') for b1 in b1s], c='b', label=r"$b_2^{01}$ (GP prediction)", zorder=0)
    ax.scatter(x, [yy['b2_01'] for yy in y], c='b', alpha=0.8, zorder=10, label=r"$b_2^{01}$ (Vlah et al.)")

    # new axes
    ax.set_xlabel(r"$b_1$", fontsize=16)
    ax.set_ylabel(r"$b_2$", fontsize=16)
    ax.legend(ncol=2, loc=0)

    return plt.gcf()
