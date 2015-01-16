from ... import numpy as np

#-------------------------------------------------------------------------------
def compute_sigma_level(trace1, trace2, nbins=20):
    """
    From a set of traces, bin by number of standard deviations
    """
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1e-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)
#-------------------------------------------------------------------------------

def plot_mcmc_trace(ax, trace1, trace2, scatter=True, **kwargs):
    """
    Plot 2D traces with contours showing the 1-sigma and 2-sigma levels
    """
    xbins, ybins, sigma = compute_sigma_level(trace1, trace2)
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace1, trace2, ',k', alpha=0.8)
    
    return ax
#-------------------------------------------------------------------------------   
