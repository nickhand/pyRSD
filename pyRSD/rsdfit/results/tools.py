from ... import numpy as np
import scipy.stats
import scipy.signal

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

def plot_mcmc_trace(ax, trace1, trace2, smooth=0, percentiles=[0.683, 0.955, 0.997], 
                    colors=["red","green","blue"], **kwargs):
    """
    Plot 2D traces with contours showing the 1-sigma and 2-sigma levels
    """
    # make the 2D histogram
    n2dbins = 300
    zz, xx, yy = np.histogram2d(trace1, trace2, bins=n2dbins)
    xxbin = xx[1]-xx[0]
    yybin = yy[1]-yy[0]
    xx = xx[1:] + 0.5*xxbin
    yy = yy[1:] + 0.5*yybin
    
    # optionally smooth
    if (smooth > 0):
        kern_size = int(10*smooth)
        sx, sy = scipy.mgrid[-kern_size:kern_size+1, -kern_size:kern_size+1]
        kern = np.exp(-(sx**2 + sy**2)/(2.*smooth**2))
        zz = scipy.signal.convolve2d(zz, kern/np.sum(kern), mode='same')
        
    hist, bins = np.histogram(zz.flatten(), bins=1000)
    sortzz = np.sort(zz.flatten())
    cumhist = np.cumsum(sortzz)*1./np.sum(zz)
    levels = np.array([sortzz[(cumhist>(1-pct)).nonzero()[0][0]] for pct in percentiles])

    ax.contour(xx, yy, zz.T, levels=levels, colors=colors)
    
    return ax
#-------------------------------------------------------------------------------   
