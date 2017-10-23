import matplotlib
import numpy as np
import scipy.stats as st

def kde_scipy(trace1, trace2, xlims, ylims, N=100):
    """
    Perform kernel density estimation on two MCMC traces.
    """
    x = np.linspace(xlims[0], xlims[-1], N)
    y = np.linspace(ylims[0], ylims[-1], N)
    X,Y = np.meshgrid(x, y)
    positions = np.vstack([Y.ravel(), X.ravel()])

    values = np.vstack([trace2, trace1])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    return [x, y, Z]

def matplotlib_to_plotly(cmap, pl_entries):
    """
    Convert matplotlib color map to plotly list of RGB tuples.
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
