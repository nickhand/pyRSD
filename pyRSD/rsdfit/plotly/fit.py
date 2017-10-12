from . import py, go
from pyRSD.rsdfit.util import plot
import seaborn as sns
import numpy as np

def plot_fit_comparison(driver, colors=None, filename=None, labels=None):
    """
    Plot a data vs theory comparison using :mod:`plotly`.
    """
    # use a Paired color
    if colors is None:
        colors = sns.color_palette("Paired", 14).as_hex()
    else:
        assert isinstance(colors, list) and len(colors) >= 2*driver.data.size

    if labels is not None:
        assert len(labels) == driver.data.size

    # offset
    offset = -0.1 if driver.mode == 'pkmu' else 0.

    data = []
    labels_ = []

    # plot the data
    c = colors[1::2][:driver.data.size]
    for res in plot.plot_normalized_data(driver, offset=offset, use_labels=True, labels=labels, color=c):
        x, y, yerr, meta = res
        labels_.append(meta['label']) # save the label for later

        # format the color
        color = meta['color']
        if isinstance(color, tuple):
            color = 'rgb%s' %str(color)

        # save the data
        yerr = {'type':'data', 'array':np.around(yerr, 5), 'visible':True, 'color':color, 'width':0}
        data.append(go.Scatter(x=np.around(x, 5), y=np.around(y, 5), error_y=yerr, line={'color':'transparent'}, showlegend=False))

    # plot the theory
    c = colors[::2][:driver.data.size]
    for i, res in enumerate(plot.plot_normalized_theory(driver, offset=0., color=c)):
        x, y, meta = res
        color = meta['color']
        if isinstance(color, tuple):
            color = 'rgb%s' %str(color)
        data.append(go.Scatter(x=np.around(x, 5), y=np.around(y, 5), line={'color':color}, name=labels_[i]))

    xlabel = plot.get_xlabel(driver, with_latex=True)
    ylabel = plot.get_ylabel(driver, with_latex=True)
    title = plot.get_title(driver, with_latex=True)

    layout = go.Layout(xaxis=dict(title=xlabel),
                       yaxis=dict(title=ylabel),
                       title=title,
                       hovermode='closest')

    # reversing data puts error bar points on top
    return go.Figure(data=data[::-1], layout=layout)
