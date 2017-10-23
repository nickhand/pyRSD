from . import py, go, pytools
from . import tools
import numpy as np
import matplotlib.mlab as mlab
from pyRSD.rsdfit.analysis import tex_names

def plot_traces(r, names, burnin=None, max_walkers=10, opacity=0.4, rangeslider=False):
    """
    Plot the traces of the specified parameters using :mod:`plotly`.

    .. note::
        This is designed to be used in a notebook environment.
    """
    N = len(names)
    if N < 1:
        raise ValueError('Must specify at least one parameter name for trace plot')

    valid = r.free_names + r.constrained_names
    if not all(name in valid for name in names):
        raise ValueError("specified parameter names not valid")

    fig = pytools.make_subplots(rows=N, print_grid=False, shared_xaxes=True, start_cell='bottom-left')

    if burnin is None:
        burnin = r.burnin

    for i, name in enumerate(names):
        param = r[name]

        iter_num = np.arange(r.iterations, dtype='i4')[burnin:]
        trace = np.around(param.trace()[:, burnin:], 5)

        # plot each walker
        for itrace, t in enumerate(trace):
            if itrace > max_walkers: break
            line = go.Scatter(x=iter_num, y=t, line={'color':'#7F7F7F'}, hoverinfo='none',
                                opacity=opacity, showlegend=False)
            fig.append_trace(line, i+1, 1)

        # plot the mean
        fig.append_trace(go.Scatter(x=iter_num, y=trace.mean(axis=0), line={'color':'Crimson'},
                                    showlegend=False, name='mean', hoverinfo='closest'), i+1, 1)

        # label the y axes
        if name in tex_names:
            name = tex_names[name]
        fig['layout'].update(**{'yaxis'+str(i+1):dict(title=name)})

    # label the x axis
    xaxis = {'title':'iteration number'}
    if rangeslider: xaxis['rangeslider'] = dict()
    fig['layout'].update(xaxis=xaxis)

    # and plot
    return fig

def jointplot_2d(r, param1, param2,
                    thin=5,
                    rename={},
                    cmap='magma'):
    """
    Make a 2D plot of the joint posterior distributions of the specified
    parameters using kernel density estimation and :mod:`plotly`.

    .. note::
        This is designed to be used in a notebook environment.
    """
    valid = r.free_names + r.constrained_names
    if not all(name in valid for name in [param1, param2]):
        raise ValueError("specified parameter names not valid")

    # default names
    rename.setdefault(param1, tex_names.get(param1, param1))
    rename.setdefault(param2, tex_names.get(param2, param2))

    # make the pandas Series of the flattened traces
    trace1 = r[param1].trace()[:, r.burnin::thin].flatten()
    trace2 = r[param2].trace()[:, r.burnin::thin].flatten()

    # use 3-sigma limits
    xlims = [r[param1].median + x for x in r[param1].three_sigma]
    ylims = [r[param2].median + x for x in r[param2].three_sigma]

    # do the KDE
    x, y, Z = tools.kde_scipy(trace1, trace2, xlims, ylims, N=200)

    # the colors we want to use
    colors = tools.matplotlib_to_plotly(cmap, 6)

    # data and layout
    data = go.Data([go.Contour(
           z=np.around(Z, 5),
           x=np.around(x, 5),
           y=np.around(y, 5),
        showscale=False,
           colorscale=colors,
           opacity=0.9,
           contours=go.Contours(
               showlines=False)
        ),
     ])

    layout = go.Layout(
        showlegend=False,
        autosize=False,
        width=650,
        height=650,
        xaxis=go.XAxis(
            range=xlims,
            showgrid=False,
            nticks=7,
            title=rename[param1]
        ),
        yaxis=go.YAxis(
            range=ylims,
            showgrid=False,
            nticks=7,
            title=rename[param2]
        ),
    )

    # and plot
    return go.Figure(data=data, layout=layout)

def hist_1d(r, name, thin=1, rename=None, color="#1f77b4", hide_shapes=False):
    """
    Make a 1D histogram plot of the posterior distributions of the specified
    parameter using :mod:`plotly`.

    .. note::
        This is designed to be used in a notebook environment.
    """
    valid = r.free_names + r.constrained_names
    if not name in valid:
        raise ValueError("specified parameter names not valid")

    # make the pandas Series of the flattened traces
    par = r[name]
    trace = par.trace()[:, r.burnin::thin].flatten()

    counts, bins = np.histogram(trace, bins='auto')
    dx = np.diff(bins)
    counts = counts.astype(float) / (dx*counts).sum()
    bincenters = 0.5*(bins[1:] + bins[:-1])

    xlims = [r[name].median + x for x in par.three_sigma]
    data = [go.Bar(x=np.around(bincenters, 5), y=np.around(counts, 5), width=dx, opacity=0.75, showlegend=False,
                    marker={'color':color})]

    mu = trace.mean(); sigma = trace.std()
    x = np.linspace(0.9*trace.min(), 1.1*trace.max(), 500)
    y = mlab.normpdf(x, mu, sigma)
    data.append(go.Scatter(x=np.around(x, 5), y=np.around(y, 5), line={'color':'black'}, showlegend=False))

    # setup shapes for vertical lines
    if not hide_shapes:
        shapes_dict = {'type':'line', 'xref':'x', 'yref':'y', 'y0':0., 'y1':1.05*counts.max(), 'fillcolor':'black'}
        shapes = []

        # add the median value
        s = shapes_dict.copy()
        s['x0'] = s['x1'] = par.median
        shapes.append(s)

        # add the 1 sigma bounds
        for val in par.one_sigma:
            s = shapes_dict.copy()
            s['x0'] = s['x1'] = par.median + val
            s['line'] = {'dash':'dash'}
            shapes.append(s)

    if rename is not None: name = rename
    if name in tex_names and rename is None:
        name = tex_names[name]

    # add annotations for mu and sigma
    title = {}
    title['xref'] = 'x'; title['yref'] = 'y'
    title['showarrow'] = False
    text = r"$\mu = %.3f \pm %.3f$" %(mu, sigma)

    # mean annotation
    xcen = 0.5*(xlims[1] + xlims[0])
    title.update(x=xcen, y=1.1*counts.max(), text=text)

    layout = dict(
        autosize=False,
        width=650,
        height=650,
        annotations=[title],
        xaxis=dict(title=name, range=xlims),
        yaxis=dict(title=r"$N_\mathrm{samples}$"),
    )
    if not hide_shapes:
        layout['shapes'] = shapes

    return go.Figure(data=data, layout=layout)

def plot_triangle(r, params=None, thin=1, width=1500, height=1500, hide_shapes=False):

    if params is None:
        params = r.free_names + r.constrained_names

    N = len(params)
    fig = pytools.make_subplots(rows=N, cols=N, print_grid=False,
                                horizontal_spacing=0.05, vertical_spacing=0.05)

    data = []; annotations = []; shapes = []
    cnt = 1
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i < j:
                fig['layout']['xaxis'+str(cnt)].update({'visible':False})
                fig['layout']['yaxis'+str(cnt)].update({'visible':False})
                cnt += 1
                continue

            if i != j:
                subfig = jointplot_2d(r, params[j-1], params[i-1], thin=thin)
                sp = subfig['data']
            else:
                subfig = hist_1d(r, params[j-1], hide_shapes=hide_shapes)
                sp = subfig['data']

                # update axes for annotations
                for t in subfig['layout']['annotations']:
                    t['xref'] = 'x'+str(cnt)
                    t['yref'] = 'y'+str(cnt)
                annotations += subfig['layout']['annotations']

                # update axes for shapes
                if not hide_shapes:
                    for t in subfig['layout']['shapes']:
                        t['xref'] = 'x'+str(cnt)
                        t['yref'] = 'y'+str(cnt)
                    shapes += subfig['layout']['shapes']

            xaxis = subfig['layout']['xaxis']
            yaxis = subfig['layout']['yaxis']
            if i != N: xaxis.pop('title')
            if j != 1 or j == 1 and i == 1: yaxis.pop('title')
            fig['layout']['xaxis'+str(cnt)].update(xaxis)
            fig['layout']['yaxis'+str(cnt)].update(yaxis)

            for ea in sp:
                ea.update(xaxis='x{}'.format(cnt),
                          yaxis='y{}'.format(cnt),
                          name='{0} vs {1}'.format(params[j-1],params[i-1]))
            data += sp
            cnt += 1

    fig['layout'].update(height=height, width=width, annotations=annotations)
    if not hide_shapes:
        fig['layout'].update(shapes=shapes)
    fig['data'] += data
    return fig
