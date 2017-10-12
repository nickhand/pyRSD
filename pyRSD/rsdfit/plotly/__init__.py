try:
    import plotly.offline as py
    import plotly.graph_objs as go
    import plotly.tools as pytools
except ImportError as e:
    raise ImportError("install 'plotly' to use interactive plotting features")

def enable_latex():
    """
    A workaround in :mod:`plotly` where latex labels are broken in notebooks.

    See also: https://github.com/plotly/plotly.py/issues/515
    """
    from IPython.core.display import display, HTML
    display(HTML(
    '<script>'
        'var waitForPlotly = setInterval( function() {'
            'if( typeof(window.Plotly) !== "undefined" ){'
                'MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });'
                'MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);'
                'clearInterval(waitForPlotly);'
            '}}, 250 );'
    '</script>'
    ))

    py.init_notebook_mode(connected=True)


from .fit import plot_fit_comparison
from .mcmc import jointplot_2d, hist_1d, plot_traces, plot_triangle
