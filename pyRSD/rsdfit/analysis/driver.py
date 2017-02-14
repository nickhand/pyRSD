"""
    driver.py
    pyRSD.rsdfit.analysis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : extract data from rsdfit chains and produce plots
    
    notes:
        parts of this moduel are directly adapted from the 
        `Monte Python` <http://montepython.net/>` code from Baudren et. al.
"""
from __future__ import print_function

from ... import os, numpy as np
from ..util import rsd_io
from . import tools, stats, plot
from .. import logging

logger = logging.getLogger('rsdfit.analyze')
logger.addHandler(logging.NullHandler())

def add_console_logger():
    """
    Add a logger that logs to the console at level `INFO`.
    """ 
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

def set_analysis_defaults(kwargs):
    """
    Set the keyword defaults for the ``AnalysisDriver`` class
    """
    kwargs.setdefault('minimal', False)
    kwargs.setdefault('bins', 20)
    kwargs.setdefault('mean_likelihood', True)
    kwargs.setdefault('plot', True)
    kwargs.setdefault('plot_2d', True)
    kwargs.setdefault('contours_only', False)
    kwargs.setdefault('subplot', False)
    kwargs.setdefault('extension', 'pdf')
    kwargs.setdefault('fontsize', 16)
    kwargs.setdefault('ticksize', 14)
    kwargs.setdefault('line_width', 4)
    kwargs.setdefault('decimal', 3)
    kwargs.setdefault('ticknumber', 3)
    kwargs.setdefault('optional_plot_file', '')
    kwargs.setdefault('tex_names', {})
    kwargs.setdefault('to_plot_1d', [])
    kwargs.setdefault('to_plot_2d', {})
    kwargs.setdefault('scales', {})
    kwargs.setdefault('save_output', True)
    kwargs.setdefault('show_fiducial', True)
    kwargs.setdefault('fiducial', {})
    kwargs.setdefault('burnin', None)
    kwargs.setdefault('thin', 1)
    kwargs.setdefault('rescale_errors', False)
    

class AnalysisDriver(object):
    """
    A class to serve as the driver for analyzing runs
    """    
    # Global colormap for the 1d plots. Colours will get chosen from this.
    # Some old versions of matplotlib do not have CMRmap, so the colours will
    # be harcoded
    # Note that, as with the other customisation options, you can specify new
    # values for this in the extra plot_file.
    import matplotlib.pyplot as plt
    cm = [
        (0.,      0.,      0.,      1.),
        (0.30235, 0.15039, 0.74804, 1.),
        (0.99843, 0.25392, 0.14765, 1.),
        (0.90000, 0.75353, 0.10941, 1.)]

    # Define colormaps for the contour plots
    cmaps = [plt.cm.gray_r, plt.cm.Purples, plt.cm.Reds_r]
    alphas = [1.0, 0.8, 0.6, 0.4]

    def __init__(self, **kwargs):
        """    
        Parameters
        ----------
        kwargs : key/value pairs
            the relevant keyword arguments are:
                files : list of str
                    list of a directory or series of files to analyze
                minimal : bool, optional (`False`)
                    if `True`, only write the covmat and bestfit, without 
                    computing the posterior or making plots. 
                bins : int, optional (20)
                    number of bins in the histograms used to derive posterior 
                mean_likelihood : bool, optional (`True`)
                    show the mean likelihood on the 1D posterior plots
                plot : bool, optional (`True`)
                    if `False`, do not make any plots, simply compute the posterior
                plot_2d : bool, optional (`True`)
                    if `False`, do not produce the 2D posterior plots
                contours_only : bool, optional (`False`)
                    if `True`, do not fill the contours on the 2d plots
                subplot : bool, optional (`False`)
                    if `True`, output every subplot and data in separate files
                extension : {'pdf', 'png', 'eps'}, optional (`pdf`)
                    the extension to use for output plots
                fontsize : int, optional (16)
                    the fontsize to use on the plots
                ticksize : int, optional (14)
                    the ticksize to use on the plots
                line_width : int, optional (4)
                    the line-width of 1D plots
                decimal : int, optional (3)
                    the number of decimal places on ticks
                ticknumber : int, optional (3)
                    the number of ticks on each axis
                optional_plot_file : str, optional ("")
                    extra file to customize the output plots
                tex_names : dict, optional, (`{}`)
                    dict holding a latex name to use for each parameter
                to_plot_1d : list, optional (`[]`)
                    list of parameters to plot 1D posteriors of
                to_plot_2d : dict, optional (`{}`)
                    dict holding groups of parameters to make 2D plots of
                scales : dict, optional (`{}`)
                    dict holding the rescaling factors that the posterior
                    will be divided by
                save_output : bool, optional (`True`)
                    if `False`, do not save any output or make new directories
                show_fiducial : bool, optional (`True`)
                    whether to show the fiducial values as vertical lines on 
                    the 1D posterior plots
                fiducial : dict, optional {`{}`}
                    a dictionary holding fidicual values to use, which will override
                    the original fiducial values
                burnin : float, optional (`None`)
                    the fraction of samples to consider burnin
        """
        add_console_logger()
        
        # set the defaults in kwargs
        if 'files' not in kwargs:
            raise rsd_io.AnalyzeError("please specify a list of files or directories to analyze")
        set_analysis_defaults(kwargs)

        # reference names for all parameters
        self.ref_names = []

        # store directly all information from the command_line object into this
        # instance, except the protected members (begin and end with __)
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # read a potential file describing changes to be done for the parameter
        # names, and number of paramaters plotted (can be let empty, all will
        # then be plotted), but also the style of the plot. Note that this
        # overrides the command line options
        if kwargs['optional_plot_file']:
            plot_file_vars = {'analyze': self}
            with open(kwargs['optional_plot_file'], 'r') as f:
                code = compile(f.read(), kwargs['optional_plot_file'], 'exec')
                exec(code, plot_file_vars)
            
        self.prepared = False
        self.ticks_defined = False
    
    def run(self):
        """
        Run the full analysis of the input mcmc files
        """
        # load the files into the `chains` attribute
        tools.prepare(self)
        
        # Compute the mean, maximum of likelihood, 1-sigma variance for this
        # main folder. This will create the info.chain object, which contains
        # all the points computed stacked in one big array.
        stats.convergence(self)
        
        # compute the convariance matrix
        logger.info('computing covariance matrix')
        self.write_covariance()
                         
        if not self.minimal:
            # Computing 1,2 and 3-sigma errors, and plot. This will create the
            # triangle and 1d plot by default.
            plot.compute_and_plot_posteriors(self)

            logger.info('writing .info and .tex files')
            self.write_information_files()
            
        # save the combined output
        bestfit_path = os.path.join(self.folder, 'info', 'combined_result.npz')
        self.combined_result.to_npz(bestfit_path)
        print(self.combined_result)
    
    #--------------------------------------------------------------------------
    # Properties
    #--------------------------------------------------------------------------
    @property
    def error_rescaling(self):
        """
        Scale the error on parameters due to covariance matrix from mocks
        """
        try:
            return self._error_rescaling
        except:
            Nb = self.Nb*1.
            Ns = self.Nmocks*1.
            Np = self.Np*1.
            A = 2. / (Ns - Nb - 1.) / (Ns - Nb - 4.)
            B = (Ns - Nb - 2.) / (Ns - Nb - 1.) / (Ns - Nb - 4.)
            self._error_rescaling = ((1 + B*(Nb-Np)) / (1 + A + B*(Np+1)))**0.5
            return self._error_rescaling
            
    @property
    def min_minus_lkl(self):
        """
        The global minimum of the minus log likelihood, across all chains
        that we are analyzing
        """
        try:
            return self._min_minus_lkl
        except AttributeError:
            if not hasattr(self, 'chains'):
                raise AttributeError("trying to compute ``min_minus_lkl`` without the ``chains`` attribute")
            self._min_minus_lkl = min([-result.lnprobs.mean(axis=0).max() for result in self.chains])
            return self._min_minus_lkl
            
    @property
    def covar(self):
        """
        The covariance matrix of free + constrained parameters
        """
        try:
            return self._covar
        except AttributeError:
            if not hasattr(self, 'chain'):
                raise AttributeError("trying to compute ``covar`` without the ``chain`` attribute")
            self._covar = np.cov(self.chain, rowvar=0)
            return self._covar
    
    @property
    def bounds(self):
        """
        The 1, 2, and 3-sigma bounds
        """
        try:
            return self._bounds
        except AttributeError:
            if not hasattr(self, 'combined_result'):
                raise AttributeError("trying to compute ``bounds`` without the ``combined_result`` attribute")
        bounds = np.zeros((len(self.ref_names), 3, 2))
        
        # bounds from the 1,2,3 sigma percentiles
        for index, name in enumerate(self.ref_names):
            par = self.combined_result[name]
            this_bounds = np.array([par.one_sigma, par.two_sigma, par.three_sigma])
            scale = 1.
            if hasattr(self, 'scales'):
                scale = self.scales[index, index]
            bounds[index] = this_bounds / scale
        self._bounds = bounds
        return self._bounds
            
    @property
    def bestfit_params(self):
        """
        Bestfit parameter set
        """
        try:
            return self._bestfit_params
        except AttributeError:
            from . import bestfit
            names = self.free_names + self.constrained_names
            indices = [self.ref_names.index(name) for name in names]
            
            data = []
            best_idx = self.combined_result.lnprobs.argmax()
            for i, name in zip(indices, names):
                par = self.combined_result[name] 
                best = par.flat_trace[best_idx] / self.scales[i,i]
                median = par.median / self.scales[i, i]
                d = [self.tex_names[name], 
                        self.scales[i, i],
                        self.R[i], 
                        best, 
                        median,
                        0.5*(self.bounds[i, 0, 1]-self.bounds[i, 0, 0]),
                        self.bounds[i, 0, 0], 
                        self.bounds[i, 0, 1],
                        self.bounds[i, 1, 0], 
                        self.bounds[i, 1, 1],
                        median+self.bounds[i, 0, 0],
                        median+self.bounds[i, 0, 1],
                        median+self.bounds[i, 1, 0],
                        median+self.bounds[i, 1, 1], 
                        name in self.free_names]
                data.append(d)
                
            columns = ['tex_name', 'scale', 'R', 'best_fit', 'median', 'sigma', 'lower_1sigma', 'upper_1sigma',
                        'lower_2sigma', 'upper_2sigma', 'gt_1sigma', 'lt_1sigma',
                        'lt_2sigma', 'gt_2sigma', 'free']
            self._bestfit_params = bestfit.BestfitParameterSet(data, index=names, columns=columns)
            meta = {'min_minus_lkl' : self.min_minus_lkl, 'Np' : self.Np, 'Nb' : self.Nb}
            self._bestfit_params.add_metadata(**meta)
            return self._bestfit_params
                
    def define_ticks(self):
        """
        Define the min and max x-range values and set the axis ticks accordingly
        """        
        self.max_values = np.empty(len(self.ref_names))
        self.min_values = np.empty(len(self.ref_names))
        for i, name in enumerate(self.ref_names):
            trace = self.combined_result[name].flat_trace
            avg = trace.mean()
            stddev = trace.std()
            self.max_values[i] = (avg + 3.5*stddev) / self.scales[i,i]
            self.min_values[i] = (avg - 3.5*stddev) / self.scales[i,i]
        self.span = (self.max_values-self.min_values)
        # Define the place of ticks, given the number of ticks desired, stored
        # in conf.ticknumber
        self.ticks = np.array(
            [np.linspace(self.min_values[i]+self.span[i]*0.1,
                         self.max_values[i]-self.span[i]*0.1,
                         self.ticknumber) for i in range(len(self.span))])
        # Define the x range (ticks start not exactly at the range boundary to
        # avoid display issues)
        self.x_range = np.array((self.min_values, self.max_values)).T

        # In case the exploration hit a boundary (as defined in the parameter
        # file), at the level of precision defined by the number of bins, the
        # ticks and x_range should be altered in order to display this
        # meaningful number instead.
        for i in range(np.shape(self.ticks)[0]):
            x_range = self.x_range[i]
            bounds = self.boundaries[i]
            # Left boundary
            if bounds[0] is not None:
                if abs(x_range[0]-bounds[0]) < self.span[i]/self.bins:
                    self.ticks[i][0] = bounds[0]
                    self.x_range[i][0] = bounds[0]
            # Right boundary
            if bounds[-1] is not None:
                if abs(x_range[-1]-bounds[-1]) < self.span[i]/self.bins:
                    self.ticks[i][-1] = bounds[-1]
                    self.x_range[i][-1] = bounds[-1]
                    
        self.ticks_defined = True
        
    def write_covariance(self):
        """
        Write out the covariance matrix
        """
        if self.save_output:
            rsd_io.write_covariance_matrix(self.covar, self.ref_names, self.cov_path)

    def write_information_files(self):
        """
        Write out the best-fit info file and the latex tables
        """
        if not self.prepared:
            self.prepare()
        
        # write down to the .info file all necessary information
        self.bestfit_params.to_info(self.info_path)
        
        # write out the latex table for free parameters
        free_params = self.bestfit_params.loc[self.free_names]
        free_params.to_latex(self.free_tex_path)
        
        # write out the latex table for free parameters
        constrained_params = self.bestfit_params.loc[self.constrained_names]
        constrained_params.to_latex(self.constrained_tex_path)




