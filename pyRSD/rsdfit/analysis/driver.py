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
            execfile(kwargs['optional_plot_file'], plot_file_vars)
            
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
        self.write_median_file()
                         
        if not self.minimal:
            # Computing 1,2 and 3-sigma errors, and plot. This will create the
            # triangle and 1d plot by default.
            plot.compute_and_plot_posteriors(self)

            logger.info('writing .info and .tex files')
            self.write_information_files()
    
    
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
    
    def write_covariance(self):
        """
        Write out the covariance matrix
        """
        if self.save_output:
            rsd_io.write_covariance_matrix(self.covar, self.ref_names, self.cov_path)
            
    def write_median_file(self):
        """
        Write out the file holding the parameter medians
        """
        if self.save_output:
            rsd_io.write_bestfit_file(self.combined_result, self.ref_names,
                                         self.medians_path, scales=np.diag(self.scales))
    
    def define_ticks(self):
        """
        Define the ticks
        """        
        self.max_values = np.array([self.combined_result[name].flat_trace.max()/self.scales[i,i] for i,name in enumerate(self.ref_names)])
        self.min_values = np.array([self.combined_result[name].flat_trace.min()/self.scales[i,i] for i,name in enumerate(self.ref_names)])
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

    def write_information_files(self):

        if not self.prepared:
            self.prepare()
        
        N_free = len(self.free_names)
        info_names = self.free_names + self.constrained_names
        indices = [self.ref_names.index(name) for name in info_names]
        tex_names = [self.tex_names[name] for name in info_names]
        info_names = [name.replace('$', '') for name in info_names]

        # Define the bestfit array
        self.bestfit_params = np.zeros(len(self.ref_names))
        best_idx = self.combined_result.lnprobs.argmax()
        for i, name in enumerate(self.ref_names):
            self.bestfit_params[i] = self.combined_result[name].flat_trace[best_idx]
                
        # write out the free and constrained tables
        self.write_tex(indices[:N_free], tex_names[:N_free], self.free_tex_path)
        self.write_tex(indices[N_free:], tex_names[N_free:], self.constrained_tex_path)
        
        # Write down to the .h_info file all necessary information
        self.write_h_info(indices, info_names)
        self.write_v_info(indices, info_names)

    def write_h_info(self, indices, info_names, filename=None):

        filename = self.h_info_path if filename is None else filename
        with open(filename, 'w') as h_info:
            h_info.write(' param names\t:  ')
            for name in info_names:
                h_info.write("%-14s" % name)

            tools.write_h(h_info, indices, 'R-1 values', '% .6f', self.R)
            tools.write_h(h_info, indices, 'Best Fit  ', '% .6e', self.bestfit_params)
            tools.write_h(h_info, indices, 'mean      ', '% .6e', self.mean)
            tools.write_h(h_info, indices, 'sigma     ', '% .6e',
                            (self.bounds[:, 0, 1]-self.bounds[:, 0, 0])/2.)
            h_info.write('\n')
            tools.write_h(h_info, indices, '1-sigma - ', '% .6e',
                    self.bounds[:, 0, 0])
            tools.write_h(h_info, indices, '1-sigma + ', '% .6e',
                    self.bounds[:, 0, 1])
            tools.write_h(h_info, indices, '2-sigma - ', '% .6e',
                    self.bounds[:, 1, 0])
            tools.write_h(h_info, indices, '2-sigma + ', '% .6e',
                    self.bounds[:, 1, 1])
            tools.write_h(h_info, indices, '3-sigma - ', '% .6e',
                    self.bounds[:, 2, 0])
            tools.write_h(h_info, indices, '3-sigma + ', '% .6e',
                    self.bounds[:, 2, 1])

            # bounds
            h_info.write('\n')
            tools.write_h(h_info, indices, '1-sigma > ', '% .6e',
                    self.mean+self.bounds[:, 0, 0])
            tools.write_h(h_info, indices, '1-sigma < ', '% .6e',
                    self.mean+self.bounds[:, 0, 1])
            tools.write_h(h_info, indices, '2-sigma > ', '% .6e',
                    self.mean+self.bounds[:, 1, 0])
            tools.write_h(h_info, indices, '2-sigma < ', '% .6e',
                    self.mean+self.bounds[:, 1, 1])
            tools.write_h(h_info, indices, '3-sigma > ', '% .6e',
                    self.mean+self.bounds[:, 2, 0])
            tools.write_h(h_info, indices, '3-sigma < ', '% .6e',
                    self.mean+self.bounds[:, 2, 1])

    def write_v_info(self, indices, info_names, filename=None):
        
        filename = self.v_info_path if filename is None else filename
        with open(filename, 'w') as v_info:
            v_info.write('%-15s\t:  %-11s' % ('param names', 'R-1'))
            v_info.write(' '.join(['%-11s' % elem for elem in [
                'Best fit', 'mean', 'sigma', '1-sigma -', '1-sigma +',
                '2-sigma -', '2-sigma +', '1-sigma >', '1-sigma <',
                '2-sigma >', '2-sigma <']]))
            for index, name in zip(indices, info_names):
                v_info.write('\n%-15s\t: % .4e' % (name, self.R[index]))
                v_info.write(' '.join(['% .4e' % elem for elem in [
                    self.bestfit_params[index], self.mean[index],
                    (self.bounds[index, 0, 1]-self.bounds[index, 0, 0])/2.,
                    self.bounds[index, 0, 0], self.bounds[index, 0, 1],
                    self.bounds[index, 1, 0], self.bounds[index, 1, 1],
                    self.mean[index]+self.bounds[index, 0, 0],
                    self.mean[index]+self.bounds[index, 0, 1],
                    self.mean[index]+self.bounds[index, 1, 0],
                    self.mean[index]+self.bounds[index, 1, 1]]]))

    def write_tex(self, indices, tex_names, filename):
        
        with open(filename, 'w') as tex:
            tex.write("\\begin{tabular}{|l|c|c|c|c|} \n \\hline \n")
            tex.write("Param & best-fit & mean$\pm\sigma$ ")
            tex.write("& 95\% lower & 95\% upper \\\\ \\hline \n")
            for index, name in zip(indices, tex_names):
                tex.write("%s &" % name)
                tex.write("$%.4g$ & $%.4g_{%.2g}^{+%.2g}$ " % (
                    self.bestfit_params[index], self.mean[index],
                    self.bounds[index, 0, 0], self.bounds[index, 0, 1]))
                tex.write("& $%.4g$ & $%.4g$ \\\\ \n" % (
                    self.mean[index]+self.bounds[index, 1, 0],
                    self.mean[index]+self.bounds[index, 1, 1]))

            tex.write("\\hline \n \\end{tabular} \\\\ \n")
            tex.write("$-\ln{\cal L}_\mathrm{min} =%.6g$, " % (
                self.min_minus_lkl))
            tex.write("minimum $\chi^2=%.4g$ \\\\ \n" % (
                self.min_minus_lkl*2.))

