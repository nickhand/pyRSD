from ... import os, numpy as np
from ..util import rsd_io
from . import stats
from .. import logging

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger('rsdfit.analyze')
logger.addHandler(logging.NullHandler())

def ctr_level(histogram2d, lvl, infinite=False):
    """
    Extract the contours for the 2d plots (Karim Benabed)
    """

    hist = histogram2d.flatten()*1.
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0]+[hist[-i] for i in alvl]+[hist.max()]
    if not infinite:
        return clist[1:]
    return clist
    
def cubic_interpolation(hist, bincenters):
    """
    Small routine to accomodate the absence of the interpolate module
    """
    from scipy.interpolate import interp1d
    interp_grid = np.linspace(bincenters[0], bincenters[-1], len(bincenters)*10)
    f = interp1d(bincenters, hist, kind='cubic')
    interp_hist = f(interp_grid)
    return interp_hist, interp_grid

def compute_and_plot_posteriors(info):
    """
    Compute the marginalized 1D and 2D posterior distributions, 
    and optionally plot them
    """    
    # set some plotting defaults
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=11)
    matplotlib.rc('xtick', labelsize='8')
    matplotlib.rc('ytick', labelsize='8')

    # Recover max and min values for each instance, defining the a priori place
    # of ticks (in case of a comparison, this should change)
    info.define_ticks()
    plot_path = os.path.join(info.folder, 'plots')
    if info.save_output and not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # plot the 1D posteriors first
    if info.plot:
        saved_subplots = set()
        for tag, param_names in info.plot_params_1d.items():
            plot_1d_posteriors(info, tag, param_names, saved_subplots)
        
    # plot the 2D posteriors second
    if info.plot and info.plot_2d:
        saved_subplots = set()
        for tag, param_names in info.plot_params_2d.items():
            plot_2d_posteriors(info, tag, param_names, saved_subplots)
                
def get_1d_histogram(info, trace, name):
    
    if hasattr(info, '_1d_posteriors') and name in info._1d_posteriors:
        return info._1d_posteriors[name]
    elif not hasattr(info, '_1d_posteriors'):
        info._1d_posteriors = {}
        
    logger.info('computing histograms for %s' %name)
    hist, bin_edges = np.histogram(trace, bins=info.bins, normed=False)
    bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    
    # interpolated histogram 
    interp_hist, interp_grid = cubic_interpolation(hist, bincenters)
    interp_hist /= np.max(interp_hist)
    info._1d_posteriors[name] = (interp_grid, interp_hist)
    return info._1d_posteriors[name]
    
def get_2d_histogram(info, index1, index2, trace1, trace2):
    
    key = tuple(sorted([index1, index2]))
    if hasattr(info, '_2d_posteriors') and key in info._2d_posteriors:
        return info._2d_posteriors[key]
    elif not hasattr(info, '_2d_posteriors'):
        info._2d_posteriors = {}
        
    bins = (info.bins, info.bins)
    n, xedges, yedges = np.histogram2d(trace1, trace2, bins=bins, normed=False)
    extent = [info.x_range[index2][0],
              info.x_range[index2][1],
              info.x_range[index1][0],
              info.x_range[index1][1]]
    x_centers = 0.5*(xedges[1:] + xedges[:-1])
    y_centers = 0.5*(yedges[1:] + yedges[:-1])
    info._2d_posteriors[key] = (x_centers, y_centers, n, extent)
    return info._2d_posteriors[key]
    
def get_mean_likelihood(info, trace, name):
    
    if hasattr(info, '_mean_likelihoods') and name in info._mean_likelihoods:
        return info._mean_likelihoods[name]
    elif not hasattr(info, '_mean_likelihoods'):
        info._mean_likelihoods = {}
        
    lnprobs     = info.combined_result.lnprobs.flatten()
    weights     = np.exp(info.min_minus_lkl+lnprobs)
    lkl_mean, bin_edges = np.histogram(trace, bins=info.bins, normed=False, weights=weights)
    bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    lkl_mean /= lkl_mean.max()
    interp_lkl_mean, interp_grid = cubic_interpolation(lkl_mean, bincenters)
    info._mean_likelihoods[name] = (interp_grid, interp_lkl_mean)
    return info._mean_likelihoods[name]
    
def get_bounds(info, name):
    """
    Get the 1, 2, and 3-sigma bounds for the specified parameter, 
    simply returning them if they already have been computed
    """
    index = info.ref_names.index(name)
    return info.bounds[index]


def plot_1d_posteriors(info, tag, param_names, saved_subplots):
    """
    Plot the 1D posteriors for the specified parameters on one figure
    """
    if not info.ticks_defined:
        info.define_ticks()
        
    Nplot = len(param_names) 
    if not Nplot:
        raise rsd_io.AnalyzeError("no parameters to mak 1D posterior plot for")
                                    
    # Find the appropriate number of columns and lines for the 1d posterior
    # plot
    num_columns = np.round(Nplot**0.5).astype('int')
    num_lines = np.ceil(1.*Nplot/num_columns).astype('int')
    fig = plt.figure(num=1, figsize=(3*num_columns,3*num_lines), dpi=80)
    
    logger.info('-----------------------------------------------')
    subplots = {}
    for index, name in enumerate(param_names):
        native_index = info.ref_names.index(name)
        par = info.combined_result[name]
        trace = par.flat_trace / info.scales[native_index, native_index]
        
        # adding the subplots to the respective figures
        ax = fig.add_subplot(num_lines, num_columns, index+1, yticks=[])
        ax.set_color_cycle(info.cm)
            
        bounds = get_bounds(info, name)
        
        ## set the title
        args = (info.plot_tex_names[native_index], trace.mean(),
                bounds[0, -1], bounds[0, 0])
        title = '%s=$%.{0}g^{{+%.{0}g}}_{{%.{0}g}}$'.format(info.decimal) %args
        ax.set_title(title, fontsize=info.fontsize, y=1.05)
        
        # the x-ticks
        ax.set_xticks(info.ticks[native_index])
        xticks = ['%.{0}g'.format(info.decimal) %s for s in info.ticks[native_index]]
        ax.set_xticklabels(xticks, fontsize=info.ticksize)
        ax.axis([info.x_range[native_index][0], info.x_range[native_index][1],0, 1.05])

        # actually plot
        interp = get_1d_histogram(info, trace, name)
        ax.plot(interp[0], interp[1], lw=info.line_width, ls='-')
        
        # fiducial?
        if name in info.fiducial:
            fid = info.fiducial[name]
        else:
            fid = info.param_set[name].fiducial
        if info.show_fiducial and fid is not None:
            ax.axvline(x=fid, lw=2, ls='--', alpha=0.6, c='Crimson', zorder=0)
        
        ax.set_color_cycle(info.cm)
        if info.mean_likelihood:
            try:
                interp = get_mean_likelihood(info, trace, name)
                ax.plot(interp[0], interp[1], ls='--', lw=info.line_width)
            except Exception as e:
                logger.warning('could not find likelihood contour for %s' %info.ref_names[native_index])
                
        if info.subplot:
            plot_name = 'posterior_1d_%s' %name
            filename = os.path.join(info.folder, 'plots', plot_name+'.'+info.extension)
            if name not in subplots:
                subplots[name] = (ax, filename)        
            hist_file_name = os.path.join(info.folder, 'info', plot_name+'.hist')
            logger.info("   writing 1D posterior to %s" %hist_file_name)
            rsd_io.write_histogram(hist_file_name, interp[0], interp[1])
                
    # save the subplots
    if info.subplot:
        for name in subplots:
            if name in saved_subplots:
                continue
            ax, filename = subplots[name]
            extent1d = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            logger.info("   saving 1D subplot %s" %filename)
            fig.savefig(filename, bbox_inches=extent1d.expanded(1.1, 1.4))
            saved_subplots.add(name)
            
    logger.info('-----------------------------------------------')
    logger.info('saving `{0}` 1D posterior figures to .{1} files'.format(tag, info.extension))
    fig.set_tight_layout(True)
    plot_name = 'posteriors_1d_%s' %tag
    plot_path = os.path.join(info.folder, 'plots', '{0}.{1}'.format(
                                plot_name, info.extension))
    fig.savefig(plot_path, bbox_inches=0)
    plt.close()
                
def plot_2d_posteriors(info, tag, param_names, saved_subplots):
    """
    Plot the 2D posteriors for the specified parameters on one figure
    """
    if not info.ticks_defined:
        info.define_ticks()
        
    Nplot = len(param_names) 
    if not Nplot:
        raise rsd_io.AnalyzeError("no parameters to mak 2D posterior plot for")
                                    
    # initialize the figure
    fig = plt.figure(num=2, figsize=(3*Nplot,3*Nplot), dpi=80)
    
    subplots = {}
    for index, name in enumerate(param_names):
        native_index = info.ref_names.index(name)
        par = info.combined_result[name]
        trace = par.flat_trace / info.scales[native_index, native_index]
        
        # setup the axes
        ax = fig.add_subplot(Nplot, Nplot, index*(Nplot+1)+1, yticks=[])
        ax.set_color_cycle(info.cm)
        
        # plot 1D
        interp = get_1d_histogram(info, trace, name)
        plot = ax.plot(interp[0], interp[1], linewidth=info.line_width, ls='-')
        
        # fiducial?
        if name in info.fiducial:
            fid1 = info.fiducial[name]
        else:
            fid1 = info.param_set[name].fiducial
        if info.show_fiducial and fid1 is not None:
            ax.axvline(x=fid1, lw=2, ls='--', alpha=0.6, c='Crimson', zorder=0)
            
        ax.set_xticks(info.ticks[native_index])
        bounds = get_bounds(info, name)
        
        # set the title
        args = (info.plot_tex_names[native_index],
                trace.mean(), bounds[0, -1],bounds[0, 0])
        title = '%s=$%.{0}g^{{+%.{0}g}}_{{%.{0}g}}$'.format(info.decimal) %args
        ax.set_title(title, fontsize=info.fontsize, y=1.05)
        ax.set_xlabel(info.plot_tex_names[native_index],fontsize=info.fontsize)

        # xtick labels
        ax.set_xticklabels(['%.{0}g'.format(info.decimal) % s
                             for s in info.ticks[native_index]],
                                fontsize=info.ticksize)
        ax.axis([info.x_range[native_index][0],
                   info.x_range[native_index][1], 0, 1.05])
                   
        ax.set_color_cycle(info.cm)
        # mean likelihood
        if info.mean_likelihood:
            try:
                interp = get_mean_likelihood(info, trace, name)
                ax.plot(interp[0], interp[1], ls='--', lw=info.line_width)
            except:
                logger.warning('could not find likelihood contour for %s' %info.ref_names[native_index])
                
        # do the rest of the triangle plot
        for second_index in range(index):
            second_name = param_names[second_index]
            native_second_index = info.ref_names.index(second_name)
            scale = info.scales[native_second_index, native_second_index]
            second_trace = info.combined_result[second_name].flat_trace / scale
            
            axsub = fig.add_subplot(Nplot, Nplot, index*Nplot+second_index+1)        
            # plotting contours, using the ctr_level method (from Karim
            # Benabed). Note that only the 1 and 2 sigma contours are
            # displayed (due to the line with info.levels[:2])
            interp = get_2d_histogram(info, native_index, native_second_index, par.flat_trace, second_trace)
            try:
                contours = axsub.contourf(interp[1], interp[0], interp[2],
                    extent=interp[3], levels=ctr_level(interp[2], [0.6826, 0.9545]),
                    zorder=4, cmap=info.cmaps[0], alpha=info.alphas[0])
            except Warning:
                   logger.warning("The routine could not find the contour of the " + \
                         "'%s-%s' 2d-plot" % (info.ref_names[native_index],
                                                info.ref_names[native_second_index]))
                                                
            axsub.axis([info.x_range[native_second_index][0], info.x_range[native_second_index][1], 
                        info.x_range[native_index][0], info.x_range[native_index][1]])
            
            # fiducial?
            if second_name in info.fiducial:
                fid2 = info.fiducial[second_name]
            else:
                fid2 = info.param_set[second_name].fiducial
            if info.show_fiducial:
                if fid1 is not None and fid2 is not None:
                    axsub.axvline(x=fid2, lw=2, ls='--', alpha=0.6, c='Crimson', zorder=5)
                    axsub.axhline(y=fid1, lw=2, ls='--', alpha=0.6, c='Crimson', zorder=5)
                    
                
            axsub.set_xticks(info.ticks[native_second_index])
            if index == Nplot-1:
                axsub.set_xticklabels(['%.{0}g'.format(info.decimal) % s for s in
                                         info.ticks[native_second_index]],
                                         fontsize=info.ticksize)
                axsub.set_xlabel(info.plot_tex_names[native_second_index],
                                    fontsize=info.fontsize)
            else:
                axsub.set_xticklabels([''])

            axsub.set_yticks(info.ticks[native_index])
            if second_index == 0:
                axsub.set_yticklabels(['%.{0}g'.format(info.decimal) % s for s in
                             info.ticks[native_index]],
                             fontsize=info.ticksize)
                axsub.set_ylabel(info.plot_tex_names[native_index],
                                    fontsize=info.fontsize)
            else:
                axsub.set_yticklabels([''])
                
            # Store the individual 2d plots.
            if info.subplot:
                filename = os.path.join(info.folder, 'plots','posterior_2d_%s-%s.%s' %(name, second_name,info.extension))
                key = tuple(sorted([name, second_name]))
                if key not in subplots:
                    subplots[key] = (axsub, filename)
                    
                # store the coordinates of the points
                hist_basename = os.path.join(info.folder, 'plots','posterior_2d_{0}-{1}'.format(name,second_name))
                logger.info("   writing contours to %s.dat" %hist_basename)
                rsd_io.store_contour_coordinates(hist_basename+'.dat', name, second_name, contours, [0.6826, 0.9545])
                logger.info("   writing 2D posterior to %s.hist" %hist_basename)
                rsd_io.write_histogram_2d(hist_basename+'.hist', interp[0], interp[1], interp[3], interp[2])
                                  
    # save the subplots
    if info.subplot:
        for name in subplots:
            if name in saved_subplots:
                continue
            ax, filename = subplots[name]
            area = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())    
            logger.info("   saving 2D subplot %s" %filename)
            fig.savefig(filename, bbox_inches=area.expanded(1.4, 1.4))
            saved_subplots.add(name)
            

    
    logger.info('-----------------------------------------------')
    logger.info('saving `{0}` 2D posterior figures to .{1} files'.format(tag, info.extension))
    fig.set_tight_layout(True)
    plot_name = 'posteriors_2d_triangle_%s' %tag
    plot_path = os.path.join(info.folder, 'plots', '{0}.{1}'.format(
                                plot_name, info.extension))
    fig.savefig(plot_path, bbox_inches=0)
    plt.close()


        
        

    
    
