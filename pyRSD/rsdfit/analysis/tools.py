from ... import os, numpy as np
from .. import params_filename, logging
from ..util import rsd_io
from ..theory import GalaxyPowerTheory
from ..data import PowerData
from ..results import EmceeResults
from collections import defaultdict

LOG_LKL_CUTOFF = 3

logger = logging.getLogger('rsdfit.analyze')
logger.addHandler(logging.NullHandler())

def thin_chains(info):
    """
    Thin the chains
    """
    t = info.thin
    for r in info.chains:
        r.constrained_chain = r.constrained_chain[:,::t,:] 
        r.chain = r.chain[:,::t,:]
        r.lnprobs = r.lnprobs[:,::t]
        r._save_results()

def format_scaled_param(name, number):
    """
    Format the latex name of a parameter that has been scaled
    """
    if number == 1:
        return name
        
    left = name.find('$')
    right = name.rfind('$')
    if left != -1 and right != -1 and left != right:
        name = name[left+1:right]
        
    if number < 1000 and number > 1:
        name = "$%0.d~%s$" % (number, name)
    else:
        import re
        
        temp_name = "$%0.e%s$" % (number, name)
        m = re.search(r'(?:\$[0-9]*e\+[0]*)([0-9]*)(.*)', temp_name)
        sign = '+'
        if m is None:
            m = re.search(r'(?:\$[0-9]*e\-[0]*)([0-9]*)(.*)', temp_name)
            sign = '-'
        name = '$10^{'+sign+m.groups()[0]+'}'+m.groups()[1]

    return name
    
def remove_burnin(info, cutoff=LOG_LKL_CUTOFF):
    """
    Create an array with all the points from the chains, after burnin
    
    Returns
    -------
    chains : list of arrays
        A list of chains, where each chain contains
        the free + constrained params and 'burn-in' iterations have 
        been removed
    combined_result : EmceeResults
        a EmceeResults object holding the combined chains 
    """
    # prepare, if we need to
    if not info.prepared:
        info.prepare()
        
    # spam will brutally contain all the chains with sufficient number of
    # points, after the burn-in was removed.
    spam = list()

    # Recover the longest file name, for pleasing display
    max_name_length = max([len(e) for e in info.files])

    # Total number of steps done:
    steps = 0
    accepted_steps = 0

    new_results = []
    for index, chain_file in enumerate(info.files):
        result = info.chains[index].copy()
        
        # To improve presentation, and print only once the full path of the
        # analyzed folder, we recover the length of the path name, and
        # create an empty complementary string of this length
        total_length = 18+max_name_length
        empty_length = 18+len(os.path.dirname(chain_file))+1

        basename = os.path.basename(chain_file)
        if index == 0:
            exec("logger.info('Scanning file %-{0}s' % chain_file)".format(max_name_length))
        else:
            exec("logger.info('%{0}s%-{1}s' % ('', basename))".format(empty_length, total_length-empty_length))

        local_min_minus_lkl = -result.lnprobs.mean(axis=0)
        inds = local_min_minus_lkl < info.min_minus_lkl + cutoff
        if info.burnin is not None:
            burnin = int(info.burnin*result.iterations)
            inds[:burnin] = False
            
        steps += len(inds)
        accepted_steps += inds.sum()
        if not inds.sum():
            continue
            #raise rsd_io.AnalyzeError('no iterations left after removing burnin: chain not converged')
        else:
            logger.info('removed {0}/{1} iterations when discarding burn-in'.format(len(inds)-inds.sum(), len(inds)))

        # deal with single file case
        if len(info.chains) == 1:
            logger.warning("convergence computed for a single file...uh oh")
            bacon, egg, sausauge = result.copy(), result.copy(), result.copy()
            for i, x in enumerate([bacon, egg, sausage]):
                x.chain = np.copy(result.chain[:,inds,:][:, i::3, :])
                x.constrained_chain = np.copy(result.constrained_chain[:,inds][:, i::3])
                x.lnprobs = np.copy(result.lnprobs[:,inds][:,i::3])
                new_results.append(x)
            continue
        else:
            # ham contains chain without the burn-in, if there are any points
            result.chain = result.chain[:,inds,:]
            result.constrained_chain = result.constrained_chain[:,inds]
            result.lnprobs = result.lnprobs[:,inds]
            new_results.append(result)

    # test the length of the list
    if len(new_results) == 0:
        raise rsd_io.AnalyzeError("no sufficiently sized chains were found.")

    for r in new_results:
        
        # flatten to (XX, Np)
        x = r.chain.reshape((-1, len(r.free_names)))
        
        # flatten the structured array 
        y = r.constrained_chain.flatten()
        # grab only the constrained parameters that aren't vectors
        y = np.vstack([y[name] for name in info.constrained_names]).T
        spam.append(np.hstack((x,y)))
        
    # Applying now new rules for scales, if the name is contained in the
    # referenced names
    for name in info.ref_names:
        index = info.ref_names.index(name)
        num = info.param_indices[name]
        for i in range(len(spam)):
            spam[i][...,num] *= 1./info.scales[index, index]

    info.steps = steps
    info.accepted_steps = accepted_steps
    info.combined_result = new_results[0]
    if len(new_results) > 1:
        for r in new_results[1:]: info.combined_result += r
    info.combined_result.burnin = 0
    
    if info.rescale_errors:
        logger.info("rescaling error on parameters by %.3f" %info.error_rescaling)
        info.combined_result.error_rescaling = info.error_rescaling
    
    # compute the mean
    info.mean = np.array([info.combined_result[par].flat_trace.mean() for par in info.ref_names]) 
    
    return spam, info.combined_result


def prepare(info):
    """
    Scan the whole input folder, and include all chains in it.
    Since you can decide to analyze some file(s), or a complete folder, this
    function first needs to separate between the two cases.

    Parameters
    ----------
    info : AnalysisDriver instance
        Used to store the result
    files : list
        list of potentially only one element, containing the files to analyze.
        This can be only one file, or the encompassing folder, files
    """
    # grab all the files and load the result pickles
    folder, files, basename, chains = recover_folder_and_files(info.files)

    info.files    = files
    info.chains   = chains
    info.folder   = folder
    info.basename = basename

    info_path = os.path.join(folder, 'info')
    if info.save_output and not os.path.exists(info_path):
        os.makedirs(info_path)
    
    # load the theory params as a ParameterSet
    if not os.path.exists(os.path.join(folder, params_filename)):
        raise rsd_io.AnalyzeError("no parameter file in directory to analyze")
    param_path = os.path.join(folder, params_filename)
    theory = GalaxyPowerTheory(param_path)
    info.param_set = theory.fit_params
    info.Np = theory.ndim
    try:
        info.param_set.update_fiducial()
    except Exception as e:
        logger.warning("unable to update fiducial values: %s" %str(e))
        
    # also load the data so we get can number of bins
    data_params = PowerData(param_path)
    info.Nb = data_params.ndim
    info.Nmocks = data_params.params['covariance_Nmocks'].value

    # output paths for later use
    info.info_path = os.path.join(folder, 'info', 'params.info')
    info.free_tex_path = os.path.join(folder, 'info', 'free_params.tex')
    info.constrained_tex_path = os.path.join(folder, 'info', 'constrained_params.tex')
    info.cov_path = os.path.join(folder, 'info', 'covmat.dat')
    
    # recover parameter names and scales, creating tex names, etc
    extract_parameter_names(info)
    
    # we are prepared
    info.prepared = True

def recover_folder_and_files(files):
    """
    Distinguish the cases when analyze is called with files or folder
    Note that this takes place chronologically after the function
    `separate_files`
    """
    # make sure they all exists
    for f in files:
        if not os.path.exists(f):
            raise rsd_io.AnalyzeError('you provided a nonexistent file/folder: `%s`' %f)
        
    # The following list defines the substring that a chain should contain for
    # the code to recognise it as a proper chain.
    substrings = ['.npz', '__']
    limit = 10
    if len(files) == 1 and os.path.isdir(files[0]):
            folder = os.path.normpath(files[0])
            files = [os.path.join(folder, elem) for elem in os.listdir(folder)
                     if not os.path.isdir(os.path.join(folder, elem))
                     and not os.path.getsize(os.path.join(folder, elem)) < limit
                     and all([x in elem for x in substrings])]
        
    else:
        folder = os.path.relpath(
                os.path.dirname(os.path.realpath(files[0])), os.path.curdir)
        files = [os.path.join(folder, elem) for elem in os.listdir(folder)
                 if os.path.join(folder, elem) in np.copy(files)
                 and not os.path.isdir(os.path.join(folder, elem))
                 and not os.path.getsize(os.path.join(folder, elem)) < limit
                 and all([x in elem for x in substrings])]
    basename = os.path.basename(folder)
    
    chains = [EmceeResults.from_npz(f) for f in files]
    for i in range(1, len(chains)):
        chains[i].verify_param_ordering(chains[0].free_names, chains[0].constrained_names)
    return folder, files, basename, chains

def extract_parameter_names(info):
    """
    Extract parameter names from the results files
    """
    import itertools
    
    if not hasattr(info, 'chains'):
        raise rsd_io.AnalyzeError('cannot extract parameter names without ``chains`` attribute')
    
    # make sure all the free parameter names are the same
    free_names = [sorted(chain.free_names) for chain in info.chains]
    if not all(names == free_names[0] for names in free_names):
        raise rsd_io.AnalyzeError('mismatch in free parameters for loaded results to analyze')
        
    # make sure all the constrained parameter names are the same
    constrained_names = [sorted(chain.constrained_names) for chain in info.chains]
    if not all(names == constrained_names[0] for names in constrained_names):
        raise rsd_io.AnalyzeError('mismatch in constrained parameters for loaded results to analyze')
        
    # find constrained parameters that are vector
    info.vector_params = []
    for name in constrained_names:
        if info.chains[0].constrained_chain[name].dtype.subdtype is not None:
            info.vector_params.append(name)
                
    # remove vector parameters
    for name in info.vector_params:
        i = constrained_params.index(name)
        constrained_params.pop(i)
        
    # get the free and constrained param names
    info.free_names = free_names[0]
    info.constrained_names = constrained_names[0]
    param_names = info.free_names + info.constrained_names
    
    plot_params_1d = defaultdict(list)
    plot_params_2d = defaultdict(list)
    
    boundaries = []
    ref_names = []
    tex_names = []
    scales = []
    param_indices = {}
    
    # do 2D plot groups
    if len(info.to_plot_2d):
        keys = info.to_plot_2d.keys()
        if len(keys) == 1:
            for k in info.to_plot_2d[keys[0]]:
                if k in param_names:
                    plot_params_2d[keys[0]].append(k)
        else:
            combos = list(itertools.combinations(keys, 2))
            for i, j in combos:
                tag = "%s_vs_%s" %(i,j)
                tot = info.to_plot_2d[i] + info.to_plot_2d[j]
                for k in tot:
                    if k in param_names:
                        plot_params_2d[tag].append(k)
    else:
        plot_params_2d['free'] += info.free_names
                
    # loop over all parameters
    for name in param_names:
        par = info.param_set[name]       
        if not len(info.to_plot_1d):
            if name in info.free_names:
                plot_params_1d['free'].append(name)
            else:
                plot_params_1d['constrained'].append(name)
        else:
            if name in info.to_plot_1d:
                plot_params_1d['subset'].append(name)
        
        # append to the boundaries array
        bounds = [None, None]
        for i, k in enumerate(['min', 'max']):
            val = getattr(par, k)
            if val is not None and np.isfinite(val):
                bounds[i] = val
        boundaries.append(bounds)
        ref_names.append(name)
        scale = 1.
        if name in info.scales.keys():
            scale = info.scales[name]
        scales.append(scale)

        # given the scale, decide for the pretty tex name
        if name in info.tex_names:
            info.tex_names[name] = format_scaled_param(info.tex_names[name], 1./scale)
            tex_names.append(info.tex_names[name])
        else:
            raise ValueError("please specify a proper tex name for `%s`" %name)
            
        if name in info.free_names:
            param_indices[name] = info.free_names.index(name)
        else:
            param_indices[name] = len(info.free_names) + info.constrained_names.index(name)
            
    scales = np.diag(scales)
    info.ref_names = ref_names
    info.plot_tex_names = tex_names
    info.boundaries = boundaries
    info.scales = scales
    info.param_indices = param_indices
    
    # Beware, the following two numbers are different. The first is the total
    # number of parameters stored in the chain, whereas the second is for
    # plotting purpose only.
    info.number_parameters = len(ref_names)
    info.plot_params_1d    = plot_params_1d
    info.plot_params_2d    = plot_params_2d
    