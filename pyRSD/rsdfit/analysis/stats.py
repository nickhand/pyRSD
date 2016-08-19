from ... import numpy as np
from .. import logging
from . import tools

logger = logging.getLogger('rsdfit.analyze')
logger.addHandler(logging.NullHandler())

def gelman_rubin_convergence(chains):
    """
    Return the Gelman-Rubin convergence parameter
    """
    Nchains = len(chains)
    n, ndim = chains[0].shape
    withinchainvar, meanchain = [],[]
    for chain in chains:
        withinchainvar.append(np.var(chain, axis=0))
        meanchain.append(np.mean(chain, axis=0))
    
    meanall = np.mean(meanchain, axis=0)
    W = np.mean(withinchainvar, axis=0)
    B = np.zeros(ndim)

    for jj in range(0, Nchains):
        B += n*(meanall - meanchain[jj])**2 / (Nchains-1.)
    estvar = (1. - 1./n)*W + B/n
    return np.sqrt(estvar/W)

def convergence(info):
    """
    Compute convergence for the desired chains, using Gelman-Rubin diagnostic
    Chains have been stored in the info instance of :class:`Information`. Note
    that the G-R diagnostic can be computed for a single chain, albeit it will
    most probably give absurd results. To do so, it separates the chain into
    three subchains.
    """
    # prepare, if is isn't already
    if not info.prepared:
        tools.prepare(info)
        
    # thin?
    if info.thin != 1:
        tools.thin_chains(info)
        
    # Circle through all files to find the global maximum of likelihood
    logger.info('finding global maximum of likelihood')
    global_min_minus_lkl = info.min_minus_lkl

    # Restarting the circling through files, this time removing the burnin,
    # given the maximum of likelihood previously found and the global variable
    # LOG_LKL_CUTOFF. spam now contains all the accepted points that were
    # explored once the chain moved within min_minus_lkl - LOG_LKL_CUTOFF
    logger.info('removing burn-in')
    spam, _ = tools.remove_burnin(info)
    
    # Now that the list spam contains all the different chains removed of
    # their respective burn-in, proceed to the convergence computation
    logger.info('computing convergence criterion (Gelman-Rubin)')
    # Gelman Rubin Diagnostic:
    # Computes a quantity linked to the ratio of the mean of the variances of
    # the different chains (within), and the variance of the means (between)
    # Note: This is not strictly speaking the Gelman Rubin test, defined for
    # same-length MC chains. Our quantity is defined without the square root,
    # which should not change much the result: a small sqrt(R) will still be a
    # small R. The same convention is used in CosmoMC, except for the weighted
    # average: we decided to do the average taking into account that longer
    # chains should count more
    R = gelman_rubin_convergence(spam)

    for i in range(info.number_parameters):
        if i == 0:
            logger.info(' R is %.6f' %R[i] + '\tfor %s' %info.ref_names[i])
        else:
            logger.info('         %.6f' %R[i] + '\tfor %s' %info.ref_names[i])

    # Log finally the total number of steps, and absolute loglikelihood
    logger.info("total    number    of    steps: %d\n" %(info.steps))
    logger.info("total number of accepted steps: %d\n" %(info.accepted_steps))
    logger.info("minimum of -logLike           : %.2f" %(info.min_minus_lkl))

    # Store the remaining members in the info instance, for further writing to
    # files, storing only the mean and total of all the chains taken together
    info.R = R

    # Create the main chain, which consists in all elements of spam
    # put together. This will serve for the plotting.
    idx = [info.param_indices[name] for name in info.ref_names]
    info.chain = np.vstack([c[...,idx] for c in spam])

def minimum_credible_intervals(info):
    """
    Extract minimum credible intervals (method from Jan Haman)
    """
    histogram = info.hist
    bincenters = info.bincenters
    levels = [0.6826, 0.9545, 0.9973]

    bounds = np.zeros((len(levels), 2))
    j = 0
    delta = bincenters[1]-bincenters[0]
    left_edge = np.max(histogram[0] - 0.5*(histogram[1]-histogram[0]), 0.)
    right_edge = np.max(histogram[-1] + 0.5*(histogram[-1]-histogram[-2]), 0.)
    failed = False
    for level in levels:
        norm = float(
            (np.sum(histogram)-0.5*(histogram[0]+histogram[-1]))*delta)
        norm += 0.25*(left_edge+histogram[0])*delta
        norm += 0.25*(right_edge+histogram[-1])*delta
        water_level_up = np.max(histogram)*1.0
        water_level_down = np.min(histogram)*1.0
        top = 0.

        iterations = 0
        while (abs((top/norm)-level) > 0.0001) and not failed:
            top = 0.
            water_level = (water_level_up + water_level_down)/2.
            #ontop = [elem for elem in histogram if elem > water_level]
            indices = [i for i in range(len(histogram))
                       if histogram[i] > water_level]
            # check for multimodal posteriors
            if ((indices[-1]-indices[0]+1) != len(indices)):
                logger.warning("could not derive minimum credible intervals " + \
                        "for this multimodal posterior")
                failed = True
                break
            top = (np.sum(histogram[indices]) -
                   0.5*(histogram[indices[0]]+histogram[indices[-1]]))*(delta)

            # left
            if indices[0] > 0:
                top += (0.5*(water_level+histogram[indices[0]]) *
                        delta*(histogram[indices[0]]-water_level) /
                        (histogram[indices[0]]-histogram[indices[0]-1]))
            else:
                if (left_edge > water_level):
                    top += 0.25*(left_edge+histogram[indices[0]])*delta
                else:
                    top += (0.25*(water_level + histogram[indices[0]]) *
                            delta*(histogram[indices[0]]-water_level) /
                            (histogram[indices[0]]-left_edge))

            # right
            if indices[-1] < (len(histogram)-1):
                top += (0.5*(water_level + histogram[indices[-1]]) *
                        delta*(histogram[indices[-1]]-water_level) /
                        (histogram[indices[-1]]-histogram[indices[-1]+1]))
            else:
                if (right_edge > water_level):
                    top += 0.25*(right_edge+histogram[indices[-1]])*delta
                else:
                    top += (0.25*(water_level + histogram[indices[-1]]) *
                            delta * (histogram[indices[-1]]-water_level) /
                            (histogram[indices[-1]]-right_edge))

            if top/norm >= level:
                water_level_down = water_level
            else:
                water_level_up = water_level
            # safeguard, just in case
            iterations += 1
            if (iterations > 1000):
                warnings.warn(
                    "the loop to check for sigma deviations was " +
                    "taking too long to converge")
                break

        # min
        if indices[0] > 0:
            bounds[j][0] = bincenters[indices[0]] - delta*(histogram[indices[0]]-water_level)/(histogram[indices[0]]-histogram[indices[0]-1])
        else:
            if (left_edge > water_level):
                bounds[j][0] = bincenters[0]-0.5*delta
            else:
                bounds[j][0] = bincenters[indices[0]] - 0.5*delta*(histogram[indices[0]]-water_level)/(histogram[indices[0]]-left_edge)

        # max
        if indices[-1] < (len(histogram)-1):
            bounds[j][1] = bincenters[indices[-1]] + delta*(histogram[indices[-1]]-water_level)/(histogram[indices[-1]]-histogram[indices[-1]+1])
        else:
            if (right_edge > water_level):
                bounds[j][1] = bincenters[-1]+0.5*delta
            else:
                bounds[j][1] = bincenters[indices[-1]] + \
                    0.5*delta*(histogram[indices[-1]]-water_level) / \
                    (histogram[indices[-1]]-right_edge)

        j += 1

    name = info.ref_names[info.native_index]
    for elem in bounds:
        for j in (0, 1):
            elem[j] -= info.bestfit[name].flat_trace.mean()
    return bounds
