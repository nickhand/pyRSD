from __future__ import print_function

from ... import os, numpy as np
from datetime import date
import pickle
import copyreg
import types
from six import PY3

class PickeableClass(type):
    def __init__(cls, name, bases, attrs):
        copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def _pickle_method(method):
    if not PY3:
        func_name = method.im_func.__name__
        obj = method.im_self
        cls = method.im_class
    else:
        func_name = method.__func__.__name__
        obj = method.__self__
        cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def load_pickle(filename):
    """
    Load an instance, i.e., `FittingDriver`, `EmceeResults` that has been
    pickled
    """
    try:
        return pickle.load(open(filename, 'r'))
    except Exception as e:
        raise ConfigurationError("Cannot load the pickle `%s`; original message: %s" %(filename, e))

def save_pickle(obj, filename):
    """
    Pickle an instance using `pickle`
    """
    # make sure pool is None, so it is pickable
    pickle.dump(obj, open(filename, 'w'))

def load_model(filename, **kwargs):
    """
    Load a model from file
    """
    if not os.path.exists(filename):
        raise ConfigurationError('cannot load model from file `%s`; does not exist' %filename)
    _, ext = os.path.splitext(filename)
    if ext == '.npy':
        from ...rsd import load_model
        model = load_model(filename, **kwargs)
    elif ext == '.pickle':
        model = load_pickle(filename)
    else:
        raise ValueError("extension for model file not recognized; must be `.npy` or `.pickle`")

    return model

def create_output_file(folder, solver_type, chain_number, iterations, walkers=0, restart=None):
    """
    Automatically create a new name for the results file.

    This routine takes care of organizing the folder for you. It will
    automatically generate names for the new chains according to the date,
    number of points chosen.
    """
    if solver_type == 'mcmc':
        tag = "{}x{}".format(walkers, iterations)
    else:
        tag = solver_type + '_%di' %iterations

    # output file
    if restart is None:
        outname_base = '{0}_{1}_chain{2}__'.format(date.today(), tag, chain_number)
    else:
        try:
            # need to extract the original chain number
            fields = os.path.basename(restart).split('_')
            if solver_type == 'mcmc':
                outname_base = '{0}_{1}_{2}__'.format(date.today(), tag, fields[2])
            else:
                outname_base = '{0}_{1}_{2}__'.format(date.today(), tag, fields[3])
        except:
            outname_base = '{0}_{1}_chain{2}__'.format(date.today(), tag, chain_number)

    suffix = 0
    for files in os.listdir(folder):
        if files.find(outname_base) != -1:
            if int(files.split('__')[-1].split('.')[0]) > suffix:
                suffix = int(files.split('__')[-1].split('.')[0])
    suffix += 1
    while True:
        fname = os.path.join(folder, outname_base)+str(suffix)+'.npz'
        if os.path.exists(fname):
            suffix += 1
        else:
            break
    outfile_name = os.path.join(folder, outname_base)+str(suffix)+'.npz'
    print('Creating %s\n' %outfile_name)

    # touch the file so it exists and then return
    open(outfile_name, 'a').close()
    return outfile_name

class ConfigurationError(Exception):
    """Missing files, parameters, etc..."""
    pass

class AnalyzeError(Exception):
    """Used when encountering a fatal mistake in analyzing chains"""
    pass

def write_covariance_matrix(covariance_matrix, names, path):
    """
    Store the covariance matrix to a file
    """
    with open(path, 'w') as cov:
        cov.write('# %s\n' % ', '.join(['%16s' % name for name in names]))

        for i in range(len(names)):
            for j in range(len(names)):
                if covariance_matrix[i][j] > 0:
                    cov.write(' %.5e\t' % covariance_matrix[i][j])
                else:
                    cov.write('%.5e\t' % covariance_matrix[i][j])
            cov.write('\n')

def write_bestfit_file(bestfit, names, path, scales=None):
    """
    Store the bestfit parameters to a file
    """
    if scales is None:
        scales = [1.]*len(names)
    with open(path, 'w') as bestfit_file:
        for i, name in enumerate(names):
            bf_value = bestfit[name].mean*scales[i]
            if bf_value > 0:
                bestfit_file.write('%-15s =  %.5e\n' %(name, bf_value))
            else:
                bestfit_file.write('%-15s = %.5e\n' %(name, bf_value))
        bestfit_file.write('\n')

def write_histogram(hist_file_name, x_centers, hist):
    """
    Store the posterior distribution to a file
    """
    with open(hist_file_name, 'w') as hist_file:
        hist_file.write("# 1d posterior distribution\n")
        hist_file.write("\n# x_centers\n")
        hist_file.write(", ".join(
            [str(elem) for elem in x_centers])+"\n")
        hist_file.write("\n# Histogram\n")
        hist_file.write(", ".join(
            [str(elem) for elem in hist])+"\n")

def read_histogram(histogram_path):
    """
    Recover a stored 1d posterior
    """
    with open(histogram_path, 'r') as hist_file:
        for line in hist_file:
            if line:
                if line.find("# x_centers") != -1:
                    x_centers = [float(elem) for elem in
                                 hist_file.next().split(",")]
                elif line.find("# Histogram") != -1:
                    hist = [float(elem) for elem in
                            hist_file.next().split(",")]
    x_centers = np.array(x_centers)
    hist = np.array(hist)

    return x_centers, hist


def write_histogram_2d(hist_file_name, x_centers, y_centers, extent, hist):
    """
    Store the histogram information to a file, to plot it later
    """
    with open(hist_file_name, 'w') as hist_file:
        hist_file.write("# Interpolated histogram\n")
        hist_file.write("\n# x_centers\n")
        hist_file.write(", ".join(
            [str(elem) for elem in x_centers])+"\n")

        hist_file.write("\n# y_centers\n")
        hist_file.write(", ".join(
            [str(elem) for elem in y_centers])+"\n")

        hist_file.write("\n# Extent\n")
        hist_file.write(", ".join(
            [str(elem) for elem in extent])+"\n")

        hist_file.write("\n# Histogram\n")
        for line in hist:
            hist_file.write(", ".join(
                [str(elem) for elem in line])+"\n")


def read_histogram_2d(histogram_path):
    """
    Read the histogram information that was stored in a file.
    To use it, call something like this:
    .. code::
        x_centers, y_centers, extent, hist = read_histogram_2d_from_file(path)
        fig, ax = plt.subplots()
        ax.contourf(
            y_centers, x_centers, hist, extent=extent,
            levels=ctr_level(hist, [0.68, 0.95]),
            zorder=5, cma=plt.cm.autumn_r)
        plt.show()
    """
    with open(histogram_path, 'r') as hist_file:
        length = 0
        for line in hist_file:
            if line:
                if line.find("# x_centers") != -1:
                    x_centers = [float(elem) for elem in
                                 hist_file.next().split(",")]
                    length = len(x_centers)
                elif line.find("# y_centers") != -1:
                    y_centers = [float(elem) for elem in
                                 hist_file.next().split(",")]
                elif line.find("# Extent") != -1:
                    extent = [float(elem) for elem in
                              hist_file.next().split(",")]
                elif line.find("# Histogram") != -1:
                    hist = []
                    for index in range(length):
                        hist.append([float(elem) for elem in
                                     hist_file.next().split(",")])
    x_centers = np.array(x_centers)
    y_centers = np.array(y_centers)
    extent = np.array(extent)
    hist = np.array(hist)

    return x_centers, y_centers, extent, hist

def store_contour_coordinates(file_name, name1, name2, contours, levels):

    with open(file_name, 'w') as plot_file:
        plot_file.write(
            '# contour for confidence level {0}\n'.format(levels[1]))
        for elem in contours.collections[0].get_paths():
            points = elem.vertices
            for k in range(np.shape(points)[0]):
                plot_file.write("%.8g\t %.8g\n" % (
                    points[k, 0], points[k, 1]))
                # stop to not include the inner contours
                if k != 0:
                    if all(points[k] == points[0]):
                        plot_file.write("\n")
                        break
        plot_file.write("\n\n")
        plot_file.write(
            '# contour for confidence level {0}\n'.format(levels[0]))
        for elem in contours.collections[1].get_paths():
            points = elem.vertices
            for k in range(np.shape(points)[0]):
                plot_file.write("%.8g\t %.8g\n" % (
                    points[k, 0], points[k, 1]))
                if k != 0:
                    if all(points[k] == points[0]):
                        plot_file.write("\n")
                        break
        plot_file.write("\n\n")
