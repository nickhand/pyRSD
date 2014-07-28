#!/usr/bin/env python

import os

from pyRSD._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from cython_gsl import get_cython_include_dir, get_libraries, get_library_dir
    
    config = Configuration('rsd', parent_package, top_path)

    cython(['_kernels.pyx'], working_path=base_path)
    cython(['_integral_base.pyx'], working_path=base_path)
    cython(['_pt_integrals.pyx'], working_path=base_path)
    cython(['_fourier_integrals.pyx'], working_path=base_path)
    
    
    cython(['power_dm.pyx'], working_path=base_path)
    cython(['power_biased.pyx'], working_path=base_path)
    cython(['correlation.pyx'], working_path=base_path)

    config.add_extension('_integral_base', sources=['_integral_base.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir()],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w', '-fopenmp'],
                         extra_link_args=['-g', '-fopenmp'])
                         
    config.add_extension('_fourier_integrals', sources=['_fourier_integrals.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir()],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])
                         
    config.add_extension('_pt_integrals', sources=['_pt_integrals.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir()],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])


    config.add_extension('_kernels', sources=['_kernels.c'],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])
                         
    config.add_extension('power_dm', sources=['power_dm.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir()],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    config.add_extension('power_biased', sources=['power_biased.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])
                         
    config.add_extension('correlation', sources=['correlation.c'],
                         include_dirs=[get_numpy_include_dirs()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])                         

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='pyRSD Developers',
          author='pyRSD Developers',
          description='rsd',
          **(configuration(top_path='').todict())
          )