#!/usr/bin/env python

import os

from pyRSD._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from cython_gsl import get_cython_include_dir, get_libraries, get_library_dir
    
    config = Configuration('cosmology', parent_package, top_path)

    cython(['cosmo_tools.pyx'], working_path=base_path)
    cython(['growth.pyx'], working_path=base_path)
    cython(['power.pyx'], working_path=base_path)

    config.add_extension('cosmo_tools', sources=['cosmo_tools.c'],
                         include_dirs=[get_numpy_include_dirs()], 
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    config.add_extension('growth', sources=['growth.c', 'transfer.c', 'power_tools.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    config.add_extension('power', sources=['power.c', 'transfer.c', 'power_tools.c'],
                         include_dirs=[get_numpy_include_dirs(), get_cython_include_dir(), ''],
                         libraries=get_libraries(), library_dirs=[get_library_dir()],
                         extra_compile_args=['-O3', '-w'],
                         extra_link_args=['-g'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='pyRSD Developers',
          author='pyRSD Developers',
          description='cosmology',
          **(configuration(top_path='').todict())
          )