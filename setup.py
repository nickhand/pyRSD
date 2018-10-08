#! /usr/bin/env python
import setuptools # necessary for install_requires
from setuptools import find_packages

from distutils.core import Command
from numpy.distutils.core import Extension
from numpy.distutils.command.build_clib import build_clib
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.command.sdist import sdist
from numpy.distutils.command.build import build
from distutils.command.clean import clean

import subprocess
from glob import glob
import os
import numpy
import shutil

descr = """pyRSD

pyRSD is a Python package for computing the theoretical predictions of the
redshift-space power spectrum of galaxies. The package also includes
functionality for fitting data measurements and finding the optimal model
parameters, using both MCMC and nonlinear optimization techniques.
"""

DISTNAME         = 'pyRSD'
DESCRIPTION      = 'Accurate predictions for the clustering of galaxies in redshift-space in Python'
LONG_DESCRIPTION = descr
MAINTAINER       = 'Nick Hand'
MAINTAINER_EMAIL = 'nicholas.adam.hand@gmail.com'

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))

# determine if swig will need to be called on GCL extension
swig_needed = not all(os.path.isfile(f) for f in ['pyRSD/gcl.py', 'pyRSD/gcl_wrap.cpp'])

# the CLASS version to install
CLASS_VERSION = "2.6.1"

def check_swig_version():
    """
    Check the version of swig, >= 3.0 is required

    Notes
    -----
    *   swig is only needed for developers installing from the source directory,
        with ``python setup.py install``
    *   the swig-generated files are included by default in the pypi distribution,
        so the swig dependency is not needed
    """
    import subprocess, re
    try:
        output = subprocess.check_output(["swig", "-version"])
    except OSError:
        raise ValueError(("`swig` not found on PATH -- either install `swig` or use "
                            "``conda install -c nickhan pyrsd`` (recommended)"))

    try:
        version = re.findall("SWIG Version [0-9].[0-9].[0-9]", output)[0].split()[-1]
    except:
        return

    # need >= 3.0
    if version < "3.0":
        raise ValueError(("the version of `swig` on PATH must greater or equal to 3.0; "
                         "recommended installation without swig is ``conda install -c nickhan pyrsd``"))

def build_CLASS(prefix):
    """
    Function to download CLASS from github and and build the library
    """
    # latest class version and download link
    args = (package_basedir, CLASS_VERSION, prefix, "/opt/class/willfail")
    command = 'sh %s/depends/install_class.sh %s %s %s' %args

    ret = os.system(command)
    if ret != 0:
        raise ValueError("could not build CLASS v%s" %CLASS_VERSION)

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

class build_external_clib(build_clib):
    """
    Custom command to build CLASS first, and then GCL library
    """
    def finalize_options(self):

        build_clib.finalize_options(self)

        # create the CLASS build directory and save the include path
        self.class_build_dir = self.build_temp
        self.include_dirs.insert(0, os.path.join(self.class_build_dir, 'include'))

    def build_libraries(self, libraries):

        # build CLASS first
        build_CLASS(self.class_build_dir)

        # update the link objects with CLASS library
        link_objects = ['libclass.a']
        link_objects = list(glob(os.path.join(self.class_build_dir, '*', 'libclass.a')))

        self.compiler.set_link_objects(link_objects)
        self.compiler.library_dirs.insert(0, os.path.join(self.class_build_dir, 'lib'))

        for (library, build_info) in libraries:

            # check swig version
            if library == "gcl" and swig_needed:
                check_swig_version()

            # update include dirs
            self.include_dirs += build_info.get('include_dirs', [])

        build_clib.build_libraries(self, libraries)

class custom_build_ext(build_ext):
    """
    Custom extension building to grab include directories
    from the ``build_clib`` command
    """
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.include_dirs.append(numpy.get_include())

    def run(self):
        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
            build_clib = self.get_finalized_command('build_clib')
            self.include_dirs += build_clib.include_dirs
            self.library_dirs += build_clib.compiler.library_dirs

        # copy data files from temp to pyRSD package directory
        shutil.rmtree(os.path.join(self.build_lib, 'pyRSD', 'data', 'class'), ignore_errors=True)
        shutil.copytree(os.path.join(self.build_temp, 'data'), os.path.join(self.build_lib, 'pyRSD', 'data', 'class'))

        build_ext.run(self)

class custom_sdist(sdist):

    def run(self):
        from six.moves.urllib import request

        # download CLASS
        tarball_link = "https://github.com/lesgourg/class_public/archive/v%s.tar.gz" %CLASS_VERSION
        tarball_local = os.path.join('depends', 'class-v%s.tar.gz' %CLASS_VERSION)
        request.urlretrieve(tarball_link, tarball_local)

        # run the default
        sdist.run(self)


class custom_clean(clean):

    def run(self):

        # run the built-in clean
        clean.run(self)

        # remove the CLASS tmp directories
        os.system("rm -rf depends/tmp*")
        os.system("rm -f pyRSD/*.so*")

        # remove build directory
        if os.path.exists('build'):
            shutil.rmtree('build')

def libgcl_config():

    # c++ GCL sources and fortran FFTLog sources
    gcl_sources = list(glob("pyRSD/_gcl/cpp/*cpp"))

    # GCL library extension
    gcl_info = {}
    gcl_info['sources'] =  gcl_sources
    gcl_info['include_dirs'] = ['pyRSD/_gcl/include', '/usr/local/include']
    gcl_info['language'] = 'c++'
    gcl_info['extra_compiler_args'] = ["-O2", '-std=c++11']
    return ('gcl', gcl_info)

def libfftlog_config():
    info = {}
    info['sources'] =  list(glob("pyRSD/_gcl/extern/fftlog/*f"))
    return ('fftlog', info)

def libemu_config():
    info = {}

    # determine gsl path
    try:
        gsl_prefix = subprocess.check_output('gsl-config --prefix', shell=True).decode('utf-8').strip()
    except:
        raise ValueError("GSL is not installed!")

    info['sources'] =  list(glob("pyRSD/_gcl/extern/FrankenEmu/src/*.c"))
    info['include_dirs'] = ['pyRSD/_gcl/extern/FrankenEmu/include', os.path.join(gsl_prefix, 'include')]
    info['library_dirs'] = [os.path.join(gsl_prefix, 'lib')]
    info['extra_compiler_args'] = ["-O2", "-Wno-missing-braces"]
    return ('emu', info)


def gcl_extension_config():

    # the configuration for GCL python extension
    config = {}
    config['name'] = 'pyRSD._gcl'
    config['extra_link_args'] = ['-g', '-fPIC']
    config['extra_compile_args'] = []
    config['libraries'] = ['gcl', 'fftlog', 'emu', 'class', 'gsl', 'gslcblas', 'gfortran']

    # determine if swig needs to be called
    if not swig_needed:
        config['sources'] = ['pyRSD/gcl_wrap.cpp']
    else:
        config['sources'] = ['pyRSD/gcl.i']
        config['depends'] = ['pyRSD/_gcl/python/*.i']
        config['swig_opts'] = ['-c++']

    return config

# the dependencies
with open('requirements.txt', 'r') as fh:
    dependencies = [l.strip() for l in fh]

# extra dependencies
extras = {}
with open('requirements-extras.txt', 'r') as fh:
    extras['extras'] = [l.strip() for l in fh][1:]
    extras['full'] = extras['extras']

with open('requirements-tests.txt', 'r') as fh:
    extras['tests'] = [l.strip() for l in fh][1:]
    extras['test'] = extras['tests']

pkg_data = ['data/dark_matter/pkmu_P*',
            'data/dark_matter/hzpt*',
            'data/galaxy/full/*',
            'data/galaxy/2-halo/*',
            'data/params/*',
            'data/simulation_fits/*',
            'data/examples/*',
            'tests/baseline/*png']

if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(name=DISTNAME,
          version=find_version("pyRSD/version.py"),
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          license='GPL3',
          zip_safe=False,
          ext_modules = [Extension(**gcl_extension_config())],
          libraries=[libfftlog_config(), libemu_config(), libgcl_config()],
          cmdclass = {
              'sdist': custom_sdist,
              'build_clib': build_external_clib,
              'build_ext': custom_build_ext,
              'clean': custom_clean
          },
          packages=find_packages(),
          install_requires=dependencies,
          extras_require=extras,
          package_data={'pyRSD': pkg_data},
          entry_points={'console_scripts' :
                      ['rsdfit = pyRSD.rsdfit.rsdfit:main',
                       'pyrsd-quickstart = pyRSD.quickstart.core:main']}
    )
