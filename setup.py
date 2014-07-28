#! /usr/bin/env python

descr = """pyRSD

Algorithms to compute the redshift space matter power spectra using 
perturbation theory and the redshift space distortion (RSD) model based
on a distribution function velocity moments approach
"""

DISTNAME            = 'pyRSD'
DESCRIPTION         = 'Redshift space power spectra in python'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Nick Hand'
MAINTAINER_EMAIL    = 'nicholas.adam.hand@gmail.com'
VERSION             = '0.10dev'
PYTHON_VERSION      = (2, 5)
DEPENDENCIES        = {
                        'numpy': (1, 5),
                        'Cython': (0, 6),
                        'cython_gsl' : ()
                      }


import os
import sys
import re
import setuptools
from numpy.distutils.core import setup
from distutils.command.build_py import build_py


def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(
            ignore_setup_xxx_py=True,
            assume_default_configuration=True,
            delegate_options_to_subpackages=True,
            quiet=True)

    config.add_subpackage('pyRSD')
    config.add_data_dir('pyRSD/data')

    return config

#-------------------------------------------------------------------------------
def write_version_py(filename='pyRSD/version.py'):
    template = """# THIS FILE IS GENERATED FROM THE PYRSD SETUP.PY
version='%s'
"""

    vfile = open(os.path.join(os.path.dirname(__file__),
                              filename), 'w')

    try:
        vfile.write(template % VERSION)
    finally:
        vfile.close()

#-------------------------------------------------------------------------------
def get_package_version(package):
    version = []
    for version_attr in ('version', 'VERSION', '__version__'):
        if hasattr(package, version_attr) \
                and isinstance(getattr(package, version_attr), str):
            version_info = getattr(package, version_attr, '')
            for part in re.split('\D+', version_info):
                try:
                    version.append(int(part))
                except ValueError:
                    pass
    return tuple(version)

#-------------------------------------------------------------------------------
def check_requirements():
    if sys.version_info < PYTHON_VERSION:
        raise SystemExit('You need Python version %d.%d or later.' \
                         % PYTHON_VERSION)

    for package_name, min_version in DEPENDENCIES.items():
        dep_error = False
        try:
            package = __import__(package_name)
        except ImportError:
            dep_error = True
        else:
            package_version = get_package_version(package)
            if min_version > package_version:
                dep_error = True

        if dep_error:
            raise ImportError('You need `%s` version %d.%d or later.' \
                              % ((package_name, ) + min_version))


#-------------------------------------------------------------------------------
if __name__ == "__main__":

    check_requirements()

    write_version_py()

    # use gcc not clang
    os.environ['CC'] = 'gcc'
    
    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,

        configuration=configuration,

        packages=setuptools.find_packages(),
        include_package_data=True,
        zip_safe=False, # the package can run out of an .egg file

        cmdclass={'build_py': build_py},
    )