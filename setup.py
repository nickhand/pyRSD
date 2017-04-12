#! /usr/bin/env python
from setuptools import setup, Command, find_packages
from setuptools.command.install import install as Install
from setuptools.command.develop import develop as Develop
import os
import shutil

CLASS_VERSION="2.5.0"

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))

def build_CLASS():
    """
    Function to dowwnload CLASS from github and and build the library
    """
    # latest class version and download link    
    
    prefix = os.path.join('depends', 'build', 'class')
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

def make_pygcl(path, dist_name):
    """
    Make the ``pygcl`` package
    """
    # check that setup.config exists
    if not os.path.isfile('setup.config'):
        msg = "Installation requires 'setup.config' file in base directory."
        if os.path.isfile('setup.config.example'):
            msg += " Try copying over 'setup.config.example' with the appropriate edits."
        raise OSError(msg)
    
    # build CLASS first
    build_CLASS()
    
    # make pygcl
    install_path_args = path, dist_name
    data_dir = "{}/{}/data/params".format(*install_path_args)
    ans = os.system("cd pyRSD/gcl; make gcl")
    if (ans > 0): raise ValueError("Failed to make `pygcl` module; installation cannot continue.")
    
    # copy over the CLASS data
    build_dir = os.path.join(package_basedir, 'depends', 'build', 'class')
    shutil.rmtree(os.path.join(path, dist_name, 'data', 'class'), ignore_errors=True)
    shutil.copytree(os.path.join(build_dir, 'data'), os.path.join(path, dist_name, 'data', 'class'))
    
class MyInstall(Install):
    
    def run(self):
        make_pygcl(self.install_libbase, self.config_vars['dist_name'])
        Install.run(self)
        
class MyDevelop(Develop):
    
    def run(self):        
        make_pygcl(self.egg_path, self.config_vars['dist_name'])
        Develop.run(self)

#-------------------------------------------------------------------------------        
# my own command to do a clean of all necessary files      
class MyClean(Command):
    description = "custom clean command that removes build directories and runs make clean on pygcl"
    user_options = []
    def initialize_options(self):
        self.cwd = None
    def finalize_options(self):
        self.cwd = os.getcwd()
    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        
        shutil.rmtree('build', ignore_errors=True)
        shutil.rmtree(os.path.join('depends', 'build'), ignore_errors=True)
        os.system("rm -rf depends/tmp*")
        os.system('cd pyRSD/gcl; make clean;')

#-------------------------------------------------------------------------------
descr = """pyRSD

Algorithms to compute the redshift space matter power spectra using 
perturbation theory and the redshift space distortion (RSD) model based
on a distribution function velocity moments approach
"""

DISTNAME         = 'pyRSD'
DESCRIPTION      = 'Anisotropic RSD Fourier space modeling in Python'
LONG_DESCRIPTION = descr
MAINTAINER       = 'Nick Hand'
MAINTAINER_EMAIL = 'nicholas.adam.hand@gmail.com'

# the dependencies
with open('requirements.txt', 'r') as fh:
    dependencies = [l.strip() for l in fh]

# extra dependencies
extras = {}
with open('requirements-extras.txt', 'r') as fh:
    extras['extras'] = [l.strip() for l in fh][1:]
    extras['full'] = extras['extras'] 
    
pkg_data = ['data/dark_matter/pkmu_P*', 'data/galaxy/full/*', 'data/galaxy/2-halo/*', 
            'data/params/*', 'data/simulation_fits/*', 'gcl/python/_gcl*.so']

#-------------------------------------------------------------------------------   
setup(  cmdclass={'install':MyInstall, 'develop':MyDevelop, 'clean':MyClean},
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=find_version("pyRSD/version.py"),
        packages=find_packages(),
        license="GPL3",
        install_requires=dependencies,
        extras_require=extras,
        package_data={'pyRSD': pkg_data},
        entry_points={'console_scripts' : ['rsdfit = pyRSD.rsdfit.rsdfit:main', 'rsdfit-batch = pyRSD.rsdfit.rsdfit_batch:main']}
    )
