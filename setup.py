#! /usr/bin/env python
from setuptools import setup, Command, find_packages
from setuptools.command.install import install as Install
import os

#-------------------------------------------------------------------------------
# my own install class to make pygcl before installing
class MyInstall(Install):
    
    def run(self):
        # check that setup.config exists
        if not os.path.isfile('setup.config'):
            msg = "Installation requires 'setup.config' file in base directory."
            if os.path.isfile('setup.config.example'):
                msg += " Try copying over 'setup.config.example' with the appropriate edits."
            raise OSError(msg)
        
        # make pygcl
        install_path_args = self.install_libbase, self.config_vars['dist_name']
        data_dir = "{}/{}/data/params".format(*install_path_args)
        ans = os.system("cd pyRSD/gcl; make gcl DATADIR=%s;" %data_dir)
        if (ans > 0): raise ValueError("Failed to make `pygcl` module; installation cannot continue.")
        
        # run the python setup
        Install.run(self)

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
        os.system('rm -rf ./build')
        os.system('cd pyRSD/gcl; make clean;')

#-------------------------------------------------------------------------------
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
VERSION             = '1.0'

#-------------------------------------------------------------------------------   
pkg_data = ['data/dark_matter/pkmu_P*', 'data/galaxy/full/*', 'data/galaxy/2-halo/*', 
            'data/params/*', 'data/simulation_fits/*', 'gcl/python/_gcl.so']
setup(  cmdclass={'install': MyInstall, 'clean' : MyClean},
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,
        packages=find_packages(),
        install_requires=[
            'pandas', 
            'scikit-learn',
            'numpy',
            'scipy'
        ],
        package_data={'pyRSD': pkg_data},
        entry_points={'console_scripts' : ['rsdfit = pyRSD.rsdfit.rsdfit:run']}
    )
