#! /usr/bin/env python
from distutils.core import setup, Command
from distutils.command.install import install as DistutilsInstall
import os

# my own install class to make pygcl before installing
class MyInstall(DistutilsInstall):
    
    def run(self):
        ans = os.system("cd pyRSD/gcl; make gcl;")
        if (ans > 0): raise ValueError("Failed to make `pygcl` module; installation cannot continue")
        DistutilsInstall.run(self)

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

    
setup(  cmdclass={'install': MyInstall, 'clean' : MyClean},
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,
        packages=['pyRSD', 'pyRSD.data', 'pyRSD.rsd'],
        scripts=['pyRSD/scripts/' + script for script in os.listdir('pyRSD/scripts')]
    )
