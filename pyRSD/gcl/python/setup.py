#!/usr/bin/env python
"""
 setup.py
 This will compile the python bindings for the GCL C++ library
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/27/2014
"""
import distutils
from numpy.distutils.core import setup, Extension
import numpy
import os
import shutil

# clean previous build
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if (not name.endswith(".i") and (name.startswith("gcl") or name.startswith("_gcl"))):
            os.remove(os.path.join(root, name))
            
    for name in dirs:
        if (name == "build"):
            shutil.rmtree(name)

# setup the extension
gcl_module = Extension('_gcl',
                         sources=['gcl.i'],
                         swig_opts=['-c++', '-Wall'], 
                         include_dirs=[numpy.get_include()],
                         extra_link_args=["-L..", "-g", "-fPIC"],
                         extra_compile_args=["-fopenmp", "-O2"]
                         )

# actually do the setup 
setup(name = 'gcl', version = '0.1', ext_modules = [gcl_module], py_modules = ["gcl"])
       