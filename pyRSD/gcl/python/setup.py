#!/usr/bin/env python
"""
 setup.py
 This will compile the python bindings for the GCL C++ library
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/27/2014
"""
import distutils
from distutils.core import setup, Extension
import numpy
import os
import shutil

# c++ so use g++
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

# clean previous build
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if (not name.endswith(".i") and (name.startswith("pygcl") or name.startswith("_pygcl"))):
            os.remove(os.path.join(root, name))
            
    for name in dirs:
        if (name == "build"):
            shutil.rmtree(name)

# setup the extension
pygcl_module = Extension('_pygcl',
                         sources=['pygcl.i'],
                         swig_opts=['-c++', '-Wall', '-I../src'], 
                         include_dirs=['../src', '../include', numpy.get_include()],
                         libraries=['gcl', 'gsl', 'gslcblas'],
                         extra_link_args=["-L..", "-g", "-fPIC"],
                         library_dirs=['/opt/local/lib'],
                         extra_compile_args=["-fopenmp", "-O4", "-ffast-math"]
                         )

# actually do the setup 
setup(name = 'pygcl', version = '0.1', ext_modules = [pygcl_module], py_modules = ["pygcl"])
       