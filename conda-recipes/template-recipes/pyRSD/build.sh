#!/bin/bash

if [ `uname` == Darwin ]; then
	export LIBS="-Wl,-rpath,$CONDA_PREFIX/lib"
fi

$PYTHON setup.py install 

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.