#!/bin/bash

# make class
CLASS_VERSION=2.5.0
wget https://github.com/lesgourg/class_public/archive/v$CLASS_VERSION.tar.gz -O ./class-v$CLASS_VERSION.tar.gz 
gzip -dc class-v$CLASS_VERSION.tar.gz | tar xf - -C .
pushd class_public-$CLASS_VERSION
make libclass.a
popd

cp $RECIPE_DIR/setup.config .
$PYTHON setup.py install 

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.