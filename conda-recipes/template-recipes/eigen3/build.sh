#!/bin/sh

mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DCMAKE_BUILD_TYPE=Release ..
make install