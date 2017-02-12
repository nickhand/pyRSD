#!/bin/sh
g++ -I${CONDA_PREFIX}/include -o test test.cc
./test