#!/bin/csh -f

mkdir -p build_`uname -s`_`uname -m`
cd build_`uname -s`_`uname -m`
cmake ..
make

