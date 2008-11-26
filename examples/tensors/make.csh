#!/bin/csh -f

mkdir -p `uname -s`_`uname -m`
cd `uname -s`_`uname -m`
cmake -DTORCH_OS=`uname -s` -DTORCH_ARCH=`uname -m` ..
make

