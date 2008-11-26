#!/bin/csh -f

mkdir -p build_`uname -s`_`uname -m`
cd build_`uname -s`_`uname -m`
cmake ../src
make
cd ..
cd extras/jpeg
./makejpeg.csh
cd ../..

