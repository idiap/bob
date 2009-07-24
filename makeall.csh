#!/bin/csh -f

mkdir -p build_`uname -s`_`uname -m`
cd build_`uname -s`_`uname -m`
cmake ../src
make
cd ..

cd extras/jpeg
./makejpeg.csh
cd ../..

cd extras/oourafft
./makeoourafft.csh
cd ../..

cd extras/lbfgs
./makelbfgs.csh
cd ../..


