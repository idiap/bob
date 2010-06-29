#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 28 Jun 2010 17:16:56 CEST

# Builds all projects using cmake
dir=build
[ ! -d ${dir} ] && mkdir ${dir}
cd ${dir}
cmake ..
make all
make install
