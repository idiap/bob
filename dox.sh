#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 05 Aug 2010 15:51:05 CEST

output=build/dox
[ ! -d ${output} ] && mkdir -p ${output};
doxygen
cd ${output}/html
ln -s index.html main.html #for trac-doxygen plugin
