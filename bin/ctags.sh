#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 02 Aug 2012 11:52:36 CEST

install_dir=$(readlink -f $(dirname $(dirname $0)));
files=$(find ${install_dir}/cxx ${install_dir}/python -name '*.h' -or -name '*.cc' -or -name '*.c' -or -name '*.cxx' -or -name '*.cpp' -or -name '*.C' -or -name '*.CC')
ctags --output=${install_dir}/tags ${files}
