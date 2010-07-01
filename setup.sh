#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 30 Jun 2010 16:22:58 CEST

# To setup in debug mode, just do
# source ./setup.sh -d

source ~aanjos/sw/setup.sh #this will get you cmake-2.8

build_type=release
if [ $# -ge 1 ]; then
  case $1 in
    (-d|--debug)
      build_type=debug;;
    (*)
      echo "usage: source ./setup.sh -[hd]"
      echo " -h|--help  Prints this help message"
      echo " -d|--debug Sets up in debug mode"
      echo "NOTE: You have to be on the directory this file is to source it."
      return;;
  esac
fi

platform=`uname -s`-`uname -m`-${build_type}
platform=`echo ${platform} | tr 'A-Z' 'a-z'` #lowercase transformation
prefix=`pwd`
build_dir=${prefix}/build/${platform}
install_dir=${prefix}/install/${platform}
include_dir=${prefix}/install/include

export LD_LIBRARY_PATH=${install_dir}/lib:${LD_LIBRARY_PATH}
export PATH=${install_dir}/bin:${PATH}
