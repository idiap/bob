#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 29 Jun 2010 10:54:41 CEST

# This is a generic script to build a certain release configuration. Do NOT
# call this script directly. Create symlinks to it using the configuration type
# as script name and call the symlink instead.

bname=`basename $0 .sh`
if [ "${bname}" = "build" ]; then
  echo "usage: `dirname $0`/<symlink.sh>"
  echo "  Do NOT call this program directly, use the symlinks!"
  echo "  If none exists, create one pointing to this script. Example:"
  echo "  $ ln -s ./build.sh ./release.sh"
  echo "  Please note that the names of the symlinks should be valid CMAKE"
  echo "  build types in lowercase (e.g. debug.sh or release.sh)."
  exit 1
fi

# Here we compute all the relevant directories needed for our build and the
# variables that need to be taken under consideration while building and
# installing the projects.
build_type=`basename $0 .sh | tr 'a-z' 'A-Z'` #uppercase transformation
platform=`uname -s`-`uname -m`-${build_type}
platform=`echo ${platform} | tr 'A-Z' 'a-z'` #lowercase transformation
prefix=`pwd`
build_dir=${prefix}/build/${platform}
install_dir=${prefix}/install/${platform}
include_dir=${prefix}/install/include
if [ -r /proc/cpuinfo ]; then
  cpu_count=`cat /proc/cpuinfo | egrep -c '^processor'`
else
  cpu_count=2 #a good default for current number of procs in a machine
fi

# Print out some stuff so the user knows what we are doing
echo "Build type: ${build_type}"
echo "CPU count: ${cpu_count}"
echo "Platform: ${platform}"
echo "Prefix: ${prefix}"
echo "Build directory: ${build_dir}"
echo "Install directory: ${install_dir}"
echo "Includes directory: ${include_dir}"

# Here we create a directory for the build and call cmake followed by a make
# all and install.
[ ! -d ${build_dir} ] && mkdir -p ${build_dir}
cd ${build_dir}
cmake -DCMAKE_BUILD_TYPE=${build_type} -DPLATFORM=${platform} -DINSTALL_DIR=${install_dir} -DINCLUDE_DIR=${include_dir} -DCPU_COUNT=${cpu_count} ${prefix}
make all
make -j${cpu_count} install
