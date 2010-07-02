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

case $1 in
  (-h|-?|--help)
    echo "usage: `basename $0` [<example>|all-examples]"
    echo "  By calling me with no arguments, I will build a full release of"
    echo "Torch. If you give either the name of an example (all lowercase!)"
    echo "or the special keyword 'all-examples', I'll also build those."
    echo "Please note that by executing this symlink you will be building a"
    echo "version of the system in '${bname}' mode."
    exit 2;;
esac

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

# Here you can choose if you want to have dynamic or static linking for
# examples. All torch and auxiliary libraries will be compiled in both flavors
# though. Static linkage is beneficial if you need to export the programs to
# run in arbitrary environment where you don't expect requirements (libraries)
# to be properly installed and compatible. Dynamic linkage is prefered if you
# are running in Idiap or on know platforms that have all required libraries
# installed. In this case, program loading may become faster.

linkage=dynamic
#linkage=static

# Print out some stuff so the user knows what we are doing
echo "Build type: ${build_type}"
echo "Executable linkage: ${linkage}"
echo "CPU count: ${cpu_count}"
echo "Platform: ${platform}"
echo "Prefix: ${prefix}"
echo "Build directory: ${build_dir}"
echo "Install directory: ${install_dir}"
echo "Includes directory: ${include_dir}"

# Here we create a directory for the build and call cmake followed by a make
# all and install.
[ ! -d ${build_dir} ] && mkdir -p ${build_dir}
root_dir=`pwd`
cd ${build_dir}
cmake -DCMAKE_BUILD_TYPE=${build_type} -DPLATFORM=${platform} -DINSTALL_DIR=${install_dir} -DINCLUDE_DIR=${include_dir} -DDIST_ROOT=${prefix} -DCPU_COUNT=${cpu_count} -DTORCH_LINKAGE=${linkage} ${prefix}
make -j${cpu_count} all
make -j${cpu_count} install 

# Observation on "make -jX": Please note that the subproject ffmpeg does not
# compile on the first try, in parallel. A second call to "make -j" sorts the
# problem out. As an option, you can disable the parallel compilation of ffmpeg
# by removing the cmake option "-DCPU_COUNT=${cpu_count}". 

# If the user has given a clue on what to build (examples), we do it
if [ $# -ge 1 ]; then
  #make VERBOSE=1 install-$1 #use this to debug the examples building
  make -j${cpu_count} install-$1
fi

