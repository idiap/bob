#!/bin/bash
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 03 Jun 2014 11:45:19 CEST

# Builds a virtualenv with all packages in layer2, downloaded from
# some webserver (could be PyPI)

# Requirement: virtualenv *should* be installed with the base framework
# pre-compiled (e.g. NumPy, SciPy, Matplotlib, etc.), to avoid unnecessary time
# spent on compiling trivialities.

# Usage info
show_help() {
cat << EOF
Usage: ${0##*/} [-r requirements] [-x externals] [-f find-links]
                [-d folder] [-o debug/release] [-p threads] [-P python] [-h] [-i] [-u] [-F]

Creates a virtual environment, basing itself on the given externals area.
Pre-populates the virtual environment with the package list provided, if
one is given.

-h       display this help and exit
-r FILE  use the following requirements file to extra packages to install
-x EXE   use the virtualenv executable pointed by this flag instead of the
         default one
-f URL   find packages for installation on this URL
-d DIR   create the new virtual environment on this directory
-o debug/release  compiles the Source code in debug or release mode
         (default: release)
-p INT   If specified, Bob will be compiled with the given number of parallel
         threads
-P EXE   If specified, uses the given python to initialize the virtualenv
-i       ignore installation errors and keep going
-u       upgrade the given requirements
-F       forces the installation of a certain package
EOF
}

requirements=""
virtualenv="virtualenv"
find_links=""
directory=""
optimize="release"
parallel=""
python=""
upgrade=""
ignore=0
force=""

while getopts "hr:x:f:d:o:p:P:iuF" opt; do
  case "$opt" in
    h)
      show_help
      exit 0
      ;;
    r)  requirements=$OPTARG
      ;;
    x)  virtualenv=$OPTARG
      ;;
    f)  find_links=$OPTARG
      ;;
    d)  directory=$OPTARG
      ;;
    o)  optimize=$OPTARG
      ;;
    p)  parallel=$OPTARG
      ;;
    P)  python="-p $OPTARG"
      ;;
    i)  ignore=1
      ;;
    u)  upgrade="--upgrade"
      ;;
    F)  force="--force-reinstall"
      ;;

    '?')
      show_help >&2
      exit 1
      ;;
  esac
done

# Prepares the new virtual environment
tmpdir=""
if [ -z "${directory}" ]; then
  tmpdir=`mktemp --tmpdir -d ${0##*/}.XXXXXX`
  directory=${tmpdir}
  echo "Installing on temporary directory \`${directory}'..."
fi

if [ ! -d ${directory}/bin ]; then
  echo "Initializing virtual environment at \`${directory}'..."
  ${virtualenv} --system-site-packages ${python} ${directory}
  ${directory}/bin/pip install --upgrade setuptools pip
else
  echo "Skipped virtual environment initialization at \`${directory}'"
fi

if [ "${optimize}" == "release" ]; then
  echo "Setting RELEASE flags"
  export CFLAGS="-O3 -g0 -mtune=generic"
  export CXXFLAGS="-O3 -g0 -mtune=generic"
else
  echo "Setting DEBUG flags"
  export CFLAGS='-O0 -g -DBOB_DEBUG -DBZ_DEBUG'
  export CXXFLAGS='-O0 -g -DBOB_DEBUG -DBZ_DEBUG'
fi

if [ -n "${parallel}" ]; then
  echo "Using ${parallel} parallel threads for compilation"
  export BOB_BUILD_PARALLEL=${parallel}
fi

# Installs all components listed the requirements file
if [ -n "${requirements}" ]; then
  pip_opt="--verbose --pre --egg"

  if [ -n "${find_links}" ]; then
    pip_opt="--find-links=${find_links} ${pip_opt}"
  fi

  echo "Installing all requirements in \`${requirements}'..."
  for req in `cat ${requirements}`; do
    echo "Installing \`${req}'..."
    ${directory}/bin/pip install ${upgrade} ${force} ${pip_opt} "${req}"
    status=$?
    if [ ${ignore} == 0 -a ${status} != 0 ]; then
      echo "Installation of package ${req} failed; aborting"
    exit ${status}; fi
  done
fi

# Removes temporary directory if one was created
if [ -n "${tmpdir}" ]; then
  echo "Removing temporary directory \`${directory}'..."
  rm -rf ${tmpdir}
fi
