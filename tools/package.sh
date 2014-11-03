#!/bin/bash
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 28 Aug 2012 11:34:25 CEST

if [ $# = 0 ]; then
  echo "usage: $0 destination-directory"
  exit 1
fi

basedir=`python -c "import os; print os.path.dirname(os.path.dirname(os.path.abspath('$0')))"`
dest=`python -c "import os; print os.path.abspath('$1')"`

if [ ! -d ${dest} ]; then
  echo "Creating destination directory ${dest}...";
  mkdir -v -p ${dest};
fi

# package all submodules
echo "Creating packages..."
git submodule foreach `pwd`/bin/python setup.py sdist --formats=zip
echo "Copying packages to ${dest}..."
git submodule foreach cp -va dist/*.zip ${dest}
