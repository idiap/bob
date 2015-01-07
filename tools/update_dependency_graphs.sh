#!/bin/bash
# Manuel Guenther <manuel.guenther@idiap.ch>
# Wed  7 Jan 19:25:56 CET 2015

if [ $# == 0 ]; then
  echo "usage: $0 directory [directory] [...]"
  echo "e.g. $0 layers/0/bob.io.base layers/1/bob.ip.base ..."
  echo "or: $0 bob"
  exit 1
fi

if [ "$1" == "all" ]; then
  # generate overall plot for all packages
  bin/bob_dependecy_graph.py -P requirements*.txt -W x.dot -v -X
else
  # generate dependency graph for each of the given directories
  while [ $# != 0 ]; do
    directory=$1
    shift
    if [ -d $directory ]; then
      package=`basename "$directory"`
      bin/bob_dependecy_graph.py -p $package -X -w $directory/dependencies.svg -v
    fi
  done
fi


