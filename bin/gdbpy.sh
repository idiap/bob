#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 20 Jul 2010 09:52:17 CEST

if [ $# = 0 ]; then
  echo "usage: `basename $0` <script.py>"
  exit 1
fi

gdb -silent --args `which python` $*
