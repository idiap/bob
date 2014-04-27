#!/bin/bash
# Andre Anjos <andre.anjos@idiap.ch>
# Sun 27 Apr 19:19:41 2014 CEST
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

export LD_LIBRARY_PATH=$2;
string=`$1 -version 2>&1 | grep -i "$1 version"`;
python -c "print('${string}'.split(' ')[2].strip(','))"
