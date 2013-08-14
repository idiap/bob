#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 29 Nov 2011 13:36:43 CET
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

export LD_LIBRARY_PATH=$2;
string=`$1 -version 2>&1 | grep -i 'ffmpeg version'`;
python -c "print('${string}'.split(' ')[2].strip(','))"
