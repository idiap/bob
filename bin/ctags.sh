#!/bin/bash 
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 02 Aug 2012 11:52:36 CEST
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

install_dir=$(readlink -f $(dirname $(dirname $0)));
files=$(find ${install_dir}/src ${install_dir}/python -name '*.h' -or -name '*.cc' -or -name '*.c' -or -name '*.cxx' -or -name '*.cpp' -or -name '*.C' -or -name '*.CC')
ctags --output=${install_dir}/tags ${files}
