# Andre Anjos <andre.anjos@idiap.ch>
# Sat  1 Sep 22:04:04 2012 CEST
# <patch> Flavio Tarsetti <Tarsetti.Flavio@gmail.com>
# Mon  1 Jul 10:00:02 2013 CEST
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

# Support settings for external code built on Bob-cxx

get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(${SELF_DIR}/bob-targets.cmake)
get_filename_component(bob_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}../../../include" ABSOLUTE)
get_filename_component(bob_LIBRARY_DIRS "${CMAKE_CURRENT_LIST_DIR}../../../lib" ABSOLUTE)
