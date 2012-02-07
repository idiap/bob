/**
 * @file python/daq/src/version.cc
 * @date Mon 06 Feb 2012 15:21:59 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>
#include <boost/format.hpp>

#if defined(HAVE_OPENCV)
#include <cvver.h>
#endif

using namespace boost::python;

/**
 * OpenCV version
 */
static str opencv_version() {
#if defined(HAVE_OPENCV)
  return str(CV_VERSION);
#else
  return str("unavailable");
#endif
}

void bind_daq_version() {
  dict vdict;
  vdict["OpenCV"] = opencv_version();
  scope().attr("version") = vdict;
}
