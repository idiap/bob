/**
 * @file python/sp/src/version.cc
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 29 Nov 2011 11:02:26 CET
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include <fftw3.h>

using namespace boost::python;

/**
 * FFTW3 support
 */
static tuple fftw3_version() {
  return make_tuple((const char*)fftw_version, 
                    (const char*)fftw_cc, 
                    (const char*)fftw_codelet_optim);
}

void bind_sp_version() {
  dict vdict;
  vdict["FFTW"] = fftw3_version();
  scope().attr("version") = vdict;
}
