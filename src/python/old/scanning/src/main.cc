/**
 * @file python/old/scanning/src/main.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
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

using namespace boost::python;

void bind_scanning_pattern();
void bind_scanning_ipgeomnorm();
void bind_scanning_scanner();
void bind_scanning_facefinder();
void bind_scanning_gtfile();
void bind_scanning_explorer();

BOOST_PYTHON_MODULE(libpytorch_scanning) {
  scope().attr("__doc__") = "Torch classes and sub-classes for scanning images";
  bind_scanning_pattern();
  bind_scanning_ipgeomnorm();
  bind_scanning_scanner();
  bind_scanning_facefinder();
  bind_scanning_gtfile();
  bind_scanning_explorer();
}
