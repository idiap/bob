/**
 * @file measure/python/main.cc
 * @date Wed Apr 20 08:19:36 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/python/ndarray.h"

void bind_measure_error();

BOOST_PYTHON_MODULE(_measure) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob error measure classes and sub-classes");

  bind_measure_error();
}
