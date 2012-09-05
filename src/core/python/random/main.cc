/**
 * @file python/core/src/main_random.cc
 * @date Mon Jul 11 18:31:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief boost::random bindings
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

void bind_core_random();

BOOST_PYTHON_MODULE(_ext) {
  docstring_options docopt; 
# if !defined(BOB_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "bob core classes and sub-classes for accessing boost::random objects from python";
  bind_core_random();
}
