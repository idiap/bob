/**
 * @file python/core/src/main_tuple.cc
 * @date Mon Jul 25 14:02:42 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief boost::random bindings
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

void bind_core_tuple();

BOOST_PYTHON_MODULE(libpytorch_core_tuple) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch core classes and sub-classes for accessing boost::tuple objects from python";
  bind_core_tuple();
}
