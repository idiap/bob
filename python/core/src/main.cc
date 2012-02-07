/**
 * @file python/core/src/main.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
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

using namespace boost::python;

void bind_core_version();
void bind_core_exception();
void bind_core_logging();
void bind_core_profiler();

BOOST_PYTHON_MODULE(libpybob_core) {
  docstring_options docopt; 
# if !defined(BOB_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "bob core classes and sub-classes";

  bind_core_version();
  bind_core_exception();
  bind_core_logging();
  bind_core_profiler();
}
