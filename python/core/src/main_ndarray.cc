/**
 * @file python/core/src/main_ndarray.cc
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

#include "core/python/ndarray.h"

void bind_core_bz_numpy();
void bind_core_ndarray_numpy();
void bind_core_array_tinyvector();
void bind_core_array_typeinfo();
//void bind_core_array_examples(); ///< examples
void bind_core_array_convert();

BOOST_PYTHON_MODULE(_cxx) {
  bob::python::setup_python("bob core classes and sub-classes for array manipulation");

  bind_core_bz_numpy();
  bind_core_ndarray_numpy();
  bind_core_array_tinyvector();
  bind_core_array_typeinfo();
  //bind_core_array_examples(); ///< examples
  bind_core_array_convert();
}
