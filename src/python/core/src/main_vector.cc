/**
 * @file python/core/src/main_vector.cc
 * @date Mon Apr 18 16:45:57 2011 +0200
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

#include "core/python/ndarray.h"

void bind_core_vectors();
void bind_core_arrayvectors_1();
void bind_core_arrayvectors_2();
void bind_core_arrayvectors_3();
void bind_core_arrayvectors_4();

BOOST_PYTHON_MODULE(libpytorch_core_vector) {

  Torch::python::setup_python("Torch core classes and sub-classes for std::vector manipulation from python");

  bind_core_vectors();
  bind_core_arrayvectors_1();
  bind_core_arrayvectors_2();
  bind_core_arrayvectors_3();
  bind_core_arrayvectors_4();
}
