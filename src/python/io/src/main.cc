/**
 * @file python/io/src/main.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
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

void bind_io_exception();
void bind_io_file();
void bind_io_array();
void bind_io_arrayset();
void bind_io_hdf5();
void bind_io_hdf5_extras();
void bind_io_datetime();
void bind_io_video();

/**
void bind_io_binfile();
void bind_io_tensorfile();
**/

BOOST_PYTHON_MODULE(libpytorch_io) {

  Torch::python::setup_python("Torch classes and sub-classes for io access");

  bind_io_exception();
  bind_io_file();
  bind_io_array();
  bind_io_arrayset();
  bind_io_hdf5();
  bind_io_hdf5_extras();
  bind_io_datetime();
  bind_io_video();

  /**
  bind_io_binfile();
  bind_io_tensorfile();
  **/
}
