/**
 * @file python/sp/src/main.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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

void bind_sp_version();
void bind_sp_spcore();
void bind_sp_convolution();
void bind_sp_extrapolate();
void bind_sp_fft_dct();

BOOST_PYTHON_MODULE(libpybob_sp) {

  bob::python::setup_python("bob signal processing classes and sub-classes");

  bind_sp_version();
  bind_sp_spcore();
  bind_sp_convolution();
  bind_sp_extrapolate();
  bind_sp_fft_dct();
}
