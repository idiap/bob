/**
 * @file visioner/python/main.cc
 * @date Thu Jul 21 13:13:06 2011 +0200
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

#include "bob/core/python/ndarray.h"

void bind_visioner_version();
void bind_visioner_localize();
void bind_visioner_train();

BOOST_PYTHON_MODULE(_visioner) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("Face detection, keypoint localization and pose estimation using Boosting and LBP-like features (Visioner)");

  bind_visioner_version();
  bind_visioner_localize();
  bind_visioner_train();
}
