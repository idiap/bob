/**
 * @file python/trainer/src/svm.cc
 * @date Sun  4 Mar 20:06:49 2012 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to libsvm (training bits)
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

#include <core/python/ndarray.h>
#include "trainer/SVMTrainer.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;
namespace mach = bob::machine;
namespace train = bob::trainer;

void bind_trainer_svm() {
}
