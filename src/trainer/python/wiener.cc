/**
 * @file python/trainer/src/wiener.cc
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to WienerTrainer
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
#include <boost/shared_ptr.hpp>
#include "bob/trainer/WienerTrainer.h"

using namespace boost::python;
namespace io = bob::io;
namespace mach = bob::machine;
namespace train = bob::trainer;

boost::shared_ptr<mach::WienerMachine> wiener_train1 (const train::WienerTrainer& t, const io::Arrayset& data) {
  boost::shared_ptr<mach::WienerMachine> m;
  t.train(*m, data);
  return m;
}

void wiener_train2 (const train::WienerTrainer& t, mach::WienerMachine& m,
    const io::Arrayset& data) {
  t.train(m, data);
}

void bind_trainer_wiener() {

  class_<train::WienerTrainer>("WienerTrainer", "Sets a Wiener machine and train it on a given dataset.", init<>("Initializes a new Wiener Trainer."))
    .def("train", &wiener_train1, (arg("self"), arg("data")), "Trains a WienerMachine to perform the filtering. This method returns a tuple containing the resulting Wiener machine in a 2D array.")
    .def("train", &wiener_train2, (arg("self"), arg("machine"), arg("data")), "Trains the provided WienerMachine.")
    ;

}
