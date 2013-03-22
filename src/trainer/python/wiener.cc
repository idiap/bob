/**
 * @file trainer/python/wiener.cc
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to WienerTrainer
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

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <bob/trainer/WienerTrainer.h>

using namespace boost::python;

boost::shared_ptr<bob::machine::WienerMachine> 
wiener_train1(const bob::trainer::WienerTrainer& t, const blitz::Array<double,3>& data) {
  boost::shared_ptr<bob::machine::WienerMachine> m;
  t.train(*m, data);
  return m;
}

void wiener_train2(const bob::trainer::WienerTrainer& t, bob::machine::WienerMachine& m,
    const blitz::Array<double,3>& data) {
  t.train(m, data);
}

void bind_trainer_wiener() {

  class_<bob::trainer::WienerTrainer>("WienerTrainer", "Sets a WienerMachine and train it on a given dataset.\nReference:\n'Computer Vision: Algorithms and Applications', Richard Szeliski\n(Part 3.4.3)", init<>("Initializes a new WienerTrainer."))
    .def(init<const bob::trainer::WienerTrainer&>(args("other")))
    .def("train", &wiener_train1, (arg("self"), arg("data")), "Trains a WienerMachine using the given dataset to perform the filtering. This method returns the trained WienerMachine.")
    .def("train", &wiener_train2, (arg("self"), arg("machine"), arg("data")), "Trains the provided WienerMachine with the given dataset.")
    ;

}
