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
#include <bob/python/ndarray.h>
#include <bob/trainer/WienerTrainer.h>
#include <bob/machine/WienerMachine.h>
#include <boost/shared_ptr.hpp>

using namespace boost::python;

void py_train1(bob::trainer::WienerTrainer& t, 
  bob::machine::WienerMachine& m, bob::python::const_ndarray data)
{
  const blitz::Array<double,3> data_ = data.bz<double,3>();
  t.train(m, data_);
}

object py_train2(bob::trainer::WienerTrainer& t, 
  bob::python::const_ndarray data)
{
  const blitz::Array<double,3> data_ = data.bz<double,3>();
  const int height = data_.extent(1);
  const int width = data_.extent(2);
  bob::machine::WienerMachine m(height, width, 0.);
  t.train(m, data_);
  return object(m);
}

void bind_trainer_wiener() {

  class_<bob::trainer::WienerTrainer, boost::shared_ptr<bob::trainer::WienerTrainer> >("WienerTrainer", "Trains a WienerMachine on a given dataset.\nReference:\n'Computer Vision: Algorithms and Applications', Richard Szeliski\n(Part 3.4.3)", init<>("Initializes a new WienerTrainer."))
    .def(init<const bob::trainer::WienerTrainer&>(args("other"), "Copy constructs a WienerTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::WienerTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this WienerTrainer with the 'other' one to be approximately the same.")
    .def("train", &py_train1, (arg("self"), arg("machine"), arg("data")), "Trains the provided WienerMachine with the given dataset.")
    .def("train", &py_train2, (arg("self"), arg("data")), "Trains a WienerMachine using the given dataset to perform the filtering. This method returns the trained WienerMachine.")
    ;

}
