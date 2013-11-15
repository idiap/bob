/**
 * @file trainer/python/wiener.cc
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to WienerTrainer
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
  t.train(m, data.bz<double,3>());
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

  class_<bob::trainer::WienerTrainer, boost::shared_ptr<bob::trainer::WienerTrainer> >("WienerTrainer", "Trains a WienerMachine on a given dataset.\nReference:\n'Computer Vision: Algorithms and Applications', Richard Szeliski\n(Part 3.4.3)", init<>((arg("self")), "Initializes a new WienerTrainer."))
    .def(init<const bob::trainer::WienerTrainer&>((arg("self"), arg("other")), "Copy constructs a WienerTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::WienerTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this WienerTrainer with the 'other' one to be approximately the same.")
    .def("train", &py_train1, (arg("self"), arg("machine"), arg("data")), "Trains the provided WienerMachine with the given dataset.")
    .def("train", &py_train2, (arg("self"), arg("data")), "Trains a WienerMachine using the given dataset to perform the filtering. This method returns the trained WienerMachine.")
    ;

}
