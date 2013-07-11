/**
 * @file trainer/python/shuffler.cc
 * @date Mon 24 Jun 17:29:12 2013 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings for the DataShuffler
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
#include <boost/python/stl_iterator.hpp>
#include <boost/make_shared.hpp>
#include <bob/python/ndarray.h>
#include <bob/trainer/DataShuffler.h>

using namespace boost::python;

static tuple call_shuffler1(bob::trainer::DataShuffler& s, size_t N) {
  blitz::Array<double,2> data(N, s.getDataWidth());
  blitz::Array<double,2> target(N, s.getTargetWidth());
  s(data, target);
  return make_tuple(data, target);
}

static tuple call_shuffler2(bob::trainer::DataShuffler& s, boost::mt19937& rng,
    size_t N) {
  blitz::Array<double,2> data(N, s.getDataWidth());
  blitz::Array<double,2> target(N, s.getTargetWidth());
  s(rng, data, target);
  return make_tuple(data, target);
}

static void call_shuffler3(bob::trainer::DataShuffler& s, boost::mt19937& rng, 
  bob::python::ndarray d, bob::python::ndarray t)
{
  blitz::Array<double,2> data_ = d.bz<double,2>();
  blitz::Array<double,2> target_ = t.bz<double,2>();
  s(rng, data_, target_);
}

static void call_shuffler4(bob::trainer::DataShuffler& s, 
  bob::python::ndarray d, bob::python::ndarray t)
{
  blitz::Array<double,2> data_ = d.bz<double,2>();
  blitz::Array<double,2> target_ = t.bz<double,2>();
  s(data_, target_);
}

static tuple stdnorm(bob::trainer::DataShuffler& s) {
  blitz::Array<double,1> mean(s.getDataWidth());
  blitz::Array<double,1> stddev(s.getDataWidth());
  s.getStdNorm(mean, stddev);
  return make_tuple(mean, stddev);
}

static boost::shared_ptr<bob::trainer::DataShuffler> shuffler_from_arrays
(object data, object target) {
  //data
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,2>());

  //target
  stl_input_iterator<bob::python::const_ndarray> vtarget(target), tend;
  std::vector<blitz::Array<double,1> > vtarget_ref;
  vtarget_ref.reserve(len(target));
  for (; vtarget != tend; ++vtarget) 
    vtarget_ref.push_back((*vtarget).bz<double,1>());

  return boost::make_shared<bob::trainer::DataShuffler>(vdata_ref, vtarget_ref);
}

static boost::shared_ptr<bob::trainer::DataShuffler> shuffler_from_arraysets
(object data, object target) {
  //data
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,2>());

  //target
  stl_input_iterator<bob::python::const_ndarray> vtarget(target), tend;
  std::vector<blitz::Array<double,1> > vtarget_ref;
  vtarget_ref.reserve(len(target));
  for (; vtarget != tend; ++vtarget) 
    vtarget_ref.push_back((*vtarget).bz<double,1>());

  return boost::make_shared<bob::trainer::DataShuffler>(vdata_ref, vtarget_ref);
}

#ifdef __LP64__
#  define SSIZE_T_FMT "%ld"
#else
#  define SSIZE_T_FMT "%d"
#endif

static boost::shared_ptr<bob::trainer::DataShuffler> shuffler_from_arrays_or_arraysets
(object data, object target) {

  //checks what can be extracted from the first element of the iterable
  if (len(data) == 0) {
    PYTHON_ERROR(RuntimeError, "Input data parameter should be an iterable with, at least, length 1");
  }

  if (len(data) != len(target)) {
    PYTHON_ERROR(RuntimeError, "Data and target lengths differ: len(data) = " SSIZE_T_FMT " and len(target) = " SSIZE_T_FMT, len(data), len(target));
  }

  //Let's test the first element.
  extract<blitz::Array<double,2> > check_set(data[0]);

  if (check_set.check()) { //good, those are arraysets
    return shuffler_from_arraysets(data, target); 
  }

  //we try arrays
  return shuffler_from_arrays(data, target);
}

void bind_trainer_shuffler() {
  class_<bob::trainer::DataShuffler, boost::shared_ptr<bob::trainer::DataShuffler> >("DataShuffler", "A data shuffler is capable of being populated with data from one or multiple classes and matching target values. Once setup, the shuffer can randomly select a number of vectors and accompaning targets for the different classes, filling up user containers.\n\nData shufflers are particular useful for training neural networks.", no_init)
    .def("__init__", make_constructor(&shuffler_from_arrays_or_arraysets, default_call_policies(), (arg("data"), arg("target"))), "Initializes the shuffler with some data classes and corresponding targets. The data is read by considering examples are lying on different rows of the input data if it is composed of a list of NumPy ndarrays or copied internally if it is composed of a list of io.Arraysets.")
    .def("stdnorm", &bob::trainer::DataShuffler::getStdNorm, (arg("self"), arg("mean"), arg("stddev")), "Calculates and returns mean and standard deviation from the input data.")
    .def("stdnorm", &stdnorm, (arg("self")), "Calculates and returns mean and standard deviation from the input data.")
    .add_property("auto_stdnorm", &bob::trainer::DataShuffler::getAutoStdNorm, &bob::trainer::DataShuffler::setAutoStdNorm)
    .add_property("data_width", &bob::trainer::DataShuffler::getDataWidth)
    .add_property("target_width", &bob::trainer::DataShuffler::getTargetWidth)
    .def("__call__", &call_shuffler1, (arg("self"), arg("n")), "Populates the output matrices (data, target) by randomly selecting 'n' arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain 'n' rows and the number of columns that are dependent on input arraysets and target array widths.")
    .def("__call__", &call_shuffler2, (arg("self"), arg("rng"), arg("n")), "Populates the output matrices (data, target) by randomly selecting 'n' arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain 'n' rows and the number of columns that are dependent on input arraysets and target array widths. In this version you should provide your own random number generator, already initialized.")
    .def("__call__", &call_shuffler3, (arg("self"), arg("rng"), arg("data"), arg("target")), "Populates the output matrices by randomly selecting 'n' arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain 'n' rows and the number of columns that are dependent on input arraysets and target arrays.\n\nWe check don't 'data' and 'target' for size compatibility and is your responsibility to do so.")
    .def("__call__", call_shuffler4, (arg("self"), arg("data"), arg("target")), "This version is a shortcut to the previous declaration of operator() that actually instantiates its own random number generator and seed it a time-based variable. We guarantee two calls will lead to different results if they are at least 1 microsecond appart (procedure uses the machine clock).")
    ;
}
