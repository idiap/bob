/**
 * @file python/machine/src/kmeans.cc
 * @date Tue Jul 26 15:11:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include <boost/concept_check.hpp>
#include "io/Arrayset.h"
#include "machine/KMeansMachine.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace io = bob::io;
namespace mach = bob::machine;
namespace bp = bob::python;
namespace ca = bob::core::array;

static tuple py_getVariancesAndWeightsForEachCluster(const mach::KMeansMachine& machine, io::Arrayset& ar) {
  size_t n_means = machine.getNMeans();
  size_t n_inputs = machine.getNInputs();
  bp::ndarray variances(ca::t_float64, n_means, n_inputs);
  bp::ndarray weights(ca::t_float64, n_means);
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachCluster(ar, variances_, weights_);
  return boost::python::make_tuple(variances.self(), weights.self());
}

static object py_getMean(const mach::KMeansMachine& kMeansMachine, const size_t i) {
  size_t n_inputs = kMeansMachine.getNInputs();
  bp::ndarray mean(ca::t_float64, n_inputs);
  blitz::Array<double,1> mean_ = mean.bz<double,1>();
  kMeansMachine.getMean(i, mean_);
  return mean.self();
}

static void py_setMean(mach::KMeansMachine& machine, const size_t i, bp::const_ndarray mean) {
  const ca::typeinfo& info = mean.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.setMean(i, mean.bz<double,1>());
}

static object py_getMeans(const mach::KMeansMachine& kMeansMachine) {
  size_t n_means = kMeansMachine.getNMeans();
  size_t n_inputs = kMeansMachine.getNInputs();
  bp::ndarray means(ca::t_float64, n_means, n_inputs);
  blitz::Array<double,2> means_ = means.bz<double,2>();
  means_ = kMeansMachine.getMeans();
  return means.self();
}

static void py_setMeans(mach::KMeansMachine& machine, bp::const_ndarray means) {
  const ca::typeinfo& info = means.type();
  if(info.dtype != ca::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.setMeans(means.bz<double,2>());
}

static double py_getDistanceFromMean(const mach::KMeansMachine& machine, bp::const_ndarray x, const size_t i) 
{
  const ca::typeinfo& info = x.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  return machine.getDistanceFromMean(x.bz<double,1>(), i);
}

static tuple py_getClosestMean(const mach::KMeansMachine& machine, bp::const_ndarray x) 
{
  const ca::typeinfo& info = x.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  size_t closest_mean;
  double min_distance;
  machine.getClosestMean(x.bz<double,1>(), closest_mean, min_distance);
  return boost::python::make_tuple(closest_mean, min_distance);
}

static double py_getMinDistance(const mach::KMeansMachine& machine, bp::const_ndarray input) 
{
  const ca::typeinfo& info = input.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  return machine.getMinDistance(input.bz<double,1>());
}


void bind_machine_kmeans() 
{
  class_<mach::KMeansMachine, bases<mach::Machine<blitz::Array<double,1>, double> > >("KMeansMachine",
      "This class implements a k-means classifier.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<>())
    .def(init<const size_t, const size_t>(args("n_means", "n_inputs")))
    .def(init<mach::KMeansMachine&>())
    .def(init<io::HDF5File&>(args("config")))
    .def(self == self)
    .add_property("means", &py_getMeans, &py_setMeans, "The mean vectors")
    .add_property("dim_d", &mach::KMeansMachine::getNInputs, "Number of inputs")
    .add_property("dim_c", &mach::KMeansMachine::getNMeans, "Number of means (k)")
    .def("resize", &mach::KMeansMachine::resize, (arg("n_means"), arg("n_inputs")), "Resize the number of means and inputs")
    .def("get_mean", &py_getMean, (arg("i"), arg("mean")), "Get the i'th mean")
    .def("set_mean", &py_setMean, (arg("i"), arg("mean")), "Set the i'th mean")
    .def("get_distance_from_mean", &py_getDistanceFromMean, (arg("x"), arg("i")),
        "Return the power of two of the Euclidean distance of the sample, x, to the i'th mean")
    .def("get_closest_mean", &py_getClosestMean, (arg("x")),
        "Calculate the index of the mean that is closest (in terms of Euclidean distance) to the data sample, x")
    .def("get_min_distance", &py_getMinDistance, (arg("input")),
        "Output the minimum distance between the input and one of the means")
    .def("get_variances_and_weights_for_each_cluster", &py_getVariancesAndWeightsForEachCluster, (arg("machine"), arg("data")),
        "For each mean, find the subset of the samples that is closest to that mean, and calculate\n"
        "1) the variance of that subset (the cluster variance)\n"
        "2) the proportion of the samples represented by that subset (the cluster weight)")
    .def("load", &mach::KMeansMachine::load, "Load from a Configuration")
    .def("save", &mach::KMeansMachine::save, "Save to a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}
