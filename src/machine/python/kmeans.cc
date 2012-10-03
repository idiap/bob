/**
 * @file machine/python/kmeans.cc
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
#include "bob/machine/KMeansMachine.h"

#include "bob/core/python/ndarray.h"

using namespace boost::python;

static tuple py_getVariancesAndWeightsForEachCluster(const bob::machine::KMeansMachine& machine, bob::python::const_ndarray ar) {
  size_t n_means = machine.getNMeans();
  size_t n_inputs = machine.getNInputs();
  bob::python::ndarray variances(bob::core::array::t_float64, n_means, n_inputs);
  bob::python::ndarray weights(bob::core::array::t_float64, n_means);
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  const bob::core::array::typeinfo& info = ar.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.getVariancesAndWeightsForEachCluster(ar.bz<double,2>(), variances_, weights_);
  return boost::python::make_tuple(variances.self(), weights.self());
}

static void py_getVariancesAndWeightsForEachClusterInit(const bob::machine::KMeansMachine& machine, bob::python::ndarray variances, bob::python::ndarray weights) {
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachClusterInit(variances_, weights_);
}

static void py_getVariancesAndWeightsForEachClusterAcc(const bob::machine::KMeansMachine& machine, bob::python::const_ndarray ar, bob::python::ndarray variances, bob::python::ndarray weights) {
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  const bob::core::array::typeinfo& info = ar.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.getVariancesAndWeightsForEachClusterAcc(ar.bz<double,2>(), variances_, weights_);
}

static void py_getVariancesAndWeightsForEachClusterFin(const bob::machine::KMeansMachine& machine, bob::python::ndarray variances, bob::python::ndarray weights) {
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachClusterFin(variances_, weights_);
}

static object py_getMean(const bob::machine::KMeansMachine& kMeansMachine, const size_t i) {
  size_t n_inputs = kMeansMachine.getNInputs();
  bob::python::ndarray mean(bob::core::array::t_float64, n_inputs);
  blitz::Array<double,1> mean_ = mean.bz<double,1>();
  kMeansMachine.getMean(i, mean_);
  return mean.self();
}

static void py_setMean(bob::machine::KMeansMachine& machine, const size_t i, bob::python::const_ndarray mean) {
  const bob::core::array::typeinfo& info = mean.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.setMean(i, mean.bz<double,1>());
}

static object py_getMeans(const bob::machine::KMeansMachine& kMeansMachine) {
  size_t n_means = kMeansMachine.getNMeans();
  size_t n_inputs = kMeansMachine.getNInputs();
  bob::python::ndarray means(bob::core::array::t_float64, n_means, n_inputs);
  blitz::Array<double,2> means_ = means.bz<double,2>();
  means_ = kMeansMachine.getMeans();
  return means.self();
}

static void py_setMeans(bob::machine::KMeansMachine& machine, bob::python::const_ndarray means) {
  const bob::core::array::typeinfo& info = means.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.setMeans(means.bz<double,2>());
}

static double py_getDistanceFromMean(const bob::machine::KMeansMachine& machine, bob::python::const_ndarray x, const size_t i) 
{
  const bob::core::array::typeinfo& info = x.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  return machine.getDistanceFromMean(x.bz<double,1>(), i);
}

static tuple py_getClosestMean(const bob::machine::KMeansMachine& machine, bob::python::const_ndarray x) 
{
  const bob::core::array::typeinfo& info = x.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  size_t closest_mean;
  double min_distance;
  machine.getClosestMean(x.bz<double,1>(), closest_mean, min_distance);
  return boost::python::make_tuple(closest_mean, min_distance);
}

static double py_getMinDistance(const bob::machine::KMeansMachine& machine, bob::python::const_ndarray input) 
{
  const bob::core::array::typeinfo& info = input.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  return machine.getMinDistance(input.bz<double,1>());
}

static object py_getCacheMeans(const bob::machine::KMeansMachine& kMeansMachine) {
  size_t n_means = kMeansMachine.getNMeans();
  size_t n_inputs = kMeansMachine.getNInputs();
  bob::python::ndarray cache_means(bob::core::array::t_float64, n_means, n_inputs);
  blitz::Array<double,2> cache_means_ = cache_means.bz<double,2>();
  cache_means_ = kMeansMachine.getCacheMeans();
  return cache_means.self();
}

static void py_setCacheMeans(bob::machine::KMeansMachine& machine, bob::python::const_ndarray cache_means) {
  const bob::core::array::typeinfo& info = cache_means.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  machine.setCacheMeans(cache_means.bz<double,2>());
}

void bind_machine_kmeans() 
{
  class_<bob::machine::KMeansMachine, boost::shared_ptr<bob::machine::KMeansMachine>, 
         bases<bob::machine::Machine<blitz::Array<double,1>, double> > >("KMeansMachine",
      "This class implements a k-means classifier.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<>())
    .def(init<const size_t, const size_t>(args("n_means", "n_inputs")))
    .def(init<bob::machine::KMeansMachine&>())
    .def(init<bob::io::HDF5File&>(args("config")))
    .def(self == self)
    .def(self != self)
    .add_property("means", &py_getMeans, &py_setMeans, "The mean vectors")
    .add_property("__cache_means__", &py_getCacheMeans, &py_setCacheMeans, "The cache mean vectors. This should only be used when parallelizing the get_variances_and_weights_for_each_cluster() method")
    .add_property("dim_d", &bob::machine::KMeansMachine::getNInputs, "Number of inputs")
    .add_property("dim_c", &bob::machine::KMeansMachine::getNMeans, "Number of means (k)")
    .def("resize", &bob::machine::KMeansMachine::resize, (arg("n_means"), arg("n_inputs")), "Resize the number of means and inputs")
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
    .def("__get_variances_and_weights_for_each_cluster_init__", &py_getVariancesAndWeightsForEachClusterInit, (arg("machine"), arg("variances"), arg("weights")),
        "For the parallel version of get_variances_and_weights_for_each_cluster()\n"
        "Initialization step")
    .def("__get_variances_and_weights_for_each_cluster_acc__", &py_getVariancesAndWeightsForEachClusterAcc, (arg("machine"), arg("data"), arg("variances"), arg("weights")),
        "For the parallel version of get_variances_and_weights_for_each_cluster()\n"
        "Accumulation step")
    .def("__get_variances_and_weights_for_each_cluster_fin__", &py_getVariancesAndWeightsForEachClusterFin, (arg("machine"), arg("variances"), arg("weights")),
        "For the parallel version of get_variances_and_weights_for_each_cluster()\n"
        "Finalization step")
    .def("load", &bob::machine::KMeansMachine::load, "Load from a Configuration")
    .def("save", &bob::machine::KMeansMachine::save, "Save to a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}
