/**
 * @file ap/python/ceps.cc
 * @date Wed Jan 11:10:40 2013 +0200
 * @author Elie El Khoury <Elie.Khoury@idiap.ch>
 *
 * @brief Binds the Cepstral Feature Extraction to python.
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

#include "bob/ap/Ceps.h"
#include "bob/core/python/ndarray.h"

using namespace boost::python;

// documentation for classes
static const char* CEPS_DOC = "Objects of this class, after configuration, can extract Cepstral Features from a 1D array/signal.";
static const char* TESTCEPS_DOC = "Objects of this class, after configuration, can be used to test the private methods of bob.ap.Ceps.";

static object py_ceps_analysis(bob::ap::Ceps& ceps, bob::python::const_ndarray input)
{
  // Get the shape of the feature
  const blitz::Array<double,1> input_ = input.bz<double,1>();
  blitz::TinyVector<size_t,2> s = ceps.getCepsShape(input_);

  // Allocate numpy array and define blitz wrapper
  bob::python::ndarray ceps_matrix(bob::core::array::t_float64, s(0), s(1));
  blitz::Array<double,2> ceps_matrix_ = ceps_matrix.bz<double,2>();

  // Extract the features
  ceps.CepsAnalysis(input_, ceps_matrix_);
  return ceps_matrix.self();
}

static boost::python::tuple py_get_ceps_shape(bob::ap::Ceps& ceps, object input_object)
{
  boost::python::tuple res;
  extract<int> int_check(input_object);
  if (int_check.check()) { //is int
    blitz::TinyVector<int,2> size = ceps.getCepsShape(int_check());
    res = boost::python::make_tuple(size[0], size[1]);
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(input_object);
    blitz::TinyVector<int,2> size = ceps.getCepsShape(val);
    res = boost::python::make_tuple(size[0], size[1]);
  }
  return res;
}

static double py_logEnergy(bob::ap::TestCeps& ceps, bob::python::ndarray data)
{
  blitz::Array<double,1> data_ = data.bz<double,1>();

  // Get the logEnergy
  return ceps.logEnergy(data_);
}

static void py_emphasis(bob::ap::TestCeps& ceps, bob::python::ndarray data, double a)
{
  blitz::Array<double,1> data_ = data.bz<double,1>();

  //Compute the Pre-Emphasis
  ceps.emphasis(data_, a);
}

static void py_hammingWindow(bob::ap::TestCeps& ceps, bob::python::ndarray data)
{
  blitz::Array<double,1> data_ = data.bz<double,1>();

  //Compute the Hamming Wrapping
  ceps.hammingWindow(data_);
}

static object py_logFilterBank(bob::ap::TestCeps& ceps, bob::python::ndarray data, int m_win_size, int n_filters)
{

  blitz::Array<double,1> data_ = data.bz<double,1>();
  ceps.logFilterBank(data_);
  bob::python::ndarray filter(bob::core::array::t_float64, n_filters);
  blitz::Array<double,1> filter_ = filter.bz<double,1>();

  // Get the filter Bank
  filter_ = ceps.getFilter();
  return filter.self();
}

static object py_transformDCT(bob::ap::TestCeps& ceps, int n_ceps)
{
  ceps.transformDCT();
  bob::python::ndarray features(bob::core::array::t_float64, n_ceps);
  blitz::Array<double,1> features_ = features.bz<double,1>();

  // Get the Cepstral features
  features_ = ceps.getFeatures();
  return features.self();
}


void bind_ap_ceps()
{
  class_<bob::ap::Ceps, boost::shared_ptr<bob::ap::Ceps> >("Ceps", CEPS_DOC, init<double, int, int, int, int, double, double, double>
  ((arg("sf"), arg("win_length_ms"), arg("win_shift_ms"), arg("n_filters"), arg("n_ceps"), arg("f_min"), arg("f_max"), arg("delta_win"))))
        .add_property("sample_frequency", &bob::ap::Ceps::getSampleFrequency, &bob::ap::Ceps::setSampleFrequency, "The sample frequency of the input data")
        .add_property("win_length_ms", &bob::ap::Ceps::getWinLengthMs, &bob::ap::Ceps::setWinLengthMs, "The window length of the cepstral analysis in miliseconds")
        .add_property("win_length", &bob::ap::Ceps::getWinLength, "The normalized window length wrt. to the sample frequency")
        .add_property("win_shift_ms", &bob::ap::Ceps::getWinShiftMs, &bob::ap::Ceps::setWinShiftMs, "The window shift of the cepstral analysis in miliseconds")
        .add_property("win_shift", &bob::ap::Ceps::getWinShift, "The normalized window shift wrt. to the sample frequency")
        .add_property("win_size", &bob::ap::Ceps::getWinSize, "The window size")
        .add_property("n_filters", &bob::ap::Ceps::getNFilters, &bob::ap::Ceps::setNFilters, "The number of filter bands")
        .add_property("n_ceps", &bob::ap::Ceps::getNCeps, &bob::ap::Ceps::setNCeps, "The number of cepstral coefficients")
        .add_property("f_min", &bob::ap::Ceps::getFMin, &bob::ap::Ceps::setFMin, "The minimal frequency of the filter bank")
        .add_property("f_max", &bob::ap::Ceps::getFMax, &bob::ap::Ceps::setFMax, "The maximal frequency of the filter bank")
        .add_property("fb_linear", &bob::ap::Ceps::getFbLinear, &bob::ap::Ceps::setFbLinear, "Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale")
        .add_property("delta_win", &bob::ap::Ceps::getDeltaWin, &bob::ap::Ceps::setDeltaWin, "The integer delta value used for computing the first and second order derivatives")
        .add_property("dct_norm", &bob::ap::Ceps::getDctNorm, &bob::ap::Ceps::setDctNorm, "A factor by which the cepstral coefficients are multiplied")
        .add_property("with_energy", &bob::ap::Ceps::getWithEnergy, &bob::ap::Ceps::setWithEnergy, "Tells if we add the energy to the output feature")
        .add_property("with_delta", &bob::ap::Ceps::getWithDelta, &bob::ap::Ceps::setWithDelta, "Tells if we add the first derivatives to the output feature")
        .add_property("with_delta_delta", &bob::ap::Ceps::getWithDeltaDelta, &bob::ap::Ceps::setWithDeltaDelta, "Tells if we add the second derivatives to the output feature")
        .add_property("with_delta_energy", &bob::ap::Ceps::getWithDeltaEnergy, &bob::ap::Ceps::setWithDeltaEnergy, "Tells if we add the first derivative of the energy to the output feature")
        .add_property("with_delta_delta_energy", &bob::ap::Ceps::getWithDeltaDeltaEnergy, &bob::ap::Ceps::setWithDeltaDeltaEnergy, "Tells if we add the second derivative of the energy to the output feature")
        .def("ceps_analysis", &py_ceps_analysis, (arg("input")), "Compute the features")
        .def("get_ceps_shape", &py_get_ceps_shape, (arg("n_size"), arg("input_data")), "Compute the shape of the output features")
        ;

  class_<bob::ap::TestCeps, boost::shared_ptr<bob::ap::TestCeps> >("TestCeps", TESTCEPS_DOC, init<bob::ap::Ceps&>((arg("ceps"))))
        .def("mel", &bob::ap::TestCeps::mel, (arg("f")), "Compute a mel scale.")
        .def("mel_inv", &bob::ap::TestCeps::melInv, (arg("f")), "Compute an inverse mel scale.")
        .def("log_energy", &py_logEnergy, (arg("data")), "compute the gain")
        .def("pre_emphasis", &py_emphasis, (arg("data"), arg("a")), "compute pre-emphasis")
        .def("hamming_window", &py_hammingWindow, (arg("data")), "compute the wraped signal on a hamming Window")
        .def("log_filter_bank", &py_logFilterBank, (arg("data"), arg("m_win_size"), arg("n_filters")), "compute log Filter Bank")
        .def("dct_transform", &py_transformDCT, (arg("n_ceps")), "DCT Transform")
      ;
}

