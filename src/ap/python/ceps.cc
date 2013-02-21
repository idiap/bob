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

static object py_forward(bob::ap::Ceps& ceps, bob::python::const_ndarray input)
{
  // Gets the shape of the feature
  const blitz::Array<double,1> input_ = input.bz<double,1>();
  blitz::TinyVector<size_t,2> s = ceps.getCepsShape(input_);
  // Allocates a numpy array and defines the corresponding blitz wrapper
  bob::python::ndarray ceps_matrix(bob::core::array::t_float64, s(0), s(1));
  blitz::Array<double,2> ceps_matrix_ = ceps_matrix.bz<double,2>();
  // Extracts the features
  ceps(input_, ceps_matrix_);
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
  return ceps.logEnergy(data_);
}

static void py_emphasis(bob::ap::TestCeps& ceps, bob::python::ndarray data)
{
  blitz::Array<double,1> data_ = data.bz<double,1>();
  ceps.pre_emphasis(data_);
}

static void py_hammingWindow(bob::ap::TestCeps& ceps, bob::python::ndarray data)
{
  blitz::Array<double,1> data_ = data.bz<double,1>();
  ceps.hammingWindow(data_);
}

static object py_logFilterBank(bob::ap::TestCeps& ceps, bob::python::ndarray data, int m_win_size, int n_filters)
{

  blitz::Array<double,1> data_ = data.bz<double,1>();
  ceps.logFilterBank(data_);
  bob::python::ndarray filter(bob::core::array::t_float64, n_filters);
  blitz::Array<double,1> filter_ = filter.bz<double,1>();
  // Gets the filter bank output
  filter_ = ceps.getFilterOutput();
  return filter.self();
}

static object py_applyDct(bob::ap::TestCeps& ceps, int n_ceps)
{
  bob::python::ndarray ceps_row(bob::core::array::t_float64, n_ceps);
  blitz::Array<double,1> ceps_row_ = ceps_row.bz<double,1>();
  // Gets the Cepstral features
  ceps.applyDct(ceps_row_);
  return ceps_row.self();
}


void bind_ap_ceps()
{
  class_<bob::ap::Ceps, boost::shared_ptr<bob::ap::Ceps> >("Ceps", CEPS_DOC, init<double, optional<double, double, size_t, size_t, double, double, int, double, bool, bool> >
  ((arg("sampling_frequency"), arg("win_length_ms")=20., arg("win_shift_ms")=10., arg("n_filters")=24, arg("n_ceps")=19, arg("f_min")=0., arg("f_max")=4000., arg("delta_win"), arg("pre_emphasis_coeff")=0.95, arg("mel_scale")=true, arg("dct_norm")=false)))
        .def(init<bob::ap::Ceps&>(args("other"), "Constructs a new Ceps feature extractor from an existing one, using the copy constructor."))
        .def(self == self)
        .def(self != self)
        .add_property("sampling_frequency", &bob::ap::Ceps::getSamplingFrequency, &bob::ap::Ceps::setSamplingFrequency, "The sampling frequency of the input data")
        .add_property("win_length_ms", &bob::ap::Ceps::getWinLengthMs, &bob::ap::Ceps::setWinLengthMs, "The window length of the cepstral analysis in milliseconds")
        .add_property("win_length", &bob::ap::Ceps::getWinLength, "The normalized window length wrt. to the sample frequency")
        .add_property("win_shift_ms", &bob::ap::Ceps::getWinShiftMs, &bob::ap::Ceps::setWinShiftMs, "The window shift of the cepstral analysis in milliseconds")
        .add_property("win_shift", &bob::ap::Ceps::getWinShift, "The normalized window shift wrt. to the sample frequency")
        .add_property("n_filters", &bob::ap::Ceps::getNFilters, &bob::ap::Ceps::setNFilters, "The number of filter bands")
        .add_property("n_ceps", &bob::ap::Ceps::getNCeps, &bob::ap::Ceps::setNCeps, "The number of cepstral coefficients")
        .add_property("f_min", &bob::ap::Ceps::getFMin, &bob::ap::Ceps::setFMin, "The minimal frequency of the filter bank")
        .add_property("f_max", &bob::ap::Ceps::getFMax, &bob::ap::Ceps::setFMax, "The maximal frequency of the filter bank")
        .add_property("mel_scale", &bob::ap::Ceps::getMelScale, &bob::ap::Ceps::setMelScale, "Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale")
        .add_property("delta_win", &bob::ap::Ceps::getDeltaWin, &bob::ap::Ceps::setDeltaWin, "The integer delta value used for computing the first and second order derivatives")
        .add_property("pre_emphasis_coeff", &bob::ap::Ceps::getPreEmphasisCoeff, &bob::ap::Ceps::setPreEmphasisCoeff, "The coefficient used for the pre-emphasis")
        .add_property("dct_norm", &bob::ap::Ceps::getDctNorm, &bob::ap::Ceps::setDctNorm, "A factor by which the cepstral coefficients are multiplied")
        .add_property("with_energy", &bob::ap::Ceps::getWithEnergy, &bob::ap::Ceps::setWithEnergy, "Tells if we add the energy to the output feature")
        .add_property("with_delta", &bob::ap::Ceps::getWithDelta, &bob::ap::Ceps::setWithDelta, "Tells if we add the first derivatives to the output feature")
        .add_property("with_delta_delta", &bob::ap::Ceps::getWithDeltaDelta, &bob::ap::Ceps::setWithDeltaDelta, "Tells if we add the second derivatives to the output feature")
        .def("__call__", &py_forward, (arg("input")), "Computes the cepstral features")
        .def("get_ceps_shape", &py_get_ceps_shape, (arg("n_size"), arg("input_data")), "Computes the shape of the output features")
        ;

  class_<bob::ap::TestCeps, boost::shared_ptr<bob::ap::TestCeps> >("TestCeps", TESTCEPS_DOC, init<bob::ap::Ceps&>((arg("ceps"))))
        .def("herz_to_mel", &bob::ap::TestCeps::herzToMel, (arg("f")), "Converts a frequency in Herz into the corresponding one in Mel.")
        .def("mel_to_herz", &bob::ap::TestCeps::melToHerz, (arg("f")), "Converts a frequency in Mel into the corresponding one in Herz.")
        .def("log_energy", &py_logEnergy, (arg("data")), "compute the gain")
        .def("pre_emphasis", &py_emphasis, (arg("data")), "compute pre-emphasis")
        .def("hamming_window", &py_hammingWindow, (arg("data")), "compute the wraped signal on a hamming Window")
        .def("log_filter_bank", &py_logFilterBank, (arg("data"), arg("m_win_size"), arg("n_filters")), "compute log Filter Bank")
        .def("apply_dct", &py_applyDct, (arg("n_ceps")), "DCT Transform")
      ;
}

