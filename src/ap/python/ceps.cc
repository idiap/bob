/**
 * @file ap/python/ceps.cc
 * @date Wed Jan 11:10:40 2013 +0200
 * @author Elie El Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Binds the audio frame extractor to python.
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
#include <bob/ap/FrameExtractor.h>
#include <bob/ap/Energy.h>
#include <bob/ap/Spectrogram.h>
#include <bob/ap/Ceps.h>
#include <bob/python/ndarray.h>

using namespace boost::python;

// documentation for classes
static const char* FRAME_EXTRACTOR_DOC = "Objects of this class, after configuration, can extract audio frame from a 1D audio array/signal.";
static const char* ENERGY_DOC = "Objects of this class, after configuration, can extract the energy of frames extracted from a 1D audio array/signal.";
static const char* SPECTROGRAM_DOC = "Objects of this class, after configuration, can extract spectrograms from a 1D audio array/signal.";
static const char* CEPS_DOC = "Objects of this class, after configuration, can extract cepstral coefficients from a 1D audio array/signal.";

static boost::python::tuple py_extractor_get_shape(bob::ap::FrameExtractor& ext, object input_object)
{
  boost::python::tuple res;
  extract<int> int_check(input_object);
  if (int_check.check()) { //is int
    blitz::TinyVector<int,2> size = ext.getShape(int_check());
    res = boost::python::make_tuple(size[0], size[1]);
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(input_object);
    blitz::TinyVector<int,2> size = ext.getShape(val);
    res = boost::python::make_tuple(size[0], size[1]);
  }
  return res;
}

static object py_energy_call(bob::ap::Energy& energy, bob::python::const_ndarray input)
{
  // Gets the shape of the spectrogram
  const blitz::Array<double,1> input_ = input.bz<double,1>();
  const int s = energy.getShape(input_)(0);
  // Allocates a numpy array and defines the corresponding blitz wrapper
  bob::python::ndarray energy_array(bob::core::array::t_float64, s);
  blitz::Array<double,1> energy_array_ = energy_array.bz<double,1>();
  // Extracts the features
  energy(input_, energy_array_);
  return energy_array.self();
}

static object py_spectrogram_call(bob::ap::Spectrogram& spectrogram, bob::python::const_ndarray input)
{
  // Gets the shape of the spectrogram
  const blitz::Array<double,1> input_ = input.bz<double,1>();
  blitz::TinyVector<size_t,2> s = spectrogram.getShape(input_);
  // Allocates a numpy array and defines the corresponding blitz wrapper
  bob::python::ndarray spec_matrix(bob::core::array::t_float64, s(0), s(1));
  blitz::Array<double,2> spec_matrix_ = spec_matrix.bz<double,2>();
  // Extracts the features
  spectrogram(input_, spec_matrix_);
  return spec_matrix.self();
}

static object py_ceps_call(bob::ap::Ceps& ceps, bob::python::const_ndarray input)
{
  // Gets the shape of the feature
  const blitz::Array<double,1> input_ = input.bz<double,1>();
  blitz::TinyVector<size_t,2> s = ceps.getShape(input_);
  // Allocates a numpy array and defines the corresponding blitz wrapper
  bob::python::ndarray ceps_matrix(bob::core::array::t_float64, s(0), s(1));
  blitz::Array<double,2> ceps_matrix_ = ceps_matrix.bz<double,2>();
  // Extracts the features
  ceps(input_, ceps_matrix_);
  return ceps_matrix.self();
}

void bind_ap_ceps()
{
  class_<bob::ap::FrameExtractor, boost::shared_ptr<bob::ap::FrameExtractor> >("FrameExtractor", FRAME_EXTRACTOR_DOC, init<const double, optional<const double, const double> >((arg("sampling_frequency"), arg("win_length_ms")=20., arg("win_shift_ms")=10.)))
    .def(init<bob::ap::FrameExtractor&>(args("other"), "Constructs a new audio frame extractor from an existing one, using the copy constructor."))
    .def(self == self)
    .def(self != self)
    .add_property("sampling_frequency", &bob::ap::FrameExtractor::getSamplingFrequency, &bob::ap::FrameExtractor::setSamplingFrequency, "The sampling frequency of the input data")
    .add_property("win_length_ms", &bob::ap::FrameExtractor::getWinLengthMs, &bob::ap::FrameExtractor::setWinLengthMs, "The window length of the cepstral analysis in milliseconds")
    .add_property("win_length", &bob::ap::FrameExtractor::getWinLength, "The normalized window length wrt. to the sample frequency")
    .add_property("win_shift_ms", &bob::ap::FrameExtractor::getWinShiftMs, &bob::ap::FrameExtractor::setWinShiftMs, "The window shift of the cepstral analysis in milliseconds")
    .add_property("win_shift", &bob::ap::FrameExtractor::getWinShift, "The normalized window shift wrt. to the sample frequency")
    .def("get_shape", &py_extractor_get_shape, (arg("n_size"), arg("input_data")), "Computes the shape of the output features")
  ;

  class_<bob::ap::Energy, boost::shared_ptr<bob::ap::Energy>, bases<bob::ap::FrameExtractor> >("Energy", ENERGY_DOC, init<const double, optional<const double, const double> >((arg("sampling_frequency"), arg("win_length_ms")=20., arg("win_shift_ms")=10.)))
    .def(init<bob::ap::Energy&>(args("other"), "Constructs a new audio energy extractor from an existing one, using the copy constructor."))
    .def(self == self)
    .def(self != self)
    .add_property("energy_floor", &bob::ap::Energy::getEnergyFloor, &bob::ap::Energy::setEnergyFloor, "The energy flooring threshold")
    .def("__call__", &py_energy_call, (arg("input")), "Computes the energy features")
  ;

  class_<bob::ap::Spectrogram, boost::shared_ptr<bob::ap::Spectrogram>, bases<bob::ap::Energy> >("Spectrogram", SPECTROGRAM_DOC, init<const double, optional<const double, const double, const size_t, const double, const double, const double, const bool> >((arg("sampling_frequency"), arg("win_length_ms")=20., arg("win_shift_ms")=10., arg("n_filters")=24, arg("f_min")=0., arg("f_max")=4000., arg("pre_emphasis_coeff")=0.95, arg("mel_scale")=true)))
    .def(init<bob::ap::Spectrogram&>(args("other"), "Constructs a new spectrogram extractor from an existing one, using the copy constructor."))
    .def(self == self)
    .def(self != self)
    .add_property("sampling_frequency", &bob::ap::Spectrogram::getSamplingFrequency, &bob::ap::Spectrogram::setSamplingFrequency, "The sampling frequency of the input data")
    .add_property("win_length_ms", &bob::ap::Spectrogram::getWinLengthMs, &bob::ap::Spectrogram::setWinLengthMs, "The window length of the cepstral analysis in milliseconds")
    .add_property("win_shift_ms", &bob::ap::Spectrogram::getWinShiftMs, &bob::ap::Spectrogram::setWinShiftMs, "The window shift of the cepstral analysis in milliseconds")
    .add_property("n_filters", &bob::ap::Spectrogram::getNFilters, &bob::ap::Spectrogram::setNFilters, "The number of filter bands")
    .add_property("f_min", &bob::ap::Spectrogram::getFMin, &bob::ap::Spectrogram::setFMin, "The minimal frequency of the filter bank")
    .add_property("f_max", &bob::ap::Spectrogram::getFMax, &bob::ap::Spectrogram::setFMax, "The maximal frequency of the filter bank")
    .add_property("mel_scale", &bob::ap::Spectrogram::getMelScale, &bob::ap::Spectrogram::setMelScale, "Tells whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale")
    .add_property("pre_emphasis_coeff", &bob::ap::Spectrogram::getPreEmphasisCoeff, &bob::ap::Spectrogram::setPreEmphasisCoeff, "The coefficient used for the pre-emphasis")
    .add_property("energy_filter", &bob::ap::Spectrogram::getEnergyFilter, &bob::ap::Spectrogram::setEnergyFilter, "Tells whether we use the energy or the square root of the energy")
    .add_property("log_filter", &bob::ap::Spectrogram::getLogFilter, &bob::ap::Spectrogram::setLogFilter, "Tells whether we use the log triangular filter or the triangular filter")
    .add_property("energy_bands", &bob::ap::Spectrogram::getEnergyBands, &bob::ap::Spectrogram::setEnergyBands, "Tells whether we compute a spectrogram or energy bands")
    .def("__call__", &py_spectrogram_call, (arg("input")), "Computes the spectrogram")
  ;

  class_<bob::ap::Ceps, boost::shared_ptr<bob::ap::Ceps>, bases<bob::ap::Spectrogram> >("Ceps", CEPS_DOC, init<const double, optional<const double, const double, const size_t, const size_t, const double, const double, const size_t, const double, const bool, const bool> >((arg("sampling_frequency"), arg("win_length_ms")=20., arg("win_shift_ms")=10., arg("n_filters")=24, arg("n_ceps")=19, arg("f_min")=0., arg("f_max")=4000., arg("delta_win")=2, arg("pre_emphasis_coeff")=0.95, arg("mel_scale")=true, arg("dct_norm")=true)))
    .def(init<bob::ap::Ceps&>(args("other"), "Constructs a new spectrogram extractor from an existing one, using the copy constructor."))
    .def(self == self)
    .def(self != self)
    .add_property("n_filters", &bob::ap::Ceps::getNFilters, &bob::ap::Ceps::setNFilters, "The number of filter bands")
    .add_property("n_ceps", &bob::ap::Ceps::getNCeps, &bob::ap::Ceps::setNCeps, "The number of cepstral coefficients")
    .add_property("delta_win", &bob::ap::Ceps::getDeltaWin, &bob::ap::Ceps::setDeltaWin, "The integer delta value used for computing the first and second order derivatives")
    .add_property("dct_norm", &bob::ap::Ceps::getDctNorm, &bob::ap::Ceps::setDctNorm, "A factor by which the cepstral coefficients are multiplied")
    .add_property("with_energy", &bob::ap::Ceps::getWithEnergy, &bob::ap::Ceps::setWithEnergy, "Tells if we add the energy to the output feature")
    .add_property("with_delta", &bob::ap::Ceps::getWithDelta, &bob::ap::Ceps::setWithDelta, "Tells if we add the first derivatives to the output feature")
    .add_property("with_delta_delta", &bob::ap::Ceps::getWithDeltaDelta, &bob::ap::Ceps::setWithDeltaDelta, "Tells if we add the second derivatives to the output feature")
    .def("__call__", &py_ceps_call, (arg("input")), "Computes the cepstral coefficients")
  ;
}

