/**
 * @file python/ip/src/Gabor.cc
 * @date Wed Apr 13 20:16:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Gabor filters to python
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "core/python/ndarray.h"

#include "sp/convolution.h"
#include "ip/GaborSpatial.h"
#include "ip/GaborFrequency.h"
#include "ip/GaborBankSpatial.h"
#include "ip/GaborBankFrequency.h"

using namespace boost::python;
namespace ip = bob::ip;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* gabor_spatial_doc = "Objects of this class, after configuration, can filter images with a Gabor kernel, performing the operation in the spatial domain.";
static const char* gabor_bank_spatial_doc = "Objects of this class, after configuration, can filter images with a bank of Gabor kernels, performing the operation in the spatial domain.";
static const char* gabor_frequency_doc = "Objects of this class, after configuration, can filter images with a Gabor kernel, performing the operation in the frequency domain.";
static const char* gabor_bank_frequency_doc = "Objects of this class, after configuration, can filter images with a bank of Gabor kernels, performing the operation in the frequency domain.";

static void call_gabsap(ip::GaborSpatial& op, tp::const_ndarray input,
    tp::ndarray output) {
  blitz::Array<std::complex<double>,2> output_ = 
    output.bz<std::complex<double>,2>();
  op(input.bz<std::complex<double>,2>(), output_);
}

static void call_gabfreq(ip::GaborFrequency& op, tp::const_ndarray input,
    tp::ndarray output) {
  blitz::Array<std::complex<double>,2> output_ = 
    output.bz<std::complex<double>,2>();
  op(input.bz<std::complex<double>,2>(), output_);
}

static void call_gabbspa(ip::GaborBankSpatial& op, tp::const_ndarray input,
    tp::ndarray output) {
  blitz::Array<std::complex<double>,3> output_ = 
    output.bz<std::complex<double>,3>();
  op(input.bz<std::complex<double>,2>(), output_);
}

static void call_gabbfreq(ip::GaborBankFrequency& op, tp::const_ndarray input,
    tp::ndarray output) {
  blitz::Array<std::complex<double>,3> output_ = 
    output.bz<std::complex<double>,3>();
  op(input.bz<std::complex<double>,2>(), output_);
}

void bind_ip_gabor() {
  enum_<bob::ip::Gabor::NormOption>("GaborNorm")
    .value("NoNorm", bob::ip::Gabor::NoNorm)
    .value("SpatialFactor", bob::ip::Gabor::SpatialFactor)
    .value("ZeroMeanUnitVar", bob::ip::Gabor::ZeroMeanUnitVar)
    ;

  class_<ip::GaborSpatial, boost::shared_ptr<ip::GaborSpatial> >("GaborSpatial", gabor_spatial_doc, init<optional<const double, const double, const double, const double, const int, const bool, const enum bob::ip::Gabor::NormOption, const enum bob::sp::Convolution::BorderOption> >((arg("f")=0.25, arg("theta")=0., arg("gamma")=1., arg("eta")=1., arg("spatial_size")=35, arg("cancel_dc")=false, arg("norm")=ip::Gabor::SpatialFactor, arg("border_opt")=bob::sp::Convolution::Mirror), "Constructs a new Gabor filter in the spatial domain."))
    .add_property("f", &ip::GaborSpatial::getF, &ip::GaborSpatial::setF)
    .add_property("theta", &ip::GaborSpatial::getTheta, &ip::GaborSpatial::setTheta)
    .add_property("gamma", &ip::GaborSpatial::getGamma, &ip::GaborSpatial::setGamma)
    .add_property("eta", &ip::GaborSpatial::getEta, &ip::GaborSpatial::setEta)
    .add_property("spatial_size", &ip::GaborSpatial::getSpatialSize, &ip::GaborSpatial::setSpatialSize)
    .add_property("cancel_dc", &ip::GaborSpatial::getCancelDc, &ip::GaborSpatial::setCancelDc)
    .add_property("norm_option", &ip::GaborSpatial::getNormOption, &ip::GaborSpatial::setNormOption)
    .add_property("border_option", &ip::GaborSpatial::getBorderOption, &ip::GaborSpatial::setBorderOption)
    .add_property("kernel", make_function(&ip::GaborSpatial::getKernel, return_value_policy<copy_const_reference>()))
    .def("__call__", &call_gabsap, (arg("input"), arg("output")), "Call an object of this type to filter an image.")
    ;

  class_<ip::GaborFrequency, boost::shared_ptr<ip::GaborFrequency> >("GaborFrequency", gabor_frequency_doc, init<const int, const int, optional<const double, const double, const double, const double, const double, const bool, const bool, const bool> >((arg("height"), arg("width"), arg("f")=0.25, arg("theta")=0., arg("gamma")=1., arg("eta")=1., arg("pf")=0.99, arg("cancel_dc")=false, arg("use_envelope")=false, arg("output_in_frequency")=false), "Constructs a new Gabor filter in the frequency domain."))
    .add_property("height", &ip::GaborFrequency::getHeight, &ip::GaborFrequency::setHeight)
    .add_property("width", &ip::GaborFrequency::getWidth, &ip::GaborFrequency::setWidth)
    .add_property("f", &ip::GaborFrequency::getF, &ip::GaborFrequency::setF)
    .add_property("theta", &ip::GaborFrequency::getTheta, &ip::GaborFrequency::setTheta)
    .add_property("gamma", &ip::GaborFrequency::getGamma, &ip::GaborFrequency::setGamma)
    .add_property("eta", &ip::GaborFrequency::getEta, &ip::GaborFrequency::setEta)
    .add_property("pf", &ip::GaborFrequency::getPf, &ip::GaborFrequency::setPf)
    .add_property("cancel_dc", &ip::GaborFrequency::getCancelDc, &ip::GaborFrequency::setCancelDc)
    .add_property("use_envelope", &ip::GaborFrequency::getUseEnvelope, &ip::GaborFrequency::setUseEnvelope)
    .add_property("kernel", make_function(&ip::GaborFrequency::getKernel, return_value_policy<copy_const_reference>()))
    .add_property("kernel_shifted", make_function(&ip::GaborFrequency::getKernelShifted, return_value_policy<copy_const_reference>()))
    .add_property("kernel_envelope", &ip::GaborFrequency::getKernelEnvelope)
    .def("__call__", &call_gabfreq, "Call an object of this type to filter an image.")
    ;

  class_<ip::GaborBankSpatial, boost::shared_ptr<ip::GaborBankSpatial> >("GaborBankSpatial", gabor_bank_spatial_doc, init<optional<const int, const int, const double, const bool, const double, const double, const double, const double, const int, const bool, const enum bob::ip::Gabor::NormOption, const enum bob::sp::Convolution::BorderOption> >((arg("n_orient"), arg("n_freq"), arg("fmax")=0.25, arg("orientation_full")=false, arg("k")=1.414, arg("p")=0.5, arg("gamma")=1., arg("eta")=1., arg("spatial_size")=35, arg("cancel_dc")=false, arg("norm")=ip::Gabor::SpatialFactor, arg("border_opt")=bob::sp::Convolution::Mirror), "Constructs a new Gabor filter bank in the spatial domain."))
    .add_property("n_orient", &ip::GaborBankSpatial::getNOrient, &ip::GaborBankSpatial::setNOrient)
    .add_property("n_freq", &ip::GaborBankSpatial::getNFreq, &ip::GaborBankSpatial::setNFreq)
    .add_property("fmax", &ip::GaborBankSpatial::getFmax, &ip::GaborBankSpatial::setFmax)
    .add_property("orientation_full", &ip::GaborBankSpatial::getOrientationFull, &ip::GaborBankSpatial::setOrientationFull)
    .add_property("k", &ip::GaborBankSpatial::getK, &ip::GaborBankSpatial::setK)
    .add_property("p", &ip::GaborBankSpatial::getP, &ip::GaborBankSpatial::setP)
    .add_property("gamma", &ip::GaborBankSpatial::getGamma, &ip::GaborBankSpatial::setGamma)
    .add_property("eta", &ip::GaborBankSpatial::getEta, &ip::GaborBankSpatial::setEta)
    .add_property("spatial_size", &ip::GaborBankSpatial::getSpatialSize, &ip::GaborBankSpatial::setSpatialSize)
    .add_property("cancel_dc", &ip::GaborBankSpatial::getCancelDc, &ip::GaborBankSpatial::setCancelDc)
    .add_property("norm_option", &ip::GaborBankSpatial::getNormOption, &ip::GaborBankSpatial::setNormOption)
    .add_property("border_option", &ip::GaborBankSpatial::getBorderOption, &ip::GaborBankSpatial::setBorderOption)
    .def("__call__", &call_gabbspa, (arg("input"), arg("output")), "Call an object of this type to filter an image.")

    ;

  class_<ip::GaborBankFrequency, boost::shared_ptr<ip::GaborBankFrequency> >("GaborBankFrequency", gabor_bank_frequency_doc, init<const int, const int, optional<const int, const int, const double, const bool, const double, const double, const bool, const double, const double, const double, const bool/*, const bool, const bool*/> >((arg("height"), arg("width"), arg("n_orient")=8, arg("n_freq")=5, arg("fmax")=0.25, arg("orientation_full")=false, arg("k")=1.414, arg("p")=0.5, arg("optimal_gamma_eta")=false, arg("gamma")=1., arg("eta")=1., arg("pf")=0.99, arg("cancel_dc")=false/*, arg("use_envelope")=true, arg("output_in_frequency")=false*/), "Constructs a new Gabor filter in the frequency domain."))
    .add_property("height", &ip::GaborBankFrequency::getHeight, &ip::GaborBankFrequency::setHeight)
    .add_property("width", &ip::GaborBankFrequency::getWidth, &ip::GaborBankFrequency::setWidth)
    .add_property("n_orient", &ip::GaborBankFrequency::getNOrient, &ip::GaborBankFrequency::setNOrient)
    .add_property("n_freq", &ip::GaborBankFrequency::getNFreq, &ip::GaborBankFrequency::setNFreq)
    .add_property("fmax", &ip::GaborBankFrequency::getFmax, &ip::GaborBankFrequency::setFmax)
    .add_property("orientation_full", &ip::GaborBankFrequency::getOrientationFull, &ip::GaborBankFrequency::setOrientationFull)
    .add_property("k", &ip::GaborBankFrequency::getK, &ip::GaborBankFrequency::setK)
    .add_property("p", &ip::GaborBankFrequency::getP, &ip::GaborBankFrequency::setP)
    .add_property("optimal_gamma_eta", &ip::GaborBankFrequency::getOptimalGammaEta, &ip::GaborBankFrequency::setOptimalGammaEta)
    .add_property("gamma", &ip::GaborBankFrequency::getGamma, &ip::GaborBankFrequency::setGamma)
    .add_property("eta", &ip::GaborBankFrequency::getEta, &ip::GaborBankFrequency::setEta)
    .add_property("pf", &ip::GaborBankFrequency::getPf, &ip::GaborBankFrequency::setPf)
    .add_property("cancel_dc", &ip::GaborBankFrequency::getCancelDc, &ip::GaborBankFrequency::setCancelDc)
    .add_property("use_envelope", &ip::GaborBankFrequency::getUseEnvelope, &ip::GaborBankFrequency::setUseEnvelope)
    .def("__call__", &call_gabbfreq, (arg("input"), arg("output")), "Call an object of this type to filter an image.")
    ;


}
