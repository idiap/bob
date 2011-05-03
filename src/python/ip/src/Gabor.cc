/**
 * @file src/python/ip/src/Gabor.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the Gabor filters to python
 */

#include <boost/python.hpp>

#include "sp/convolution.h"
#include "ip/GaborSpatial.h"
#include "ip/GaborFrequency.h"
#include "ip/GaborBankSpatial.h"
#include "ip/GaborBankFrequency.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* gabor_spatial_doc = "Objects of this class, after configuration, can filter images with a Gabor kernel, performing the operation in the spatial domain.";
static const char* gabor_bank_spatial_doc = "Objects of this class, after configuration, can filter images with a bank of Gabor kernels, performing the operation in the spatial domain.";
static const char* gabor_frequency_doc = "Objects of this class, after configuration, can filter images with a Gabor kernel, performing the operation in the frequency domain.";
static const char* gabor_bank_frequency_doc = "Objects of this class, after configuration, can filter images with a bank of Gabor kernels, performing the operation in the frequency domain.";


void bind_ip_gabor() {
  enum_<Torch::ip::Gabor::NormOption>("GaborNorm")
    .value("NoNorm", Torch::ip::Gabor::NoNorm)
    .value("SpatialFactor", Torch::ip::Gabor::SpatialFactor)
    .value("ZeroMeanUnitVar", Torch::ip::Gabor::ZeroMeanUnitVar)
    ;

  class_<ip::GaborSpatial, boost::shared_ptr<ip::GaborSpatial> >("GaborSpatial", gabor_spatial_doc, init<optional<const double, const double, const double, const double, const int, const bool, const enum Torch::ip::Gabor::NormOption, const enum Torch::sp::Convolution::BorderOption> >((arg("f")=0.25, arg("theta")=0., arg("gamma")=1., arg("eta")=1., arg("spatial_size")=35, arg("cancel_dc")=false, arg("norm")=ip::Gabor::SpatialFactor, arg("border_opt")=Torch::sp::Convolution::Mirror), "Constructs a new Gabor filter in the spatial domain."))
    .add_property("f", &ip::GaborSpatial::getF, &ip::GaborSpatial::setF)
    .add_property("theta", &ip::GaborSpatial::getTheta, &ip::GaborSpatial::setTheta)
    .add_property("gamma", &ip::GaborSpatial::getGamma, &ip::GaborSpatial::setGamma)
    .add_property("eta", &ip::GaborSpatial::getEta, &ip::GaborSpatial::setEta)
    .add_property("spatial_size", &ip::GaborSpatial::getSpatialSize, &ip::GaborSpatial::setSpatialSize)
    .add_property("cancel_dc", &ip::GaborSpatial::getCancelDc, &ip::GaborSpatial::setCancelDc)
    .add_property("norm_option", &ip::GaborSpatial::getNormOption, &ip::GaborSpatial::setNormOption)
    .add_property("border_option", &ip::GaborSpatial::getBorderOption, &ip::GaborSpatial::setBorderOption)
    .add_property("kernel", make_function(&ip::GaborSpatial::getKernel, return_value_policy<copy_const_reference>()))
    .def("__call__", (void (ip::GaborSpatial::*)(const blitz::Array<std::complex<double>,2>&, blitz::Array<std::complex<double>,2>&))&ip::GaborSpatial::operator(), (arg("input"), arg("output")), "Call an object of this type to filter an image.")
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
    .def("__call__", (void (ip::GaborFrequency::*)(const blitz::Array<std::complex<double>,2>&, blitz::Array<std::complex<double>,2>&))&ip::GaborFrequency::operator(), (arg("input"), arg("output")), "Call an object of this type to filter an image.")
    ;

  class_<ip::GaborBankSpatial, boost::shared_ptr<ip::GaborBankSpatial> >("GaborBankSpatial", gabor_bank_spatial_doc, init<optional<const int, const int, const double, const bool, const double, const double, const double, const double, const int, const bool, const enum Torch::ip::Gabor::NormOption, const enum Torch::sp::Convolution::BorderOption> >((arg("n_orient"), arg("n_freq"), arg("fmax")=0.25, arg("orientation_full")=false, arg("k")=1.414, arg("p")=0.5, arg("gamma")=1., arg("eta")=1., arg("spatial_size")=35, arg("cancel_dc")=false, arg("norm")=ip::Gabor::SpatialFactor, arg("border_opt")=Torch::sp::Convolution::Mirror), "Constructs a new Gabor filter bank in the spatial domain."))
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
    .def("__call__", (void (ip::GaborBankSpatial::*)(const blitz::Array<std::complex<double>,2>&, blitz::Array<std::complex<double>,3>&))&ip::GaborBankSpatial::operator(), (arg("input"), arg("output")), "Call an object of this type to filter an image.")
    ;

  class_<ip::GaborBankFrequency, boost::shared_ptr<ip::GaborBankFrequency> >("GaborBankFrequency", gabor_bank_frequency_doc, init<const int, const int, optional<const int, const int, const double, const bool, const double, const double, const bool, const double, const double, const double, const bool, const bool/*, const bool*/> >((arg("height"), arg("width"), arg("n_orient")=8, arg("n_freq")=5, arg("fmax")=0.25, arg("orientation_full")=false, arg("k")=1.414, arg("p")=0.5, arg("optimal_gamma_eta")=false, arg("gamma")=1., arg("eta")=1., arg("pf")=0.99, arg("cancel_dc")=false, arg("use_envelope")=false/*, arg("output_in_frequency")=false*/), "Constructs a new Gabor filter in the frequency domain."))
    .add_property("height", &ip::GaborBankFrequency::getHeight, &ip::GaborBankFrequency::setHeight)
    .add_property("width", &ip::GaborBankFrequency::getWidth, &ip::GaborBankFrequency::setWidth)
    .add_property("n_orient", &ip::GaborBankFrequency::getNOrient, &ip::GaborBankFrequency::setNOrient)
    .add_property("n_freq", &ip::GaborBankFrequency::getNFreq, &ip::GaborBankFrequency::setNFreq)
    .add_property("fmax", &ip::GaborBankFrequency::getFmax, &ip::GaborBankFrequency::setFmax)
    .add_property("orientation_full", &ip::GaborBankFrequency::getOrientationFull, &ip::GaborBankFrequency::setOrientationFull)
    .add_property("k", &ip::GaborBankFrequency::getK, &ip::GaborBankFrequency::setK)
    .add_property("p", &ip::GaborBankFrequency::getP, &ip::GaborBankFrequency::setP)
//    .add_property("optimal_gamma_eta", &ip::GaborBankFrequency::getOptimalGammaEta, &ip::GaborBankFrequency::setOptimalGammaEta)
    .add_property("gamma", &ip::GaborBankFrequency::getGamma, &ip::GaborBankFrequency::setGamma)
    .add_property("eta", &ip::GaborBankFrequency::getEta, &ip::GaborBankFrequency::setEta)
    .add_property("pf", &ip::GaborBankFrequency::getPf, &ip::GaborBankFrequency::setPf)
    .add_property("cancel_dc", &ip::GaborBankFrequency::getCancelDc, &ip::GaborBankFrequency::setCancelDc)
    .add_property("use_envelope", &ip::GaborBankFrequency::getUseEnvelope, &ip::GaborBankFrequency::setUseEnvelope)
    .def("__call__", (void (ip::GaborBankFrequency::*)(const blitz::Array<std::complex<double>,2>&, blitz::Array<std::complex<double>,3>&))&ip::GaborBankFrequency::operator(), (arg("input"), arg("output")), "Call an object of this type to filter an image.")
    ;


}
