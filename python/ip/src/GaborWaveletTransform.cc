/**
 * @file python/ip/src/GaborWaveletTransform.cc
 * @date 2012-02-27
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Binds the Gabor wavelet transform
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
#include <core/python/ndarray.h>
#include <core/array_exception.h>

#include <ip/GaborWaveletTransform.h>

static void perform_gwt (bob::ip::GaborWaveletTransform& gwt, bob::python::const_ndarray input_image, bob::python::ndarray output_trafo_image){
  const blitz::Array<std::complex<double>,2> image = input_image.bz<std::complex<double>,2>();
  blitz::Array<std::complex<double>,3> trafo_image = output_trafo_image.bz<std::complex<double>,3>();
  gwt.performGWT(image, trafo_image);
}

static void compute_jet_image(bob::ip::GaborWaveletTransform& gwt, bob::python::const_ndarray input_image, bob::python::ndarray output_jet_image, bool normalized){
  const blitz::Array<std::complex<double>,2> image = input_image.bz<std::complex<double>,2>();

  if (output_jet_image.type().nd == 3){
    // compute jet image with absolute values only
    blitz::Array<double,3> jet_image = output_jet_image.bz<double,3>();
    gwt.computeJetImage(image, jet_image, normalized);
  } else if (output_jet_image.type().nd == 4){
    blitz::Array<double,4> jet_image = output_jet_image.bz<double,4>();
    gwt.computeJetImage(image, jet_image, normalized);
  } else throw bob::core::UnexpectedShapeError();
}

static void normalize_gabor_jet(bob::python::ndarray gabor_jet){
  if (gabor_jet.type().nd == 1){
    blitz::Array<double,1> jet(gabor_jet.bz<double,1>());
    bob::ip::normalizeGaborJet(jet);
  } else if (gabor_jet.type().nd == 2){
    blitz::Array<double,2> jet(gabor_jet.bz<double,2>());
    bob::ip::normalizeGaborJet(jet);
  } else throw bob::core::UnexpectedShapeError();
}

void bind_ip_gabor_wavelet_transform() {
  // declare GWT class
  boost::python::class_<bob::ip::GaborWaveletTransform, boost::shared_ptr<bob::ip::GaborWaveletTransform> >(
    "GaborWaveletTransform",
    "This class can be used to perform a Gabor wavelet transform from one image to an image of (normalized) Gabor jets",
    boost::python::no_init
  )

  .def(
    boost::python::init<int,int,double,double,double,double,bool>(
      (
        boost::python::arg("number_of_scales") = 5,
        boost::python::arg("number_of_angles") = 8,
        boost::python::arg("sigma") = 2. * M_PI,
        boost::python::arg("k_max") = M_PI / 2.,
        boost::python::arg("k_fac") = 1./sqrt(2.),
        boost::python::arg("pow_of_k") = 0.,
        boost::python::arg("dc_free") = true
      ),
      "Initializes the Gabor wavelet transform"
    )
  )

  .add_property(
    "number_of_kernels",
    &bob::ip::GaborWaveletTransform::numberOfKernels,
    "The number of Gabor wavelets (i.e. number of directions times number of scales, i.e. the size of the trafo image) used in this Gabor wavelet transform"
  )

  .add_property(
    "number_of_scales",
    &bob::ip::GaborWaveletTransform::m_number_of_scales,
    "The number of scales that this Gabor wavelet family holds"
  )

  .add_property(
    "number_of_directions",
    &bob::ip::GaborWaveletTransform::m_number_of_directions,
    "The number of directions that this Gabor wavelet family holds"
  )

  .def(
    "perform_gwt",
    &perform_gwt,
    (boost::python::arg("self"), boost::python::arg("input_image"), boost::python::arg("output_trafo_image")),
    "Performs a Gabor wavelet transform and returns the complete Gabor wavelet transformed image"
  )

  .def(
    "compute_jet_image",
    &compute_jet_image,
    (boost::python::arg("self"), boost::python::arg("input_image"), boost::python::arg("output_jet_image"), boost::python::arg("normalized")=true),
    "Performs a Gabor wavelet transform and returns the complete set of Gabor jets"
  );

  boost::python::def(
    "normalize_gabor_jet",
    &normalize_gabor_jet,
    (boost::python::arg("gabor_jet")),
    "Normalizes the Gabor jet (with or without phase) to unit length"
  );
}
