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
#include <core/array_type.h>

#include <ip/GaborWaveletTransform.h>
#include <sp/FFT2D.h>
#include <blitz/array.h>

#include <core/cast.h>
#include <ip/color.h>


template <class T> 
static inline blitz::Array<std::complex<double>,2> complex_cast(bob::python::const_ndarray input){
  blitz::Array<T,2> gray(input.type().shape[1],input.type().shape[2]);
  bob::ip::rgb_to_gray(input.bz<T,3>(), gray);
  return bob::core::cast<std::complex<double> >(gray);
}

static inline const blitz::Array<std::complex<double>, 2> convert_image(bob::python::const_ndarray input){
  if (input.type().nd == 3){
    // perform color type conversion
    switch (input.type().dtype){
      case bob::core::array::t_uint8: return complex_cast<uint8_t>(input);
      case bob::core::array::t_uint16: return complex_cast<uint16_t>(input);
      case bob::core::array::t_float64: return complex_cast<double>(input);
      default: throw bob::core::Exception();
    }
  } else {
    switch (input.type().dtype){
      case bob::core::array::t_uint8: return bob::core::cast<std::complex<double> >(input.bz<uint8_t,2>());
      case bob::core::array::t_uint16: return bob::core::cast<std::complex<double> >(input.bz<uint16_t,2>());
      case bob::core::array::t_float64: return bob::core::cast<std::complex<double> >(input.bz<double,2>());
      case bob::core::array::t_complex128: return input.bz<std::complex<double>,2>();
      default: throw bob::core::Exception();
    }
  }
}

static inline void transform(bob::ip::GaborKernel& kernel, blitz::Array<std::complex<double>,2>& input, blitz::Array<std::complex<double>,2>& output){
 // perform fft on input image
  bob::sp::FFT2D fft(input.extent(0), input.extent(1));
  fft(input);

  // apply the kernel in frequency domain
  kernel.transform(input, output);
  
  // perform ifft on the result
  bob::sp::IFFT2D ifft(output.extent(0), output.extent(1));
  ifft(output);
}

static boost::python::object gabor_wavelet_transform_1(bob::ip::GaborKernel& kernel, bob::python::const_ndarray input_image){
  // convert input ndarray to complex blitz array
  blitz::Array<std::complex<double>,2> input = convert_image(input_image);
  // allocate output array
  bob::python::ndarray result(bob::core::array::t_complex128, input.extent(0), input.extent(1));
  blitz::Array<std::complex<double>,2> output = result.bz<std::complex<double>,2>();
  
  // transform input to output
  transform(kernel, input, output);
  
  // return the py_object
  return result.self();
}

static void gabor_wavelet_transform_2(bob::ip::GaborKernel& kernel, bob::python::const_ndarray input_image, bob::python::ndarray output_image){
  // convert input image into complex type
  blitz::Array<std::complex<double>,2> input = convert_image(input_image);
  // cast output image to complex type
  blitz::Array<std::complex<double>,2> output = output_image.bz<std::complex<double>,2>();
  // transform input to output
  transform(kernel, input, output);
}

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
  // bind Gabor Kernel class
  boost::python::class_<bob::ip::GaborKernel, boost::shared_ptr<bob::ip::GaborKernel> >(
    "GaborKernel",
    "This class can be used to filter an image with a single Gabor wavelet.",
    boost::python::no_init
  )
  
  .def(
    boost::python::init< const blitz::TinyVector<int,2>&, const blitz::TinyVector<double,2>&, boost::python::optional <const double, const double, const bool, const double> >(
      (
        boost::python::arg("resolution"),
        boost::python::arg("wavelet_frequency"),
        boost::python::arg("sigma") = 2. * M_PI,
        boost::python::arg("pow_of_k") = 0.,
        boost::python::arg("dc_free") = true,
        boost::python::arg("epsilon") = 1e-10
      ),
      "Initializes the Gabor wavelet of the given wavelet frequency to be used as a filter for the given image resolution. The optional parameters can be changed, but have useful default values."
    )
  )
  
  .def(
    "__call__",
    &gabor_wavelet_transform_1,
    (boost::python::arg("self"), boost::python::arg("input_image")),
    """This function Gabor-filters the given input_image, which can be of any type. The output image is of complex type. It will be automatically generated and returned."""
  )

  .def(
    "__call__",
    &gabor_wavelet_transform_2,
    (boost::python::arg("self"), boost::python::arg("input_image"), boost::python::arg("output_image")),
    """This function Gabor-filters the given input_image, which can be of any type, to the output image. The output image needs to have the same resolution as the input image and must be of complex type."""
  );

    

  // declare GWT class
  boost::python::class_<bob::ip::GaborWaveletTransform, boost::shared_ptr<bob::ip::GaborWaveletTransform> >(
    "GaborWaveletTransform",
    "This class can be used to perform a Gabor wavelet transform from one image to an image of (normalized) Gabor jets or to a complex-valued multi-layer trafo image.",
    boost::python::no_init
  )

  .def(
    boost::python::init<boost::python::optional<int,int,double,double,double,double,bool> >(
      (
        boost::python::arg("number_of_scales") = 5,
        boost::python::arg("number_of_angles") = 8,
        boost::python::arg("sigma") = 2. * M_PI,
        boost::python::arg("k_max") = M_PI / 2.,
        boost::python::arg("k_fac") = 1./sqrt(2.),
        boost::python::arg("pow_of_k") = 0.,
        boost::python::arg("dc_free") = true
      ),
      "Initializes the Gabor wavelet transform by generating Gabor wavelets in number_of_scales different frequencies and number_of_angles different directions. The remaining parameters are parameters of the Gabor wavelets to be generated. "
    )
  )

  .add_property(
    "number_of_kernels",
    &bob::ip::GaborWaveletTransform::numberOfKernels,
    "The number of Gabor wavelets (i.e. number of directions times number of scales, i.e. the length of a Gabor jet, i.e. the number of layers of the trafo image) used in this Gabor wavelet transform."
  )

  .add_property(
    "number_of_scales",
    &bob::ip::GaborWaveletTransform::m_number_of_scales,
    "The number of scales that this Gabor wavelet family holds."
  )

  .add_property(
    "number_of_directions",
    &bob::ip::GaborWaveletTransform::m_number_of_directions,
    "The number of directions that this Gabor wavelet family holds."
  )

  .def(
    "perform_gwt",
    &perform_gwt,
    (boost::python::arg("self"), boost::python::arg("input_image"), boost::python::arg("output_trafo_image")),
    "Performs a Gabor wavelet transform and fills the given Gabor wavelet transformed image (output_trafo_image)"
  )

  .def(
    "compute_jet_image",
    &compute_jet_image,
    (boost::python::arg("self"), boost::python::arg("input_image"), boost::python::arg("output_jet_image"), boost::python::arg("normalized")=true),
    "Performs a Gabor wavelet transform and fills given image of Gabor jets. If the normalized parameter is set to True (the default), the absolute parts of the Gabor jets are normalized to unit Euclidean lenght."
  );

  boost::python::def(
    "normalize_gabor_jet",
    &normalize_gabor_jet,
    (boost::python::arg("gabor_jet")),
    "Normalizes the Gabor jet (with or without phase) to unit Euclidean length."
  );
}
