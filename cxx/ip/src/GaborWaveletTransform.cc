/**
 * @file cxx/ip/src/GaborWaveletTransform.cc
 * @date 2012-02-27
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief This file provides a class to perform a Gabor wavelet transform.
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

#include "core/array_assert.h"
#include "core/array_copy.h"
#include "ip/GaborWaveletTransform.h"
#include <numeric>
#include <sstream>
#include <fstream>

static inline double sqr(double x){return x*x;}

/**
 * Generates a Gabor kernel.
 * @param resolution The resolution of the image to generate
 * @param k  The frequency vector (i.e. the center of the Gaussian in frequency domain)
 * @param sigma  The standard deviation (i.e. the width of the Gabor wavelet)
 * @param pow_of_k  The power of \f$ k^x \f$ used as a prefactor of the Gabor wavelet
 * @param dc_free   Make the Gabor wavelet DC-free?
 * @param epsilon   The epsilon value below which the wavelet value is considered as zero
 */
bob::ip::GaborKernel::GaborKernel(
  const blitz::TinyVector<int,2>& resolution,
  const blitz::TinyVector<double,2>& k,
  const double sigma,
  const double pow_of_k,
  const bool dc_free,
  const double epsilon
)
: m_x_resolution(resolution[1]),
  m_y_resolution(resolution[0])
{
  // create Gabor wavelet with given parameters
  int32_t start_x = - (int)m_x_resolution / 2, start_y = - (int)m_y_resolution / 2;
  // take care of odd resolutions in the end points
  int32_t end_x = m_x_resolution / 2 + m_x_resolution % 2, end_y = m_y_resolution / 2 + m_y_resolution % 2;

  double k_x_factor = 2. * M_PI / m_x_resolution, k_y_factor = 2. * M_PI / m_y_resolution;
  double kx = k[1], ky = k[0];

  // iterate over all pixels of the images
  for (int y = start_y; y < end_y; ++y){

    // convert relative pixel coordinate into frequency coordinate
    double omega_y = y * k_y_factor;

    for (int x = start_x; x < end_x; ++x){

      // convert relative pixel coordinate into frequency coordinate
      double omega_x = x * k_x_factor;

      // compute value of frequency kernel function
      double omega_minus_k_squared = sqr(omega_x - kx) + sqr(omega_y - ky);
      double sigma_square = sqr(sigma);
      double k_square = sqr(kx) + sqr(ky);
      // assign kernel value
      double wavelet_value = exp(- sigma_square * omega_minus_k_squared / (2. * k_square));

      // prefactor the wavelet value with k^(pow_of_k); the default prefactor 1 might not be the best.
      wavelet_value *= std::pow(k_square, pow_of_k / 2.);

      if (dc_free){
        double omega_square = sqr(omega_x) + sqr(omega_y);

        wavelet_value -= exp(-sigma_square * (omega_square + k_square) / (2. * k_square));
      } // if ! dc_free

      if (std::abs(wavelet_value) > epsilon){
        m_kernel_pixel.push_back(std::make_pair(
          blitz::TinyVector<int,2>((y + m_y_resolution) % m_y_resolution, (x + m_x_resolution) % m_x_resolution),
          wavelet_value
        ));
      }
    } // for x
  } // for y
}

/**
 * Performs the convolution of the given image with this Gabor kernel.
 * Please note that both the inpus as well as the output image are in frequency domain.
 * @param frequency_domain_image
 * @param transformed_frequency_domain_image
 */
void bob::ip::GaborKernel::transform(
  const blitz::Array<std::complex<double>,2>& frequency_domain_image,
  blitz::Array<std::complex<double>,2>& transformed_frequency_domain_image
) const
{
  // assert same size
  bob::core::array::assertSameShape(frequency_domain_image, transformed_frequency_domain_image);
  // clear resulting image first
  transformed_frequency_domain_image = std::complex<double>(0);
  // iterate through the kernel pixels and do the multiplication
  std::vector<std::pair<blitz::TinyVector<int,2>, double> >::const_iterator it = m_kernel_pixel.begin(), it_end = m_kernel_pixel.end();
  for (; it < it_end; ++it){
    transformed_frequency_domain_image(it->first) = frequency_domain_image(it->first) * it->second;
  }
}

/**
 * Generates and returns the image for the current kernel.
 * @return The kernel image in frequency domain.
 */
blitz::Array<double,2> bob::ip::GaborKernel::kernelImage() const{
  blitz::Array<double,2> image(m_y_resolution, m_x_resolution);
  image = 0;
  // iterate through the kernel pixels
  std::vector<std::pair<blitz::TinyVector<int,2>, double> >::const_iterator it = m_kernel_pixel.begin(), it_end = m_kernel_pixel.end();
  for (; it < it_end; ++it){
    image(it->first) = it->second;
  }
  return image;
}

/***********************************************************************************
******************     GaborWaveletTransform      **********************************
***********************************************************************************/
/**
 * Initializes a discrete family of Gabor wavelets
 * @param number_of_scales     The number of scales (frequencies) to generate
 * @param number_of_directions The number of directions (orientations) to generate
 * @param sigma                The width (standard deviation) of the Gabor wavelet
 * @param k_max                The highest frequency to generate (maximum: PI)
 * @param k_fac                The logarithmical factor between two scales of Gabor wavelets; should be below one
 * @param pow_of_k             The power of k for the prefactor
 * @param dc_free              Make the Gabor wavelet DC-free?
 */
bob::ip::GaborWaveletTransform::GaborWaveletTransform(
  int number_of_scales,
  int number_of_directions,
  double sigma,
  double k_max,
  double k_fac,
  double pow_of_k,
  bool dc_free
)
: m_sigma(sigma),
  m_pow_of_k(pow_of_k),
  m_dc_free(dc_free),
  m_fft(0,0),
  m_ifft(0,0),
  m_number_of_scales(number_of_scales),
  m_number_of_directions(number_of_directions)
{
  // reserve enough space
  m_kernel_frequencies.reserve(number_of_scales * number_of_directions);
  // initialize highest frequency
  double k_abs = k_max;
  // iterate over the scales
  for (int s = 0; s < number_of_scales; ++s){

    // iterate over the directions
    for (int d = 0; d < number_of_directions; ++d )
    {
      double angle = M_PI * d / number_of_directions;
      // compute center of kernel in frequency domain in Cartesian coordinates
      m_kernel_frequencies.push_back(
        blitz::TinyVector<double,2>(k_abs * sin(angle), k_abs * cos(angle)));
    } // for d

    // move to the next frequency scale
    k_abs *= k_fac;
  } // for s
}


/**
 * Generates the kernels for the given image resolution.
 * This function dose not need to be called explicitly to be able to perform the GWT.
 * @param resolution  The resolution of the image to generate the kernels for
 */
void bob::ip::GaborWaveletTransform::generateKernels(
  blitz::TinyVector<int,2> resolution
)
{
  if (resolution[1] != m_fft.getWidth() || resolution[0] != m_fft.getHeight()){
    // new kernels need to be generated
    m_gabor_kernels.clear();
    m_gabor_kernels.reserve(m_kernel_frequencies.size());

    for (int j = 0; j < (int)m_kernel_frequencies.size(); ++j){
      m_gabor_kernels.push_back(bob::ip::GaborKernel(resolution, m_kernel_frequencies[j], m_sigma, m_pow_of_k, m_dc_free));
    }

    // reset fft sizes
    m_fft.reset(resolution[0], resolution[1]);
    m_ifft.reset(resolution[0], resolution[1]);
    m_temp_array.resize(blitz::shape(resolution[0],resolution[1]));
    m_frequency_image.resize(m_temp_array.shape());
  }
}

/**
 * Generates and returns the images of the Gabor wavelet family in frequency domain.
 * @return  The Gabor wavelets (one per layer)
 */
blitz::Array<double,3> bob::ip::GaborWaveletTransform::kernelImages() const{
  // generate array of desired size
  blitz::Array<double,3> res(m_gabor_kernels.size(), m_temp_array.shape()[0], m_temp_array.shape()[1]);
  // fill in the wavelets
  for (int j = m_gabor_kernels.size(); j--;){
    res(j, blitz::Range::all(), blitz::Range::all()) = m_gabor_kernels[j].kernelImage();
  }
  return res;
}

/**
 * Computes the Gabor wavelet transformation for the given image (in spatial domain)
 * @param gray_image  The source image in spatial domain
 * @param trafo_image The convolution result, in spatial domain
 */
void bob::ip::GaborWaveletTransform::performGWT(
  const blitz::Array<std::complex<double>,2>& gray_image,
  blitz::Array<std::complex<double>,3>& trafo_image
)
{
  // first, check if we need to reset the kernels
  generateKernels(blitz::TinyVector<int,2>(gray_image.extent(0),gray_image.extent(1)));

  // perform Fourier transformation to image
  m_fft(gray_image, m_frequency_image);

  // check that the shape is correct
  bob::core::array::assertSameShape(trafo_image, blitz::shape(m_kernel_frequencies.size(),gray_image.extent(0),gray_image.extent(1)));

  // now, let each kernel compute the transformation result
  for (int j = 0; j < (int)m_gabor_kernels.size(); ++j){
    // get a reference to the current layer of the trafo image
    m_gabor_kernels[j].transform(m_frequency_image, m_temp_array);
    // perform ifft on the trafo image layer
    blitz::Array<std::complex<double>,2> layer(trafo_image(j, blitz::Range::all(), blitz::Range::all()));
    m_ifft(m_temp_array, layer);
  } // for j
}

/**
 * Computes the Gabor jets including absolute values and phases for the given image (in spatial domain).
 * @param gray_image  The source image in spatial domain
 * @param jet_image   The resulting Gabor jet image, including absolute values and phases for each pixel
 * @param do_normalize Shall the Gabor jets be normalized?
 */
void bob::ip::GaborWaveletTransform::computeJetImage(
  const blitz::Array<std::complex<double>,2>& gray_image,
  blitz::Array<double,4>& jet_image,
  bool do_normalize
)
{
  // first, check if we need to reset the kernels
  generateKernels(blitz::TinyVector<int,2>(gray_image.extent(0),gray_image.extent(1)));

  // perform Fourier transformation to image
  m_fft(gray_image, m_frequency_image);

  // check that the shape is correct
  bob::core::array::assertSameShape(jet_image, blitz::shape(gray_image.extent(0), gray_image.extent(1), 2, m_kernel_frequencies.size()));

  // now, let each kernel compute the transformation result
  for (int j = 0; j < (int)m_gabor_kernels.size(); ++j){
    // get a reference to the current layer of the trafo image
    m_gabor_kernels[j].transform(m_frequency_image, m_temp_array);
    // perform ifft of transformed image
    m_ifft(m_temp_array);
    // convert into absolute and phase part
    blitz::Array<double,2> abs_part(jet_image(blitz::Range::all(), blitz::Range::all(), 0, j));
    abs_part = blitz::abs(m_temp_array);
    blitz::Array<double,2> phase_part(jet_image(blitz::Range::all(), blitz::Range::all(), 1, j));
    phase_part = blitz::arg(m_temp_array);
  } // for j

  if (do_normalize){
    // iterate the positions
    for (int y = jet_image.extent(0); y--;){
      for (int x = jet_image.extent(1); x--;){
        // normalize jet
        blitz::Array<double,2> jet(jet_image(y,x,blitz::Range::all(),blitz::Range::all()));
        bob::ip::normalizeGaborJet(jet);
      }
    }
  }
}

/**
 * Computes the Gabor jets including absolute values only for the given image (in spatial domain).
 * @param gray_image  The source image in spatial domain
 * @param jet_image   The resulting Gabor jet image, including only absolute values for each pixel
 * @param do_normalize Shall the Gabor jets be normalized?
 */
void bob::ip::GaborWaveletTransform::computeJetImage(
  const blitz::Array<std::complex<double>,2>& gray_image,
  blitz::Array<double,3>& jet_image,
  bool do_normalize
)
{
  // first, check if we need to reset the kernels
  generateKernels(blitz::TinyVector<int,2>(gray_image.extent(0),gray_image.extent(1)));

  // perform Fourier transformation to image
  m_fft(gray_image, m_frequency_image);

  // check that the shape is correct
  bob::core::array::assertSameShape(jet_image, blitz::shape(gray_image.extent(0), gray_image.extent(1), m_kernel_frequencies.size()));

  // now, let each kernel compute the transformation result
  for (int j = 0; j < (int)m_gabor_kernels.size(); ++j){
    // get a reference to the current layer of the trafo image
    m_gabor_kernels[j].transform(m_frequency_image, m_temp_array);
    // perform ifft of transformed image
    m_ifft(m_temp_array);
    // convert into absolute part
    blitz::Array<double,2> abs_part(jet_image(blitz::Range::all(), blitz::Range::all(), j));
    abs_part = blitz::abs(m_temp_array);
  } // for j

  if (do_normalize){
    // iterate the positions
    for (int y = jet_image.extent(0); y--;){
      for (int x = jet_image.extent(1); x--;){
        // normalize jet
        blitz::Array<double,1> jet(jet_image(y,x,blitz::Range::all()));
        bob::ip::normalizeGaborJet(jet);
      }
    }
  }
}


/**
 * Normalizes the given Gabor jet (absolute values only) to unit length.
 * @param gabor_jet The Gabor jet to be normalized.
 */
void bob::ip::normalizeGaborJet(blitz::Array<double,1>& gabor_jet){
  double norm = sqrt(std::inner_product(gabor_jet.begin(), gabor_jet.end(), gabor_jet.begin(), 0.));
  // normalize the absolute parts of the jets
  gabor_jet /= norm;
}


/**
 * Normalizes the given Gabor jet to unit length.
 * @param gabor_jet The Gabor jet to be normalized, including the phase values (which will not be altered).
 */
void bob::ip::normalizeGaborJet(blitz::Array<double,2>& gabor_jet){
  blitz::Array<double,1> abs_jet = gabor_jet(0, blitz::Range::all());
  double norm = sqrt(std::inner_product(abs_jet.begin(), abs_jet.end(), abs_jet.begin(), 0.));
  // normalize the absolute parts of the jets
  abs_jet /= norm;
}
