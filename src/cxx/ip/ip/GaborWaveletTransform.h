/**
 * @file cxx/ip/ip/GaborWaveletTransform.h
 * @date Tue Jan 10 19:52:29 2012 +0200
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief This file provides a class that performs a Gabor wavelet
 * transform.
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

#ifndef TORCH5SPRO_IP_GABOR_WAVELET_TRANSFORM_H
#define TORCH5SPRO_IP_GABOR_WAVELET_TRANSFORM_H

#endif // TORCH5SPRO_IP_GABOR_WAVELET_TRANSFORM_H

#include <blitz/array.h>
#include <ip/color.h>
#include <sp/FFT2D.h>
#include <io/Array.h>
#include <vector>
#include <utility>

namespace bob {

  namespace ip {

    class GaborKernel {

      public:

        GaborKernel(
          std::pair<int, int> resolution,
          std::pair<double, double> wavelet_frequency,
          double sigma = 2. * M_PI,
          double epsilon = 1e-5,
          bool dc_free = true
        );

        blitz::Array<double,2> kernelImage() const;

        void transform(
          const blitz::Array<std::complex<double>,2>& _frequency_domain_image,
          blitz::Array<std::complex<double>,2>& transformed_frequency_domain_image
        ) const;

      private:
        // the Gabor wavelet, stored as pairs of indices and values
        std::vector<std::pair<blitz::TinyVector<int,2>, double> > m_kernel_pixel;

        int m_x_resolution, m_y_resolution;

    }; // class GaborKernel


    //! \brief The GaborWaveletTransform class computes a Gabor wavelet transform of the given image.
    //! It computes either the complete Gabor wavelet transformed image (short: trafo image) with
    //! number_of_scales * number_of_orientations layers, or a Gabor jet image that includes
    //! one Gabor jet (with one vector of absolute values and one vector of phases) for each pixel
    class GaborWaveletTransform {

      public:

        //! \brief Constructs a Gabor wavelet transform object.
        //! This class will generate number_of_scales * number_of_orientations Gabor wavelets
        //! using the given sigma, k_max and k_fac values
        //! All parameters have reasonable defaults, as used by default algorithms
        GaborWaveletTransform(
          int number_of_scales = 5,
          int number_of_directions = 8,
          double sigma = 2. * M_PI,
          double k_max = M_PI / 2.,
          double k_fac = 1./sqrt(2.)
        );

        //! generate the kernels for the new resolution
        //! this function is called internally, no need to call it explicitly
        void generateKernels(std::pair<int, int> resolution);

        //! generates the frequency kernels as images
        blitz::Array<double,3> kernelImages() const;

        //! get the number of kernels (usually, 40) used by this GWT class
        int numberOfKernels() const{return m_kernel_frequencies.size();}

        const std::vector<std::pair<double, double> >& kernelFrequencies() const {return m_kernel_frequencies;}

        //! performs Gabor wavelet transform and returns vector of complex images
        void performGWT(
          const blitz::Array<std::complex<double>,2>& gray_image,
          blitz::Array<std::complex<double>,3>& trafo_image
        );

        //! performs Gabor wavelet transform and creates 4D image
        //! (absolute part and phase part)
        //! If do_normalize is enabled, the Gabor jets are normalized to length 1
        //! (which is a good choice for most applications,
        //! but e.g. not required to compute LGBP).
        void computeJetImage(
          const blitz::Array<std::complex<double>,2>& gray_image,
          blitz::Array<double,4>& jet_image,
          bool do_normalize = true
        );

      private:

        double m_sigma;
        std::vector<GaborKernel> m_gabor_kernels;

        std::vector<std::pair<double, double> > m_kernel_frequencies;

        bob::sp::FFT2D m_fft;
        bob::sp::IFFT2D m_ifft;

        blitz::Array<std::complex<double>,2> m_temp_array, m_frequency_image;

      public:
        const int m_number_of_scales;
        const int m_number_of_directions;
    }; // class GaborWaveletTransform

  } // namespace ip

} // namespace bob



