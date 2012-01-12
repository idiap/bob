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

        void transform(
          const blitz::Array<std::complex<double>,2>& _frequency_domain_image,
          blitz::Array<std::complex<double>,2>& transformed_frequency_domain_image
        ) const;

      private:
        // the Gabor wavelet, stored as pairs of indices and values
        std::vector<std::pair<blitz::TinyVector<int,2>, double> > m_kernel_pixel;

        int m_x_resolution, m_y_resolution;

    }; // class GaborKernel


    class GaborWaveletTransform {

      public:

        GaborWaveletTransform(
          int number_of_scales = 5,
          int number_of_orientations = 8,
          double sigma = 2. * M_PI,
          double k_max = M_PI / 2.,
          double k_fac = 1./sqrt(2.)
        );

        void generateKernels(std::pair<int, int> resolution);

        //! performs Gabor wavelet transform and returns vector of complex images
        void gwt(
          blitz::Array<std::complex<double>,3>& trafo_image,
          const blitz::Array<std::complex<double>,2>& gray_image);
        //! shortcut for gray images of any type
        template <typename T>
          void gwt(
            blitz::Array<std::complex<double>,3>& trafo_image,
            const blitz::Array<T,2>& gray_image
          ){
            return gwt(trafo_image, bob::core::cast<std::complex<double> >(gray_image));
          }
        //! shortcut for color images
        template <typename T>
          void gwt(
            blitz::Array<std::complex<double>,3>& trafo_image,
            const blitz::Array<T,3>& color_image
          ){
            // create gray image
            blitz::Array<T,2> gray_image(color_image.extend(1), color_image.extend(2));
            bob::ip::rgb_to_gray(color_image, gray_image);
            // call gray image function
            return gwt(trafo_image, bob::core::cast<std::complex<double> >(gray_image));
          }

        //! performs Gabor wavelet transform and returns pair of jet images
        // (absolute part and phase part)
        void jetImage(
          blitz::Array<double,4>& jet_image,
          const blitz::Array<std::complex<double>,2>& gray_image,
          bool do_normalize);
        //! shortcut for any type of gray image
        template <typename T>
          void jetImage(
            blitz::Array<double,4>& jet_image,
            const blitz::Array<T,2>& gray_image,
            bool do_normalize = true
          ){
            return jetImage(jet_image, bob::core::cast<std::complex<double> >(gray_image), do_normalize);
          }
        //! shortcut for color images
        template <typename T>
          void jetImage(
            blitz::Array<double,4>& jet_image,
            const blitz::Array<T,3>& color_image,
            bool do_normalize = true
          ){
            // create gray image
            blitz::Array<T,2> gray_image(color_image.extend(1), color_image.extend(2));
            bob::ip::rgb_to_gray(color_image, gray_image);
            // call gray image function
            return jetImage(jet_image, bob::core::cast<std::complex<double> >(gray_image), do_normalize);
          }

      private:

        double m_sigma;
        std::vector<GaborKernel> m_gabor_kernels;

        std::vector<std::pair<double, double> > m_kernel_frequencies;

        bob::sp::FFT2D m_fft;
        bob::sp::IFFT2D m_ifft;

        blitz::Array<std::complex<double>,2> m_temp_array, m_frequency_image;

    }; // class GaborWaveletTransform

  } // namespace ip

} // namespace bob



