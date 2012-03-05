/**
 * @file cxx/ip/ip/GaborWaveletTransform.h
 * @date 2012-02-27
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief This file provides a class that performs a Gabor wavelet transform,
 * as well as a class for the Gabor wavelets themselves.
 *
 * Copyright (C) 2011 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_IP_GABOR_WAVELET_TRANSFORM_H
#define BOB_IP_GABOR_WAVELET_TRANSFORM_H

#include <blitz/array.h>
#include <core/Exception.h>
#include <sp/FFT2D.h>
#include <vector>
#include <utility>

namespace bob {

  namespace ip {

    //! \brief This class represents a single Gabor wavelet in frequency domain.
    class GaborKernel {

      public:

        //! Generate a Gbaor kernel in frequency domain
        GaborKernel(
          blitz::TinyVector<int,2> resolution,
          blitz::TinyVector<double,2> wavelet_frequency,
          double sigma = 2. * M_PI,
          double pow_of_k = 0.,
          bool dc_free = true,
          double epsilon = 1e-10
        );

        //! Get the image represenation of the Gabor wavelet in frequency domain
        blitz::Array<double,2> kernelImage() const;

        //! Gabor transforms the given image
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
          double k_fac = 1./sqrt(2.),
          double pow_of_k = 0.,
          bool dc_free = true
        );

        //! generate the kernels for the new resolution
        void generateKernels(blitz::TinyVector<int,2> resolution);

        //! Returns the Gabor kernel for the given index
        const bob::ip::GaborKernel& getKernel(int index) {if (index < (int)m_gabor_kernels.size()) return m_gabor_kernels[index]; else throw bob::core::Exception();}

        //! generates the frequency kernels as images
        blitz::Array<double,3> kernelImages() const;

        //! get the number of kernels (usually, 40) used by this GWT class
        int numberOfKernels() const{return m_kernel_frequencies.size();}

        //! Returns the vector of central frequencies used by this Gabor wavelet family
        const std::vector<blitz::TinyVector<double,2> >& kernelFrequencies() const {return m_kernel_frequencies;}

        //! performs Gabor wavelet transform and returns vector of complex images
        void performGWT(
          const blitz::Array<std::complex<double>,2>& gray_image,
          blitz::Array<std::complex<double>,3>& trafo_image
        );

        //! \brief performs Gabor wavelet transform and creates 4D image
        //! (absolute part and phase part)
        void computeJetImage(
          const blitz::Array<std::complex<double>,2>& gray_image,
          blitz::Array<double,4>& jet_image,
          bool do_normalize = true
        );

        //! \brief performs Gabor wavelet transform and creates 3D image
        //! (absolute parts of the responses only)
        void computeJetImage(
          const blitz::Array<std::complex<double>,2>& gray_image,
          blitz::Array<double,3>& jet_image,
          bool do_normalize = true
        );


      private:

        double m_sigma;
        double m_pow_of_k;
        bool m_dc_free;
        std::vector<GaborKernel> m_gabor_kernels;

        std::vector<blitz::TinyVector<double,2> > m_kernel_frequencies;

        bob::sp::FFT2D m_fft;
        bob::sp::IFFT2D m_ifft;

        blitz::Array<std::complex<double>,2> m_temp_array, m_frequency_image;

      public:
        //! The number of scales (levels, frequencies) of this family
        const int m_number_of_scales;
        //! The number of directions (orientations) of this family
        const int m_number_of_directions;
    }; // class GaborWaveletTransform

    //! Normalizes a Gabor jet (vector of absolute values) to unit length
    void normalizeGaborJet(blitz::Array<double,1>& gabor_jet);

    //! Normalizes a Gabor jet (vector of absolute and phase values) to unit length
    void normalizeGaborJet(blitz::Array<double,2>& gabor_jet);

  } // namespace ip

} // namespace bob

#endif // BOB_IP_GABOR_WAVELET_TRANSFORM_H

