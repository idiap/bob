/**
 * @file bob/ip/WeightedGaussian.h
 * @date Sat Apr 30 17:52:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to smooth an image with a Weighted
 *        Gaussian kernel (used by the Self Quotient Image)
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

#ifndef BOB_IP_WEIGHTED_GAUSSIAN_H
#define BOB_IP_WEIGHTED_GAUSSIAN_H

#include "bob/core/assert.h"
#include "bob/core/cast.h"
#include "bob/sp/extrapolate.h"
#include "bob/ip/integral.h"

namespace bob {

  /**
    * \ingroup libip_api
    * @{
    *
    */
  namespace ip {

    /**
      * @brief This class allows to smooth images with a weighted Gaussian
      *        kernel (used by the Self Quotient Image)
      */
    class WeightedGaussian
    {
      public:
        /**
          * @brief Creates an object to smooth images with a weighted Gaussian
          *        kernel
          * @param radius_y The radius-height of the kernel along the y-axis
          *                 (height = 2*radius_y + 1)
          * @param radius_x The radius-width of the kernel along the x-axis
          *                 (width = 2*radius_x + 1)
          * @param sigma2_y The variance of the kernel along the y-axis
          * @param sigma2_x The variance of the kernel along the x-axis
          * @param border_type The interpolation type for the convolution
          */
        WeightedGaussian(const size_t radius_y=1, const size_t radius_x=1,
            const double sigma2_y=2., const double sigma2_x=2.,
            const bob::sp::Extrapolation::BorderType border_type =
              bob::sp::Extrapolation::Mirror):
          m_radius_y(radius_y), m_radius_x(radius_x), m_sigma2_y(sigma2_y),
          m_sigma2_x(sigma2_x), m_conv_border(border_type)
        {
          computeKernel();
        }

        /**
          * @brief Copy constructor
          */
        WeightedGaussian(const WeightedGaussian& other):
          m_radius_y(other.m_radius_y), m_radius_x(other.m_radius_x),
          m_sigma2_y(other.m_sigma2_y), m_sigma2_x(other.m_sigma2_x),
          m_conv_border(other.m_conv_border)
        {
          computeKernel();
        }

        /**
          * @brief Destructor
          */
        virtual ~WeightedGaussian() {}

        /**
          * @brief Assignment operator
          */
        WeightedGaussian& operator=(const WeightedGaussian& other);

        /**
          * @brief Equal to
          */
        bool operator==(const WeightedGaussian& b) const;
        /**
          * @brief Not equal to
          */
        bool operator!=(const WeightedGaussian& b) const;

        /**
          * @brief Resets the parameters of the filter
          * @param radius_y The radius-height of the kernel along the y-axis
          *                 (height = 2*radius_y + 1)
          * @param radius_x The radius-width of the kernel along the x-axis
          *                 (width = 2*radius_x + 1)
          * @param sigma2_y The variance of the kernel along the y-axis
          * @param sigma2_x The variance of the kernel along the x-axis
          * @param border_type The interpolation type for the convolution
          */
        void reset( const size_t radius_y=1, const size_t radius_x=1,
          const double sigma2_y=2., const double sigma2_x=2.,
          const bob::sp::Extrapolation::BorderType border_type =
            bob::sp::Extrapolation::Mirror);

        /**
          * @brief Getters
          */
        size_t getRadiusY() const { return m_radius_y; }
        size_t getRadiusX() const { return m_radius_x; }
        double getSigma2Y() const { return m_sigma2_y; }
        double getSigma2X() const { return m_sigma2_x; }
        bob::sp::Extrapolation::BorderType getConvBorder() const { return m_conv_border; }
        const blitz::Array<double,2>& getUnweightedKernel() const { return m_kernel; }

        /**
          * @brief Setters
          */
        void setRadiusY(const size_t radius_y)
        { m_radius_y = radius_y; computeKernel(); }
        void setRadiusX(const size_t radius_x)
        { m_radius_x = radius_x; computeKernel(); }
        void setSigma2Y(const double sigma2_y)
        { m_sigma2_y = sigma2_y; computeKernel(); }
        void setSigma2X(const double sigma2_x)
        { m_sigma2_x = sigma2_x; computeKernel(); }
        void setConvBorder(const bob::sp::Extrapolation::BorderType border_type)
        { m_conv_border = border_type; }

        /**
          * @brief Process a 2D blitz Array/Image
          * @param src The 2D input blitz array
          * @param dst The 2D output blitz array
          */
        template <typename T>
        void operator()(const blitz::Array<T,2>& src,
          blitz::Array<double,2>& dst);

        /**
          * @brief Process a 3D blitz Array/Image
          * @param src The 3D input blitz array
          * @param dst The 3D output blitz array
          */
        template <typename T>
        void operator()(const blitz::Array<T,3>& src,
          blitz::Array<double,3>& dst);

      private:
        void computeKernel();

        /**
          * @brief Attributes
          */
        size_t m_radius_y;
        size_t m_radius_x;
        double m_sigma2_y;
        double m_sigma2_x;
        bob::sp::Extrapolation::BorderType m_conv_border;

        blitz::Array<double,2> m_kernel;
        blitz::Array<double,2> m_kernel_weighted;

        blitz::Array<double,2> m_src_extra;
        blitz::Array<double,2> m_src_integral;
    };

    // Declare template method full specialization
    template <>
    void WeightedGaussian::operator()<double>(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst);

    template <typename T>
    void WeightedGaussian::operator()(const blitz::Array<T,2>& src,
      blitz::Array<double,2>& dst)
    {
      // Casts the input to double
      blitz::Array<double,2> src_d = bob::core::array::cast<double>(src);
      // Calls the specialized template function for double
      this->operator()(src_d, dst);
    }

    template <typename T>
    void WeightedGaussian::operator()(const blitz::Array<T,3>& src,
      blitz::Array<double,3>& dst)
    {
      // Check number of planes
      bob::core::array::assertSameDimensionLength(src.extent(0), dst.extent(0));

      for( int p=0; p<dst.extent(0); ++p)
      {
        const blitz::Array<T,2> src_slice =
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<double,2> dst_slice =
          dst( p, blitz::Range::all(), blitz::Range::all() );

        // Weighted Gaussian smooth plane
        this->operator()(src_slice, dst_slice);
      }
    }

  }
}

#endif
