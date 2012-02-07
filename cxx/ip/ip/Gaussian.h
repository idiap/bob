/**
 * @file cxx/ip/ip/Gaussian.h
 * @date Sat Apr 30 17:52:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to smooth an image with a Gaussian kernel
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_IP_GAUSSIAN_H
#define BOB_IP_GAUSSIAN_H

#include "core/array_assert.h"
#include "core/cast.h"
#include "sp/conv.h"
#include "sp/extrapolate.h"

namespace bob {

	/**
	 * \ingroup libip_api
	 * @{
	 *
	 */
	namespace ip {

    /**
      * @brief This class allows to smooth images with a Gaussian kernel
      */
		class Gaussian
		{
  		public:
			  /**
  			 * @brief Creates an object to smooth images with a Gaussian kernel
	  		 * @param radius_y The height of the kernel along the y-axis
	  		 * @param radius_x The width of the kernel along the x-axis
         * @param sigma The standard deviation of the kernal
		  	 * @param size_opt The size of the output wrt. to convolution
		  	 * @param border_opt The interpolation type for the convolution
			   */
	  		Gaussian(const int radius_y=1, const int radius_x=1, 
            const double sigma_y=5., const double sigma_x=5.,
            const enum bob::sp::Extrapolation::BorderType border_type =
              bob::sp::Extrapolation::Mirror):
          m_radius_y(radius_y), m_radius_x(radius_x), m_sigma_y(sigma_y),
          m_sigma_x(sigma_x), m_conv_border(border_type)
  			{
          computeKernel();
        }

        void reset( const int radius_y=1, const int radius_x=1,
          const double sigma_y=5., const double sigma_x=5.,
          const enum bob::sp::Extrapolation::BorderType border_type =
            bob::sp::Extrapolation::Mirror);

        /**
         * @brief Process a 2D blitz Array/Image
         * @param src The 2D input blitz array
         * @param src The 2D input blitz array
         */
        template <typename T> 
        void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<double,2>& dst);

        /**
         * @brief Process a 3D blitz Array/Image
         * @param src The 3D input blitz array
         * @param src The 3D input blitz array
         */
        template <typename T> 
        void operator()(const blitz::Array<T,3>& src, 
          blitz::Array<double,3>& dst);

      private:
        void computeKernel(); 

        /**
         * @brief Attributes
         */	
        int m_radius_y;
        int m_radius_x;
        double m_sigma_y;
        double m_sigma_x;
        enum bob::sp::Extrapolation::BorderType m_conv_border;

        blitz::Array<double, 1> m_kernel_y;
        blitz::Array<double, 1> m_kernel_x;

        blitz::Array<double, 2> m_tmp_int;
        blitz::Array<double, 2> m_tmp_int1;
        blitz::Array<double, 2> m_tmp_int2;
    };

    // Declare template method full specialization
    template <> 
    void bob::ip::Gaussian::operator()<double>(const blitz::Array<double,2>& src, 
      blitz::Array<double,2>& dst);

    template <typename T> 
    void bob::ip::Gaussian::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst)
    {
      blitz::Array<double,2> src_d = bob::core::cast<double>(src);
      // Checks are postponed to the convolution function.
      if(m_conv_border == bob::sp::Extrapolation::Zero)
      {
        m_tmp_int.resize(bob::sp::getConvSepOutputSize(src_d, m_kernel_y, 0, bob::sp::Conv::Same));
        bob::sp::convSep(src_d, m_kernel_y, m_tmp_int, 0, bob::sp::Conv::Same);
        bob::sp::convSep(m_tmp_int, m_kernel_x, dst, 1, bob::sp::Conv::Same);
      }
      else
      {
        m_tmp_int1.resize(bob::sp::getConvSepOutputSize(src_d, m_kernel_y, 0, bob::sp::Conv::Full));
        if(m_conv_border == bob::sp::Extrapolation::NearestNeighbour)
          bob::sp::extrapolateNearest(src_d, m_tmp_int1);
        else if(m_conv_border == bob::sp::Extrapolation::Circular)
          bob::sp::extrapolateCircular(src_d, m_tmp_int1);
        else
          bob::sp::extrapolateMirror(src_d, m_tmp_int1);
        m_tmp_int.resize(bob::sp::getConvSepOutputSize(m_tmp_int1, m_kernel_y, 0, bob::sp::Conv::Valid));
        bob::sp::convSep(m_tmp_int1, m_kernel_y, m_tmp_int, 0, bob::sp::Conv::Valid);

        m_tmp_int2.resize(bob::sp::getConvSepOutputSize(m_tmp_int, m_kernel_x, 1, bob::sp::Conv::Full));
        if(m_conv_border == bob::sp::Extrapolation::NearestNeighbour)
          bob::sp::extrapolateNearest(m_tmp_int, m_tmp_int2);
        else if(m_conv_border == bob::sp::Extrapolation::Circular)
          bob::sp::extrapolateCircular(m_tmp_int, m_tmp_int2);
        else
          bob::sp::extrapolateMirror(m_tmp_int, m_tmp_int2);
        bob::sp::convSep(m_tmp_int2, m_kernel_x, dst, 1, bob::sp::Conv::Valid);
      }
    }

    template <typename T> 
    void bob::ip::Gaussian::operator()(const blitz::Array<T,3>& src, 
      blitz::Array<double,3>& dst)
    {
      for( int p=0; p<dst.extent(0); ++p) {
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        
        // Gaussian smooth plane
        this(src_slice, dst_slice);
      }
    }

	}
}

#endif
