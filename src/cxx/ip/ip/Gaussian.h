/**
 * @file src/cxx/ip/ip/Gaussian.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a smooth an image with a Gaussian kernel
 */

#ifndef TORCH5SPRO_GAUSSIAN_H
#define TORCH5SPRO_GAUSSIAN_H

#include "core/array_assert.h"
#include "core/cast.h"
#include "sp/convolution.h"
#include "ip/Exception.h"

namespace Torch {

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
          const double sigma=5.,
          const enum Torch::sp::Convolution::SizeOption size_opt =
            Torch::sp::Convolution::Same,
          const enum Torch::sp::Convolution::BorderOption border_opt =
            Torch::sp::Convolution::Mirror)
  			{
          m_radius_x    = radius_x;
          m_radius_y    = radius_y;
          m_sigma       = sigma;
          m_conv_size   = size_opt;
          m_conv_border = border_opt;

          computeKernel();
        }

        /**
         * @brief Process a 2D blitz Array/Image
         * @param src The 2D input blitz array
         * @param src The 2D input blitz array
         */
        template <typename T> 
        void operator()(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst);

        /**
         * @brief Process a 3D blitz Array/Image
         * @param src The 3D input blitz array
         * @param src The 3D input blitz array
         */
        template <typename T> 
        void operator()(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst);

      private:
        void computeKernel(); 

        /**
         * @brief Attributes
         */	
        int m_radius_x;
        int m_radius_y;
        int m_sigma;
        enum Torch::sp::Convolution::SizeOption m_conv_size;
        enum Torch::sp::Convolution::BorderOption m_conv_border;

        blitz::Array<double, 2> m_kernel;
    };

    template <typename T> 
    void Torch::ip::Gaussian::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<T,2>& dst)
    {
      // Checks are postponed to the convolution function.
      Torch::sp::convolve(src, m_kernel, dst, m_conv_size, m_conv_border);
    }

    template <typename T> 
    void Torch::ip::Gaussian::operator()(const blitz::Array<T,3>& src, 
      blitz::Array<T,3>& dst)
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
