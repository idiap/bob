/**
 * @file src/cxx/ip/ip/gaussian.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:Niklas.Johansson@idiap.ch">Niklas Johansson</a> 
 *
 * @brief This file provides a class to extract DCT features as described in:
 *
 * This class is designed to smooth an image
 * by convolving a NxM Gaussian filter.
 * The result will be a image having the same number of planes.
 * 
 * \TEX{
 * \begin{eqnarray}
 * O(x,y) = G(x,y) * I(x,y)
 * G(x,y) = \frac{exp(x^2 + y^2)}{\sigma}
 * \end{eqnarray}
 * 
 * \begin{itemize}
 * \li $I(x,y)$ is the input image,
 * \li $O(x,y)$ is the output image,
 * \li $I,O \in {{\rm I \kern -0.2em R}}^w \times {{\rm I \kern -0.2em R}}^h$,
 * \li $w$ is the width of the image and $h$ is the height of the image,
 * \li $\sigma$ is the variance of the kernel,
 * \li * is the convolution operator.
 * \end{itemize}
 * 
 * - PARAMETERS (name, type, default value, description):
 * "RadiusX"	int	1	"Kernel radius on Ox"
 * "RadiusY"	int	1	"Kernel radius on Oy"
 * "Sigma"		double	0.25	"Variance of the kernel"
 * 
 */

#ifndef GAUSSIAN_H__
#define GAUSSIAN_H__

#include "core/cast.h"
#include "ip/Exception.h"

#include "ip/block.h"

namespace Torch {

	/**
	 * \ingroup libip_api
	 * @{
	 *
	 */
	namespace ip {

		class GaussianSmooth
		{
		public:

			GaussianSmooth(const int radius_x, const int radius_y, const double sigma = 0.25)
			{
				m_radius_x = radius_x;
				m_radius_y = radius_y;
				m_rigma   = sigma;
			  
				m_kernel = blitz::Array<double, 2>(2 * radius_y + 1, 2 * radius_x + 1);
				compute_kernel();
			}

			/**
			 * @brief Process a 2D blitz Array/Image
			 * @param src The 2D input blitz array
			 * @param src The 2D input blitz array
			 */
			template <typename T, typename U> 
			void operator()(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst) 
			{
				const int height = src.extent(0);
				const int width  = src.extent(1);

				const int start_x = radius_x;
				const int start_y = radius_y;
				const int stop_x  = width - radius_x;
				const int stop_y  = height - radius_y;

				// Fill with 0 the output image (to clear boundaries)
				dst.zeros();

				// apply kernel
				for (int y = start_y; y < stop_y; y ++) {
					for (int x = start_x; x < stop_x; x ++) {
						// Apply the kernel for the <y, x> pixel
						double sum = 0.0;
						for (int yy = -radius_y; yy <= radius_y; yy ++) {
							for (int xx = -radius_x; xx <= radius_x; xx ++) {
								sum += 	m_kernel(yy + radius_y, xx + radius_x) * src(y + yy, x + xx);
							}
						}
						
						dst(y,x) = static_cast<T>(sum);
					}
				}
			}

			/**
			 * @brief Process a 3D blitz Array/Image
			 * @param src The 3D input blitz array
			 * @param src The 3D input blitz array
			 */
			template <typename T, typename U> 
			void operator()(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst) 
			{
				for( int p=0; p<dst.extent(0); ++p) {
					const blitz::Array<T,2> src_slice = 
						src( p, blitz::Range::all(), blitz::Range::all() );
					blitz::Array<T,2> dst_slice = 
						dst( p, blitz::Range::all(), blitz::Range::all() );
					
					// Gaussian smooth plane
					this(src_slice, dst_slice);
			}

		private:
			void compute_kernel() 
			{
				// compute the kernel
				const double inv_sigma = 1.0  / sigma;
				double sum = 0.0;
		
				for (int i = -radius_x; i <= radius_x; i++) {
					for (int j = -radius_y; j <= radius_y; j ++) {
						const double weight = exp(- inv_sigma * (i * i + j * j));

						m_kernel(j + radius_y, i + radius_x) = weight;
						sum += weight;
					}
				}

				// normalize the kernel
				const double inv_sum = 1.0 / sum;
				for (int i = -radius_x; i <= radius_x; i++)
					for (int j = -radius_y; j <= radius_y; j ++)
						m_kernel(j + radius_y, i + radius_x) *= inv_sum;
			}

			int m_radius_x = radius_x;
			int m_radius_y = radius_y;
			int m_rigma   = sigma;
			  
			blitz::Array<double, 2> m_kernel;
		};
	}
}

#endif



