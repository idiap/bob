/**
 * @file cxx/ip/ip/ipMSRSQIGaussian.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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
#ifndef _TORCH5SPRO_IP_MSR_SQI_GAUSSIAN_H_
#define _TORCH5SPRO_IP_MSR_SQI_GAUSSIAN_H_

#include "ip/ipCore.h"		// <ipMSRSQIGaussian> is a <Torch::ipCore>

namespace Torch {

	/////////////////////////////////////////////////////////////////////////
	// Torch::ipMSRSQIGaussian
	//	This class is designed to apply a normalize Gaussian or Weighed Gaussian filtering 
	//	        to an image (3D ShortTensor)
	//		by convolving a NxM Gaussian or Weighed Gaussianfilter.
	//      The implementation is descrined in:
	//        \verbatim
	//            "Face Recognition under Varying Lighting conditions using Self Quotient Image"
	//                Wang, Li and Wang, 2004
	//        \endverbatim
	//        for the SelfQuotientImage.
	//
	//
	//	In particular, it performs:
	//		- Normalization by the area of the filter
	//		- Mirror interpolation at the border
	//	The result will be a 3D ShortTensor image having the same number of planes.
	//
	//	\TEX{
	//	\begin{eqnarray}
	//		O(x,y) = F(x,y) * I(x,y)
	//		F(x,y) = Regular or Weighed Gaussian Filter described by Wang, Li and Wang
	//	\end{eqnarray}
	//
	//	\begin{itemize}
	//		\li $I(x,y)$ is the input image,
	//		\li $O(x,y)$ is the output image,
	//		\li $I,O \in {{\rm I \kern -0.2em R}}^w \times {{\rm I \kern -0.2em R}}^h$,
	//		\li $w$ is the width of the image and $h$ is the height of the image,
	//		\li $\sigma$ is the variance of the kernel,
	//		\li * is the convolution operator.
	//	\end{itemize}
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"RadiusX"	int	1	"Kernel radius on Ox"
	//		"RadiusY"	int	1	"Kernel radius on Oy"
	//		"Sigma"		double	0.25	"Variance of the kernel"
	//		"Weighed"	bool	false	"Weighed Gaussian or regular one"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipMSRSQIGaussian : public ipCore
	{
	public:

		// Constructor
		ipMSRSQIGaussian();

		// Destructor
		virtual ~ipMSRSQIGaussian();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		//////////////////////////////////////////////////////////

	private:

		// Allocate and compute the Gaussian kernel
		void			prepareKernel(int radius_x, int radius_y, double sigma);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// The pre-computed 3x3 kernel
		DoubleTensor* 		m_kernel;
	};
}

#endif

