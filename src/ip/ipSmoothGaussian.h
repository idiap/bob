#ifndef _TORCHVISION_IP_SMOOTH_GAUSSIAN_H_
#define _TORCHVISION_IP_SMOOTH_GAUSSIAN_H_

#include "core/ipCore.h"		// <ipSmoothGaussian> is a <Torch::ipCore>

namespace Torch {

	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSmoothGaussian
	//	This class is designed to smooth an image (3D ShortTensor)
	//		by convolving a NxM Gaussian filter.
	//	The result will be a 3D ShortTensor image having the same number of planes.
	//
	//	\TEX{
	//	\begin{eqnarray}
	//		O(x,y) = G(x,y) * I(x,y)
	//		G(x,y) = \frac{exp(x^2 + y^2)}{\sigma}
	//	\end{eqnarray}
	//
	//	\begin{itemize}
	//		\item $I(x,y)$ is the input image,
	//		\item $O(x,y)$ is the output image,
	//		\item $I,O \in {{\rm I \kern -0.2em R}}^w \times {{\rm I \kern -0.2em R}}^h$,
	//		\item $w$ is the width of the image and $h$ is the height of the image,
	//		\item $\sigma$ is the variance of the kernel,
	//		\item * is the convolution operator.
	//	\end{itemize}
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"RadiusX"	int	1	"Kernel radius on Ox"
	//		"RadiusY"	int	1	"Kernel radius on Oy"
	//		"Sigma"		double	0.25	"Variance of the kernel"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class DoubleTensor;

	class ipSmoothGaussian : public ipCore
	{
	public:

		// Constructor
		ipSmoothGaussian();

		// Destructor
		virtual ~ipSmoothGaussian();

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
