#ifndef _TORCHVISION_IP_INTEGRAL_IMAGE_H_
#define _TORCHVISION_IP_INTEGRAL_IMAGE_H_

#include "ipCore.h"		// <ipIntegralImage> is a <Torch::ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipIntegralImage
	//	This class is designed to compute the integral image of an image
	//		the result is one 3D IntTensor having the same dimensions as the image.
	//	The integral image is computed for each color channel.

	//	\begin{equation}
	//		II(x,y) = \sum_{i=1}^{x-1} \sum_{j=1}^{y-1} I(i,j)
	//	\end{equation}
	//
    	//	\begin{verbatim}
        //		+---+          +--------------+         +---+
	//		|XXX|	       |              |         |XXX|
	//		|XXX|   ---->  |  ipIdentity  | ---->   |XXX|
	//		|XXX|          |              |         |XXX|
	//		+---+          +--------------+         +---+
	//		image                                integral image
	//	\end{verbatim}
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipIntegralImage : public ipCore
	{
	public:

		// Constructor
		ipIntegralImage();

		// Destructor
		virtual ~ipIntegralImage();

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

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
