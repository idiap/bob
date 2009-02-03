#ifndef _TORCHVISION_IP_SOBEL_H_
#define _TORCHVISION_IP_SOBEL_H_

#include "ipCore.h"		// <ipSobel> is a <Torch::ipCore>
#include "vision.h"		// <sRect2D> definition
#include "Tensor.h"
namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSobel
	//	This class is designed to convolve a sobel mask with an image.
	//	The result is a tensor of the INT storage type.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSobel : public ipCore
	{
	public:

		// Constructor
		ipSobel();

		// Destructor
		virtual ~ipSobel();



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
        int width;
        int height;
        IntTensor *Sx;
        IntTensor *Sy;

        void createMask();

	};
}

#endif
