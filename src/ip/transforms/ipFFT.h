#ifndef _TORCHVISION_IP_FFT_H_
#define _TORCHVISION_IP_FFT_H_

#include "Tensor.h"
#include "ipCore.h"		// <ipCrop> is a <Torch::ipCore>
#include "vision.h"		// <sRect2D> definition

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipFFT
	//	This class is designed to perform FFT.
	//	The result is a tensor of the same storage type.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipFFT : public ipCore
	{
	public:

		// Constructor
		ipFFT(bool inverse_ = false);

		// Destructor
		virtual ~ipFFT();

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

		bool inverse;

		int N;
		int H,W;

		FloatTensor *R;
		FloatTensor *I;
		DoubleTensor *tmp1;
		DoubleTensor *tmp2;
		FloatTensor *T;
	};
}

#endif
