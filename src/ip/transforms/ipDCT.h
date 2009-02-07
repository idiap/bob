#ifndef _TORCHSPRO_IP_DCT_H_
#define _TORCHSPRO_IP_DCT_H_

#include "Tensor.h"
#include "ipCore.h"		// <ipCrop> is a <Torch::ipCore>
#include "vision.h"		// <sRect2D> definition

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipDCT
	//	This class is designed to perform DCT.
	//	The result is a tensor of the same storage type.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipDCT : public ipCore
	{
	public:

		// Constructor
		ipDCT(bool inverse_ = false);

		// Destructor
		virtual ~ipDCT();

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
		int H;
		int W;

		FloatTensor *R;
	};
}

#endif
