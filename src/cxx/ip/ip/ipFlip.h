#ifndef _TORCHVISION_IP_FLIP_H_
#define _TORCHVISION_IP_FLIP_H_

#include "ip/ipCore.h"		// <ipFlip> is a <Torch::ipCore>
#include "ip/vision.h"		// <sRect2D> definition

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipFlip
	//	This class is designed to crop an image.
	//	The result is a tensor of the same storage type.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"vertical"	bool	false	"direction of the flipping (default vertical)"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipFlip : public ipCore
	{
	public:

		// Constructor
		ipFlip();

		// Destructor
		virtual ~ipFlip();

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

		//
	};
}

#endif
