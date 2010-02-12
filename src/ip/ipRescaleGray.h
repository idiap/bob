#ifndef _TORCH5SPRO_IP_RESCALE_GRAY_H_
#define _TORCH5SPRO_IP_RESCALE_GRAY_H_

#include "ipCore.h"		// <ipRescaleGray> is a <Torch::ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipRescaleGray
	//	This class is designed to rescale any Tensor into a "short" image (0 to 255).
	//	The result is thus a short tensor
	//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipRescaleGray : public ipCore
	{
	public:

		// Constructor
		ipRescaleGray();

		// Destructor
		virtual ~ipRescaleGray();

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

	};
}

#endif

