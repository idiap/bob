#ifndef _TORCHVISION_IP_SHIFT_H_
#define _TORCHVISION_IP_SHIFT_H_

#include "core/ipCore.h"		// <ipShift> is a <Torch::ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipShift
	//	This class is designed to shift an image.
	//	The result is a tensor of the same storage type and the same size.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"shiftx"	int	0	"variation on Ox axis"
	//		"shifty"	int	0	"variation on Oy axis"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipShift : public ipCore
	{
	public:

		// Constructor
		ipShift();

		// Destructor
		virtual ~ipShift();

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
