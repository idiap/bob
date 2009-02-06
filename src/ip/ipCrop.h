#ifndef _TORCHVISION_IP_CROP_H_
#define _TORCHVISION_IP_CROP_H_

#include "ipCore.h"		// <ipCrop> is a <Torch::ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipCrop
	//	This class is designed to crop an image.
	//	The result is a tensor of the same storage type.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"x"	int	0	"Ox coordinate of the top left corner of the cropping area"
	//		"y"	int	0	"Oy coordinate of the top left corner of the cropping area"
	//		"w"	int	0	"desired width of the cropped image"
	//		"h"	int	0	"desired height of the cropped image"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipCrop : public ipCore
	{
	public:

		// Constructor
		ipCrop();

		// Destructor
		virtual ~ipCrop();

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
