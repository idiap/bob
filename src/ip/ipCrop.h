#ifndef _TORCHVISION_IP_CROP_H_
#define _TORCHVISION_IP_CROP_H_

#include "ipCore.h"		// <ipCrop> is a <Torch::ipCore>
#include "vision.h"		// <sRect2D> definition

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipCrop
	//	This class is designed to crop an image.
	//	The result is a tensor of the same storage type.
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

		/// Change the region to crop from
		bool			setCropArea(int x, int y, int w, int h);
		bool			setCropArea(const sRect2D& area);

		/// Retrieve the cropping area
		const sRect2D&		getCropArea() const;

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

		sRect2D			m_cropArea;	// Area to crop the image from
	};
}

#endif
