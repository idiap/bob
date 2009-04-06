#ifndef _TORCH5SPRO_IP_HAAR_H_
#define _TORCH5SPRO_IP_HAAR_H_

#include "ipCore.h"
#include "Tensor.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipHaar
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipHaar : public ipCore
	{
	public:

		// Constructor
		ipHaar(int width_, int height_, int type_, int x_, int y_, int w_, int h_);

		// Destructor
		virtual ~ipHaar();

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		int width;
		int height;
		int type;
		int x;
		int y;
		int w;
		int h;

  		DoubleTensor *t_;
  		DoubleTensor *t__;
	};
}

#endif



