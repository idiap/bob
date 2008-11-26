#ifndef _TORCHVISION_IP_LBP_4R_H_
#define _TORCHVISION_IP_LBP_4R_H_

#include "ipLBP.h"		// <ipLBP4R> is an <ipLBP>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipLBP4R
	//	This class implements LBP4R operators, where R is the radius.
	//	Doesn't use the "Uniform" and "RotInvariant" boolean options.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipLBP4R : public ipLBP
	{
	public:

		// Constructor
		ipLBP4R(int R = 1);

		// Destructor
		virtual ~ipLBP4R();

		// Get the maximum possible label
		virtual int		getMaxLabel();

		/////////////////////////////////////////////////////////////////

	protected:

		//////////////////////////////////////////////////////////

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
