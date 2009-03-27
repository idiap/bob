#ifndef _TORCHSPRO_SP_DELTA_ONE_H_
#define _TORCHSPRO_SP_DELTA_ONE_H_

#include "Tensor.h"
#include "spCore.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::spDeltaOne
	//	This class applies the delta function (identity) and return a single value.
	//	The result is a tensor of the same storage type.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class spDeltaOne : public spCore
	{
	public:

		// Constructor
		spDeltaOne(int n_);

		// Destructor
		virtual ~spDeltaOne();

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

		int n;
	};
}

#endif
