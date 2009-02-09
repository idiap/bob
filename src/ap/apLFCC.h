#ifndef _TORCHSPRO_AP_LFCC_H_
#define _TORCHSPRO_AP_LFCC_H_

#include "Tensor.h"
#include "apCore.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::apLFCC
	//	This class is designed to compute Linear Frequency Cepstral Coefficients (LFCC).
	//	The result is a FloatTensor.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class apLFCC : public apCore
	{
	public:

		// Constructor
		apLFCC();

		// Destructor
		virtual ~apLFCC();

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
