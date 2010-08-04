#ifndef _TORCHSPRO_AP_MFCC_H_
#define _TORCHSPRO_AP_MFCC_H_

#include "core/Tensor.h"
#include "core/apCore.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::apMFCC
	//	This class is designed to compute Mel Frequency Cepstral Coefficients (MFCC).
	//	The result is a FloatTensor.
	//
	//	http://en.wikipedia.org/wiki/Cepstral
	//	http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class apMFCC : public apCore
	{
	public:

		// Constructor
		apMFCC();

		// Destructor
		virtual ~apMFCC();

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
