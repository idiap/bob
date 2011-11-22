/**
 * @file cxx/old/ap/ap/apMFCC.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCHSPRO_AP_MFCC_H_
#define _TORCHSPRO_AP_MFCC_H_

#include "core/Tensor.h"
#include "ap/apCore.h"

namespace Torch
{
/**
 * \ingroup libap_api libAP API
 * @{
 *
 */

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

/**
 * @}
 */

}

#endif
