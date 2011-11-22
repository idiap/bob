/**
 * @file cxx/ip/ip/ipSobel.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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
#ifndef _TORCHVISION_IP_SOBEL_H_
#define _TORCHVISION_IP_SOBEL_H_

#include "ip/ipCore.h"		// <ipSobel> is a <Torch::ipCore>
#include "ip/vision.h"		// <sRect2D> definition
#include "core/Tensor.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSobel
	//	This class is designed to convolve a sobel mask with an image.
	//	The result is 3 tensors of the INT storage type:
	//		the Ox gradient, the Oy gradient and the edge magnitude.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSobel : public ipCore
	{
	public:

		// Constructor
		ipSobel();

		// Destructor
		virtual ~ipSobel();

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

		IntTensor *Sx;
		IntTensor *Sy;

		void createMask();

	};
}

#endif
