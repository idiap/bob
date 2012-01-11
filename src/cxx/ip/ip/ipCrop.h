/**
 * @file cxx/ip/ip/ipCrop.h
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
#ifndef _BOBVISION_IP_CROP_H_
#define _BOBVISION_IP_CROP_H_

#include "ip/ipCore.h"		// <ipCrop> is a <bob::ipCore>

namespace bob
{
	/////////////////////////////////////////////////////////////////////////
	// bob::ipCrop
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
