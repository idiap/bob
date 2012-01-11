/**
 * @file cxx/ip/ip/ipShift.h
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
#ifndef _BOBVISION_IP_SHIFT_H_
#define _BOBVISION_IP_SHIFT_H_

#include "ip/ipCore.h"		// <ipShift> is a <bob::ipCore>

namespace bob
{
	/////////////////////////////////////////////////////////////////////////
	// bob::ipShift
	//	This class is designed to shift an image.
	//	The result is a tensor of the same storage type and the same size.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"shiftx"	int	0	"variation on Ox axis"
	//		"shifty"	int	0	"variation on Oy axis"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipShift : public ipCore
	{
	public:

		// Constructor
		ipShift();

		// Destructor
		virtual ~ipShift();

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
