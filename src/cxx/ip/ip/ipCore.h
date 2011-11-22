/**
 * @file cxx/ip/ip/ipCore.h
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
#ifndef IPCORE_INC
#define IPCORE_INC

#include "sp/spCore.h"

namespace Torch
{
/**
 * \defgroup libip_api libIP API
 * @{
 *
 *  The libIP API.
 */

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::ipCore:
	//	- image processing interface
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class ipCore : public spCore
	{
	public:
		/// Constructor
		ipCore();

		/// Destructor
		virtual ~ipCore();

	protected:

		/////////////////////////////////////////////
		/// Attributes

		//
	};

/**
 * @}
 */

}


/**
@page libIP IP: an Image Processing module

@section intro Introduction

IP is the Image Processing module of Torch.

@section api Documentation
- @ref libip_api "libIP API"

*/

#endif
